"""
dataset_split.py
────────────────────────────────────────────────────────────────────────────
Step 2：数据集分层划分（train / val / test = 7 : 2 : 1）

功能：
  1. 读取 data/labeled/masks/ 下全部 mask，提取每张图的类别组合指纹
  2. 基于类别组合指纹做分层划分，确保稀有类别（如 Alligator）
     在 train / val / test 三个子集中均有覆盖
  3. 输出 data/processed/splits/ 下的 train.txt / val.txt / test.txt
  4. 打印并保存划分统计报告

分层策略：
  - 每张图像的「标签指纹」= 其包含的类别 ID 集合转为有序元组
    例：只含横向裂缝和修补 → (1, 5)
  - 使用 scikit-learn StratifiedShuffleSplit，以指纹作为分层标签
  - 若某指纹类别样本数 < 2（无法分层），自动合并入最近邻指纹桶

运行方式：
  python src/data_processing/dataset_split.py
  python src/data_processing/dataset_split.py \
      --image_dir data/labeled/images \
      --mask_dir  data/labeled/masks  \
      --output_dir data/processed/splits \
      --val_ratio 0.2 --test_ratio 0.1 --seed 42
────────────────────────────────────────────────────────────────────────────

Step 2 的几个关键设计说明：
指纹合并机制（merge_rare_fingerprints）：StratifiedShuffleSplit 要求每种分层标签至少有 2 个样本才能同时出现在 train 和 test 中。
数据集里极个别的图像可能有罕见的类别组合（比如同时含 Oblique + Alligator 的图只有 1 张），这种情况下用 Jaccard 相似度把它归并到最相近的指纹桶，保证划分不报错。

两步划分顺序：先切出 test，再从剩余部分切出 val，每步独立做分层，比一次性三路划分更稳定，尤其是在样本量不大（1289张）时。

额外输出 split_meta.json：除了三个 .txt 文件，还保存了包含 image_name 和 mask_name 的完整元数据文件，Step 3（数据增强）和 Step 4（Dataset 类）会直接读取它，
不需要再重新扫描目录。
"""

import os
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit


plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# ─── 全局常量 ────────────────────────────────────────────────────────────────

CLASS_INFO = {
    0: "background",
    1: "Transverse",
    2: "Longitudinal",
    3: "Alligator",
    4: "fixpatch",
}
NUM_CLASSES = 5


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def get_label_fingerprint(mask: np.ndarray) -> tuple:
    """
    提取单张 mask 的类别组合指纹。
    指纹 = 该图中出现的所有类别 ID（排除 background=0）的有序元组。
    例：包含横向裂缝(1) 和修补(5) → (1, 5)
    """
    present = sorted(int(c) for c in np.unique(mask) if c != 0)
    return tuple(present) if present else (0,)   # 纯背景图用 (0,) 标记


def collect_fingerprints(image_dir: Path, mask_dir: Path) -> list[dict]:
    """
    遍历 image_dir，对每个文件名在 mask_dir 中找同名 mask，
    提取类别指纹。

    Returns
    -------
    records : list of dict，每个元素包含：
        stem        : 文件名（不含后缀），如 "000001"
        image_name  : 图像文件名，如 "000001.jpg"
        mask_name   : mask 文件名，如 "000001.png"
        fingerprint : tuple，类别组合指纹
    """
    # 支持的图像格式
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in img_exts
    )
    if not image_paths:
        raise FileNotFoundError(f"在 {image_dir} 下未找到图像文件")

    records = []
    missing_masks = []

    for img_path in tqdm(image_paths, desc="提取类别指纹", unit="img"):
        mask_path = mask_dir / (img_path.stem + ".png")
        if not mask_path.exists():
            missing_masks.append(img_path.stem)
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [警告] 无法读取 mask：{mask_path.name}，已跳过")
            continue

        fp = get_label_fingerprint(mask)
        records.append({
            "stem":        img_path.stem,
            "image_name":  img_path.name,
            "mask_name":   mask_path.name,
            "fingerprint": fp,
        })

    if missing_masks:
        print(f"\n  [警告] 以下 {len(missing_masks)} 张图像找不到对应 mask，已跳过：")
        for s in missing_masks[:10]:
            print(f"    {s}")
        if len(missing_masks) > 10:
            print(f"    ... 共 {len(missing_masks)} 个")

    print(f"\n  有效样本对：{len(records)} 张（原始图像 {len(image_paths)} 张）")
    return records


# ─── 分层划分 ─────────────────────────────────────────────────────────────────

def merge_rare_fingerprints(records: list[dict], min_count: int = 2) -> list[dict]:
    """
    StratifiedShuffleSplit 要求每个分层标签至少有 2 个样本。
    对样本数 < min_count 的指纹，将其合并到最相似的较大指纹桶中。

    合并策略：Jaccard 相似度最高的已有指纹桶，若无则归入 "rare_misc"。
    """
    fp_counter = Counter(r["fingerprint"] for r in records)

    # 找出需要合并的稀有指纹
    rare_fps  = {fp for fp, cnt in fp_counter.items() if cnt < min_count}
    major_fps = [fp for fp, cnt in fp_counter.items() if cnt >= min_count]

    if not rare_fps:
        return records   # 无需合并

    print(f"\n  [分层合并] 发现 {len(rare_fps)} 个稀有指纹（样本数 < {min_count}），"
          f"将合并至相邻桶：")

    def jaccard(a: tuple, b: tuple) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        return len(sa & sb) / len(sa | sb)

    # 构建重映射表
    remap = {}
    for fp in rare_fps:
        if major_fps:
            best = max(major_fps, key=lambda m: jaccard(fp, m))
            remap[fp] = best
            print(f"    {fp}  →  {best}  (Jaccard={jaccard(fp, best):.2f})")
        else:
            remap[fp] = ("rare_misc",)
            print(f"    {fp}  →  rare_misc")

    # 应用重映射（仅修改分层用的 fingerprint 字段，不影响实际文件名）
    updated = []
    for r in records:
        r2 = r.copy()
        if r["fingerprint"] in remap:
            r2["fingerprint"] = remap[r["fingerprint"]]
        updated.append(r2)

    return updated


def split_dataset(
    records:    list[dict],
    val_ratio:  float = 0.2,
    test_ratio: float = 0.1,
    seed:       int   = 42,
) -> tuple[list, list, list]:
    """
    两步分层划分：
      Step A：从全量中分离出 test（比例 = test_ratio）
      Step B：从剩余中分离出 val（比例 = val_ratio / (1 - test_ratio)）

    Returns
    -------
    train_records, val_records, test_records
    """
    records_arr = np.array(records, dtype=object)
    labels = np.array([np.bincount(r["fingerprint"], minlength=NUM_CLASSES) for r in records])
    
    # ── Step A：划出 test ────────────────────────────────────────────────────
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed
    )
    train_val_idx, test_idx = next(sss_test.split(records_arr, labels))

    train_val_records = records_arr[train_val_idx].tolist()
    test_records      = records_arr[test_idx].tolist()
    labels_tv         = labels[train_val_idx]

    # ── Step B：从 train+val 中划出 val ─────────────────────────────────────
    # 调整 val 比例：原始 val_ratio 是相对全量的，需换算为相对 train+val 的比例
    adjusted_val = val_ratio / (1.0 - test_ratio)

    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=adjusted_val, random_state=seed
    )
    train_idx, val_idx = next(
        sss_val.split(np.array(train_val_records, dtype=object), labels_tv)
    )

    train_val_arr = np.array(train_val_records, dtype=object)
    train_records = train_val_arr[train_idx].tolist()
    val_records   = train_val_arr[val_idx].tolist()

    return train_records, val_records, test_records


# ─── 写入 txt ─────────────────────────────────────────────────────────────────

def write_split_files(
    train_records: list,
    val_records:   list,
    test_records:  list,
    output_dir:    Path,
) -> None:
    """
    每行写入文件名（不含路径和后缀），例如：
        000001
        000045
        ...
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, records in [
        ("train.txt", train_records),
        ("val.txt",   val_records),
        ("test.txt",  test_records),
    ]:
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for r in sorted(records, key=lambda x: x["stem"]):
                f.write(r["stem"] + "\n")
        print(f"  [✓] {name:10s}  {len(records):5d} 张  →  {path}")


# ─── 统计报告与可视化 ─────────────────────────────────────────────────────────

def compute_split_class_dist(records: list) -> np.ndarray:
    """统计该子集中各前景类别的图像出现次数。"""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for r in records:
        for cls_id in r["fingerprint"]:
            if 0 < cls_id < NUM_CLASSES:
                counts[cls_id] += 1
    return counts


def print_split_report(
    train_records: list,
    val_records:   list,
    test_records:  list,
    total:         int,
) -> None:
    print("\n" + "=" * 65)
    print("  数据集划分结果")
    print("=" * 65)
    print(f"  {'子集':<8} {'数量':>6}  {'占比':>7}")
    print(f"  {'-'*30}")

    subsets = [
        ("train", train_records),
        ("val",   val_records),
        ("test",  test_records),
    ]
    for name, records in subsets:
        pct = len(records) / total * 100
        print(f"  {name:<8} {len(records):>6}  {pct:>6.1f}%")
    print(f"  {'total':<8} {total:>6}  {'100.0%':>7}")

    print("\n  各子集前景类别分布（图像出现次数）：")
    fg_names = [CLASS_INFO[i] for i in range(1, NUM_CLASSES)]
    header = f"  {'类别':<15}" + "".join(f"{'  '+n:<12}" for n in ["train","val","test"])
    print(header)
    print(f"  {'-'*50}")

    for cls_id in range(1, NUM_CLASSES):
        row = f"  {CLASS_INFO[cls_id]:<15}"
        for _, records in subsets:
            dist = compute_split_class_dist(records)
            row += f"  {dist[cls_id]:>8}"
        print(row)
    print("=" * 65 + "\n")


def plot_split_distribution(
    train_records: list,
    val_records:   list,
    test_records:  list,
    output_path:   Path,
) -> None:
    """绘制三个子集各类别图像出现次数的分组柱状图。"""
    fg_ids   = list(range(1, NUM_CLASSES))
    fg_names = [CLASS_INFO[i] for i in fg_ids]

    train_dist = compute_split_class_dist(train_records)[1:]
    val_dist   = compute_split_class_dist(val_records)[1:]
    test_dist  = compute_split_class_dist(test_records)[1:]

    x     = np.arange(len(fg_ids))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_train = ax.bar(x - width, train_dist, width, label="train",
                        color="#3498DB", alpha=0.85, edgecolor="white")
    bars_val   = ax.bar(x,         val_dist,   width, label="val",
                        color="#F39C12", alpha=0.85, edgecolor="white")
    bars_test  = ax.bar(x + width, test_dist,  width, label="test",
                        color="#E74C3C", alpha=0.85, edgecolor="white")

    # 标注数量
    for bars in [bars_train, bars_val, bars_test]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        str(int(h)), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(fg_names, fontsize=10)
    ax.set_ylabel("图像出现次数", fontsize=10)
    ax.set_title(
        f"train / val / test 各类别分布\n"
        f"（共 {len(train_records)+len(val_records)+len(test_records)} 张，"
        f"比例 {len(train_records)}:{len(val_records)}:{len(test_records)}）",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] 划分分布图已保存至：{output_path}")


def save_split_meta(
    train_records: list,
    val_records:   list,
    test_records:  list,
    output_dir:    Path,
) -> None:
    """
    将划分结果保存为 split_meta.json，方便后续步骤（增强、Dataset）读取。
    记录每张图的 stem、image_name、mask_name 和原始指纹。
    """
    meta = {
        "train": [
            {"stem": r["stem"], "image_name": r["image_name"],
             "mask_name": r["mask_name"]}
            for r in sorted(train_records, key=lambda x: x["stem"])
        ],
        "val": [
            {"stem": r["stem"], "image_name": r["image_name"],
             "mask_name": r["mask_name"]}
            for r in sorted(val_records, key=lambda x: x["stem"])
        ],
        "test": [
            {"stem": r["stem"], "image_name": r["image_name"],
             "mask_name": r["mask_name"]}
            for r in sorted(test_records, key=lambda x: x["stem"])
        ],
    }
    meta_path = output_dir / "split_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[✓] 划分元数据已保存至：{meta_path}")


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="路面病害数据集分层划分")
    parser.add_argument("--image_dir",  type=str, default="data/labeled/images")
    parser.add_argument("--mask_dir",   type=str, default="data/labeled/masks")
    parser.add_argument("--output_dir", type=str, default="data/processed/splits")
    parser.add_argument("--chart_dir",  type=str, default="outputs")
    parser.add_argument("--val_ratio",  type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    image_dir  = Path(args.image_dir)
    mask_dir   = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    chart_dir  = Path(args.chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 提取指纹 ──────────────────────────────────────────────────────────
    print(f"\n[Step 1/4] 提取类别指纹...")
    records = collect_fingerprints(image_dir, mask_dir)

    fp_dist = Counter(r["fingerprint"] for r in records)
    print(f"\n  共 {len(fp_dist)} 种类别组合指纹：")
    for fp, cnt in sorted(fp_dist.items(), key=lambda x: -x[1]):
        names = "+".join(CLASS_INFO[c] for c in fp if c != 0) or "background only"
        print(f"    {str(fp):<30}  {cnt:>5} 张  ({names})")

    # ── 2. 合并稀有指纹 ──────────────────────────────────────────────────────
    print(f"\n[Step 2/4] 检查并合并稀有指纹（min_count=2）...")
    records_merged = merge_rare_fingerprints(records, min_count=2)

    # ── 3. 分层划分 ──────────────────────────────────────────────────────────
    print(f"\n[Step 3/4] 执行分层划分（seed={args.seed}）...")
    train_r, val_r, test_r = split_dataset(
        records_merged, args.val_ratio, args.test_ratio, args.seed
    )
    print_split_report(train_r, val_r, test_r, total=len(records))

    # ── 4. 写入文件 ──────────────────────────────────────────────────────────
    print(f"[Step 4/4] 写入划分文件...")
    write_split_files(train_r, val_r, test_r, output_dir)
    save_split_meta(train_r, val_r, test_r, output_dir)

    chart_path = chart_dir / "split_distribution.png"
    plot_split_distribution(train_r, val_r, test_r, chart_path)

    print(f"\n[✓] Step 2 数据集划分全部完成。")
    print(f"    划分文件目录：{output_dir}")
    print(f"    分布图表：   {chart_path}\n")


if __name__ == "__main__":
    main()