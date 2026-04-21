"""
class_statistics.py
────────────────────────────────────────────────────────────────────────────
Step 1：类别像素统计 & 损失权重计算

功能：
  1. 遍历 data/labeled/masks/ 下全部 mask.png
  2. 统计各类别（0~5）像素总量及图像级出现频率
  3. 使用 Median Frequency Balancing 计算类别权重
  4. 输出可视化图表（像素占比 + 图像出现率）
  5. 将权重自动写入 configs/mask2former_crack.yaml

类别映射：
  0: background
  1: Transverse    横向裂缝
  2: Longitudinal  纵向裂缝
  3: Oblique       斜向裂缝
  4: Alligator     龟裂
  5: fixpatch      修补

Median Frequency Balancing 公式：
  freq_c     = (该类别像素总数) / (包含该类别的图像的总像素数)
  weight_c   = median(freq_all_classes) / freq_c
  background 的权重固定为 0（不参与损失计算）

运行方式：
  python src/data_processing/class_statistics.py
  python src/data_processing/class_statistics.py --mask_dir path/to/masks --output_dir path/to/outputs
────────────────────────────────────────────────────────────────────────────
"""

import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")   # 无显示器服务器环境兼容
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# ─── 全局常量 ────────────────────────────────────────────────────────────────

CLASS_INFO = {
    0: {"name": "background",   "color": "#AAAAAA"},
    1: {"name": "damage",       "color": "#E74C3C"},   # 红
}

NUM_CLASSES = len(CLASS_INFO)   # 2


# ─── 核心统计函数 ─────────────────────────────────────────────────────────────

def compute_statistics(mask_dir: Path) -> dict:
    """
    遍历 mask_dir 下所有 .png 文件，统计各类别像素数量。

    Returns
    -------
    stats : dict，包含以下字段：
        pixel_counts    : ndarray (6,)  各类别累计像素总数
        image_counts    : ndarray (6,)  包含该类别的图像数量
        total_pixels    : int           所有图像的总像素数
        num_images      : int           处理的图像总数
        per_image_pixels: list[ndarray] 每张图各类别像素数（用于方差分析）
    """
    mask_paths = sorted(mask_dir.glob("*.png"))
    if len(mask_paths) == 0:
        raise FileNotFoundError(f"在 {mask_dir} 下未找到任何 .png 文件")

    pixel_counts  = np.zeros(NUM_CLASSES, dtype=np.int64)
    image_counts  = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_pixels  = 0
    per_image_pixels = []

    for mask_path in tqdm(mask_paths, desc="统计类别像素", unit="img"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [警告] 无法读取 {mask_path.name}，已跳过")
            continue

        h, w = mask.shape
        total_pixels += h * w

        counts_this = np.zeros(NUM_CLASSES, dtype=np.int64)
        for cls_id in range(NUM_CLASSES):
            cnt = int(np.sum(mask == cls_id))
            pixel_counts[cls_id] += cnt
            counts_this[cls_id] = cnt
            if cnt > 0:
                image_counts[cls_id] += 1

        per_image_pixels.append(counts_this)

    return {
        "pixel_counts":     pixel_counts,
        "image_counts":     image_counts,
        "total_pixels":     total_pixels,
        "num_images":       len(mask_paths),
        "per_image_pixels": per_image_pixels,
    }


# ─── 权重计算 ─────────────────────────────────────────────────────────────────

def compute_weights_median_freq(stats: dict) -> tuple:
    """
    基于前景像素频率计算类别权重。

    freq_c   = 该类别总像素数 / 所有前景类别总像素数（不含background）
    weight_c = median(freq_1..5) / freq_c

    background (cls 0) 权重固定为 0.0
    权重上限截断为 MAX_WEIGHT，避免稀有类别权重过大导致训练坍塌
    """
    pixel_counts = stats["pixel_counts"]   # ndarray (6,)
    MAX_WEIGHT   = 4.0

    # 计算前景总像素数（不含 background）
    foreground_total = sum(
        int(pixel_counts[cls_id])
        for cls_id in range(1, NUM_CLASSES)
    )

    if foreground_total == 0:
        raise ValueError("所有前景类别像素数均为 0，请检查 mask 文件")

    # 计算各前景类别的频率（相对于前景总像素）
    freq = np.zeros(NUM_CLASSES, dtype=np.float64)
    for cls_id in range(1, NUM_CLASSES):
        freq[cls_id] = pixel_counts[cls_id] / foreground_total

    # 取前景类别非零频率的中位数
    foreground_freqs = np.array([
        freq[cls_id]
        for cls_id in range(1, NUM_CLASSES)
        if freq[cls_id] > 0
    ])

    if len(foreground_freqs) == 0:
        raise ValueError("所有前景类别频率均为 0")

    median_freq = float(np.median(foreground_freqs))

    # 计算权重并截断
    weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    weights[0] = 0.0   # background 不参与损失
    for cls_id in range(1, NUM_CLASSES):
        if freq[cls_id] > 0:
            raw_weight        = median_freq / freq[cls_id]
            weights[cls_id]   = float(min(raw_weight, MAX_WEIGHT))
        else:
            weights[cls_id] = 0.0
            print(f"  [警告] 类别 {CLASS_INFO[cls_id]['name']} 不存在于任何 mask 中")

    return weights, freq


# ─── 可视化 ───────────────────────────────────────────────────────────────────

def plot_statistics(stats: dict, weights: np.ndarray, freq: np.ndarray,
                    output_path: Path) -> None:
    """
    生成包含 3 个子图的统计可视化：
      1. 各类别像素占比（饼图）
      2. 各前景类别像素数量（条形图，对数刻度）
      3. 各前景类别计算权重（条形图）
    """
    pixel_counts = stats["pixel_counts"]
    image_counts = stats["image_counts"]
    num_images   = stats["num_images"]

    class_names  = [CLASS_INFO[i]["name"] for i in range(NUM_CLASSES)]
    colors       = [CLASS_INFO[i]["color"] for i in range(NUM_CLASSES)]

    fig, axes = plt.subplots(1, 4, figsize=(26, 7))
    fig.suptitle(
        f"路面病害数据集类别统计  (共 {num_images} 张图像，"
        f"总像素 {stats['total_pixels']:,})",
        fontsize=14, fontweight="bold", y=1.01
    )

    # ── 子图 1：像素占比饼图 ──────────────────────────────────────────────────
    ax1 = axes[0]
    pct = pixel_counts / pixel_counts.sum() * 100
    labels_pie = [
        f"{class_names[i]}\n{pct[i]:.2f}%" if pct[i] >= 0.5 else ""
        for i in range(NUM_CLASSES)
    ]
    wedges, _ = ax1.pie(
        pixel_counts,
        labels=labels_pie,
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax1.set_title("各类别像素占比（含背景）", fontsize=12, pad=12)
    legend_labels = [f"{class_names[i]}  {pct[i]:.3f}%" for i in range(NUM_CLASSES)]
    ax1.legend(wedges, legend_labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.18), fontsize=8.5, ncol=2)

    # ── 子图 1b：不含背景的前景类别像素占比饼图 ──────────────────────────────
    ax1b = axes[1]
    fg_ids_pie    = list(range(1, NUM_CLASSES))
    fg_counts_pie = pixel_counts[1:]
    fg_names_pie  = [class_names[i] for i in fg_ids_pie]
    fg_colors_pie = [colors[i] for i in fg_ids_pie]
    fg_total      = fg_counts_pie.sum()
    
    pct_fg = fg_counts_pie / fg_total * 100 if fg_total > 0 else fg_counts_pie
    labels_pie_fg = [
        f"{fg_names_pie[i]}\n{pct_fg[i]:.2f}%" if pct_fg[i] >= 1.0 else ""
        for i in range(len(fg_ids_pie))
    ]
    wedges_fg, _ = ax1b.pie(
        fg_counts_pie,
        labels=labels_pie_fg,
        colors=fg_colors_pie,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax1b.set_title("各前景类别像素占比（不含背景）", fontsize=12, pad=12)
    legend_labels_fg = [
        f"{fg_names_pie[i]}  {pct_fg[i]:.3f}%"
        for i in range(len(fg_ids_pie))
    ]
    ax1b.legend(wedges_fg, legend_labels_fg, loc="lower center",
                bbox_to_anchor=(0.5, -0.18), fontsize=8.5, ncol=2)
    
    # ── 子图 2：前景类别像素数量（对数坐标）────────────────────────────────────
    ax2 = axes[2]
    fg_ids    = list(range(1, NUM_CLASSES))
    fg_names  = [class_names[i] for i in fg_ids]
    fg_colors = [colors[i] for i in fg_ids]
    fg_counts = [int(pixel_counts[i]) for i in fg_ids]
    fg_img_ct = [int(image_counts[i]) for i in fg_ids]

    bars = ax2.bar(fg_names, fg_counts, color=fg_colors, edgecolor="white",
                   linewidth=1.2, alpha=0.88)
    ax2.set_yscale("log")
    ax2.set_ylabel("像素总数（对数刻度）", fontsize=10)
    ax2.set_title("各前景类别像素数量", fontsize=12)
    ax2.tick_params(axis="x", rotation=15)

    # 在每根柱上标注图像出现次数
    for bar, cnt, img_c in zip(bars, fg_counts, fg_img_ct):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f"{cnt:,}\n({img_c}张图)",
            ha="center", va="bottom", fontsize=8, color="#333333"
        )

    ax2.set_ylim(bottom=max(1, min(fg_counts) // 10))
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # ── 子图 3：类别权重 ──────────────────────────────────────────────────────
    ax3 = axes[3]
    fg_weights = [float(weights[i]) for i in fg_ids]
    bars3 = ax3.bar(fg_names, fg_weights, color=fg_colors, edgecolor="white",
                    linewidth=1.2, alpha=0.88)
    ax3.set_ylabel("类别权重（Median Freq. Balancing）", fontsize=10)
    ax3.set_title("各前景类别损失权重", fontsize=12)
    ax3.tick_params(axis="x", rotation=15)
    ax3.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6,
                label="weight = 1.0")
    ax3.legend(fontsize=9)

    for bar, w in zip(bars3, fg_weights):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{w:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax3.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] 统计图表已保存至：{output_path}")


# ─── YAML 写入 ────────────────────────────────────────────────────────────────

def update_yaml_weights(yaml_path: Path, weights: np.ndarray) -> None:
    """
    将计算好的类别权重写入 configs/mask2former_crack.yaml。
    若 yaml 文件不存在则自动创建完整的配置文件模板。
    若已存在则仅更新 class_weights 字段，保留其余配置不变。
    """
    weights_list = [round(float(w), 6) for w in weights]

    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["class_weights"] = weights_list
    else:
        # 首次运行时自动创建完整配置模板
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "model": {
                "backbone": "swin-base",
                "pretrained": "facebook/mask2former-swin-base-ade-semantic",
                "num_classes": 6,
                "num_queries": 100,
            },
            "classes": {
                0: "background",
                1: "Transverse",
                2: "Longitudinal",
                3: "Oblique",
                4: "Alligator",
                5: "fixpatch",
            },
            "training": {
                "image_size": [2720, 1530],
                "batch_size": 2,
                "accumulation_steps": 4,
                "lr": 1e-4,
                "backbone_lr_multiplier": 0.1,
                "max_epochs": 100,
                "warmup_epochs": 5,
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "num_workers": 4,
                "save_every_n_epochs": 10,
            },
            "physical_constraints": {
                "max_crack_width_px": 20,
                "max_transverse_span_ratio": 0.80,
                "min_contour_complexity": 20.0,
                "min_region_area_px": 50,
                "shadow_aspect_ratio_threshold": 30.0,
            },
            "class_weights": weights_list,
        }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"[✓] 类别权重已写入：{yaml_path}")


# ─── 控制台报告 ───────────────────────────────────────────────────────────────

def print_report(stats: dict, weights: np.ndarray, freq: np.ndarray) -> None:
    pixel_counts = stats["pixel_counts"]
    image_counts = stats["image_counts"]
    total_pixels = stats["total_pixels"]
    num_images   = stats["num_images"]

    print("\n" + "=" * 70)
    print(f"  数据集统计报告  |  图像总数：{num_images}  |  总像素数：{total_pixels:,}")
    print("=" * 70)
    header = f"{'类别ID':>6}  {'类别名称':>14}  {'像素总数':>14}  "
    header += f"{'占比(%)':>8}  {'出现图像数':>10}  {'频率':>10}  {'权重':>8}"
    print(header)
    print("-" * 70)

    for cls_id in range(NUM_CLASSES):
        name     = CLASS_INFO[cls_id]["name"]
        cnt      = int(pixel_counts[cls_id])
        pct      = cnt / total_pixels * 100
        img_cnt  = int(image_counts[cls_id])
        f_val    = freq[cls_id]
        w_val    = weights[cls_id]
        flag     = "  ← background" if cls_id == 0 else ""
        print(
            f"  {cls_id:>4}  {name:>14}  {cnt:>14,}  "
            f"{pct:>8.4f}  {img_cnt:>10}  {f_val:>10.6f}  {w_val:>8.4f}{flag}"
        )

    print("=" * 70)
    print(f"\n  Median Frequency（前景类别）= {float(np.median(freq[1:][freq[1:] > 0])):.6f}")
    print(f"  类别权重向量（写入 yaml）：")
    for cls_id in range(NUM_CLASSES):
        print(f"    [{cls_id}] {CLASS_INFO[cls_id]['name']:>14} : {weights[cls_id]:.6f}")
    print()


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="路面病害数据集类别统计与权重计算")
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="data/labeled/masks",
        help="mask.png 所在目录（默认：data/labeled/masks）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="图表输出目录（默认：outputs）",
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="configs/mask2former_crack.yaml",
        help="配置文件路径（默认：configs/mask2former_crack.yaml）",
    )
    args = parser.parse_args()

    mask_dir   = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    yaml_path  = Path(args.yaml_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 统计 ──────────────────────────────────────────────────────────────
    print(f"\n[Step 1/3] 正在统计 {mask_dir} 下的 mask 文件...")
    stats = compute_statistics(mask_dir)

    # ── 2. 权重计算 ──────────────────────────────────────────────────────────
    print("[Step 2/3] 计算 Median Frequency Balancing 权重...")
    weights, freq = compute_weights_median_freq(stats)
    print_report(stats, weights, freq)

    # ── 3. 可视化 ─────────────────────────────────────────────────────────────
    print("[Step 3/3] 生成统计图表...")
    chart_path = output_dir / "class_statistics.png"
    plot_statistics(stats, weights, freq, chart_path)

    # ── 4. 写入 YAML ──────────────────────────────────────────────────────────
    update_yaml_weights(yaml_path, weights)

    print("\n[✓] Step 1 类别统计与权重计算全部完成。")
    print(f"    统计图表：{chart_path}")
    print(f"    配置文件：{yaml_path}\n")


if __name__ == "__main__":
    main()