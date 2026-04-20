"""
postprocess.py
────────────────────────────────────────────────────────────────────────────
Step 12：后处理与可视化输出

功能：
  读取 predict.py 生成的 inference_results.json，对每张测试图像：
  1. 生成彩色叠加可视化图（预测 mask 半透明叠加在原图上）
  2. 生成预测 vs 真值并排对比图
  3. 统计每张图各类别病害面积（像素数 + 占比），输出 CSV 报表
  4. 在整个测试集上计算并输出最终评估指标（IoU / F1 / Pixel Acc）
  5. 保存归一化混淆矩阵热力图

输出目录结构：
  outputs/predictions/
    raw_masks/              ← predict.py 生成（原始预测）
    filtered_masks/         ← predict.py 生成（物理约束后）
    overlays/               ← 本模块：彩色叠加图
    comparisons/            ← 本模块：预测 vs 真值并排图
    crack_area_report.csv   ← 本模块：病害面积统计表
    test_metrics.json       ← 本模块：测试集评估指标
    confusion_matrix.png    ← 本模块：混淆矩阵热力图

运行方式：
  python src/inference/postprocess.py
  python src/inference/postprocess.py \
      --results_json outputs/predictions/inference_results.json \
      --image_dir    data/labeled/images \
      --output_dir   outputs/predictions \
      --alpha        0.5
────────────────────────────────────────────────────────────────────────────

Step 12 几个关键设计说明：
background 不叠加颜色：render_overlay 中仅对 mask > 0 的前景像素做加权混合，背景区域完全保留原图，视觉上更清晰，不会让整张图被灰色背景色污染。

对比图缩小 50%：2720×1530 的三列并排图宽度会达到 8160px，文件极大且难以显示。缩到 50% 后并排图为 4080×765，文件尺寸合理，仍然清晰可辨。

混淆矩阵仅在有真值时计算：测试集样本来自已标注的 1378 张，所以 gt_mask_path 一定存在；但接口设计上保持了对"无真值"情况的兼容，方便后续对未标注的 6500 张图推理时复用本模块。

CSV 使用 utf-8-sig 编码：fixpatch 类别名含空格，Excel 直接打开 UTF-8 文件有时会乱码，utf-8-sig 带 BOM 标识，Excel 能正确识别。
"""

import argparse
import json
import csv
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from src.training.metrics import (
    ConfusionMatrix,
    print_metrics_report,
    plot_confusion_matrix,
)

plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Zen Hei',
    'WenQuanYi Micro Hei',
    'Noto Sans CJK JP',
    'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

# ─── 常量 ─────────────────────────────────────────────────────────────────────

NUM_CLASSES = 6

CLASSES = {
    0: "background",
    1: "Transverse",
    2: "Longitudinal",
    3: "Oblique",
    4: "Alligator",
    5: "fixpatch",
}
FOREGROUND_IDS = [1, 2, 3, 4, 5]
CLASS_COLORS_BGR = {
    0: (0,   0,   0),
    1: (60,  76,  231),
    2: (219, 152,  52),
    3: (18,  156, 243),
    4: (179,  68, 142),
    5: (39,  174,  96),
}
# matplotlib 用 RGB（归一化到 0~1）
CLASS_COLORS_RGB_NORM = {
    cls_id: tuple(c / 255.0 for c in reversed(bgr))
    for cls_id, bgr in CLASS_COLORS_BGR.items()
}


# ─── 彩色 Mask 渲染 ───────────────────────────────────────────────────────────

def render_color_mask(mask: np.ndarray) -> np.ndarray:
    """
    将单通道语义 mask（0~5）渲染为 BGR 彩色图像。

    Parameters
    ----------
    mask : uint8 ndarray [H, W]，值域 0~5

    Returns
    -------
    color_mask : uint8 ndarray [H, W, 3]，BGR
    """
    H, W = mask.shape
    color_mask = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_id, bgr in CLASS_COLORS_BGR.items():
        color_mask[mask == cls_id] = bgr
    return color_mask


def render_overlay(
    image_bgr:  np.ndarray,
    mask:       np.ndarray,
    alpha:      float = 0.5,
) -> np.ndarray:
    """
    将彩色 mask 以透明度 alpha 叠加在原图上。

    background 类（0）不叠加（完全透明），保持原图像素。

    Parameters
    ----------
    image_bgr : uint8 [H, W, 3]，原始图像
    mask      : uint8 [H, W]，语义 mask
    alpha     : 叠加透明度（0=原图，1=纯 mask）

    Returns
    -------
    overlay : uint8 [H, W, 3]
    """
    color_mask = render_color_mask(mask)
    overlay    = image_bgr.copy()

    # 仅对前景像素叠加颜色
    fg_pixels = mask > 0
    overlay[fg_pixels] = (
        (1 - alpha) * image_bgr[fg_pixels].astype(np.float32)
        + alpha     * color_mask[fg_pixels].astype(np.float32)
    ).astype(np.uint8)

    return overlay


# ─── 图例绘制 ─────────────────────────────────────────────────────────────────

def draw_legend(
    image:   np.ndarray,
    present_classes: list,
    margin:  int = 10,
    box_w:   int = 18,
    box_h:   int = 18,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    在图像右下角绘制类别图例（仅显示当前图中出现的类别）。

    Parameters
    ----------
    image          : uint8 [H, W, 3]，将在原地修改
    present_classes: 当前图中出现的类别 ID 列表（不含 background）
    """
    if not present_classes:
        return image

    img_out    = image.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    thickness  = 1
    line_gap   = box_h + 6

    # 估算图例总高度，决定起始 y
    total_h = len(present_classes) * line_gap + margin
    H, W    = img_out.shape[:2]
    start_y = H - total_h - margin
    start_x = W - 160

    # 半透明背景框
    overlay = img_out.copy()
    cv2.rectangle(
        overlay,
        (start_x - 6, start_y - 6),
        (W - margin, H - margin),
        (30, 30, 30), -1
    )
    cv2.addWeighted(overlay, 0.5, img_out, 0.5, 0, img_out)

    for i, cls_id in enumerate(present_classes):
        y = start_y + i * line_gap
        # 彩色方块
        bgr = CLASS_COLORS_BGR[cls_id]
        cv2.rectangle(img_out,
                      (start_x, y),
                      (start_x + box_w, y + box_h),
                      bgr, -1)
        # 类别名
        cv2.putText(
            img_out,
            CLASSES[cls_id],
            (start_x + box_w + 6, y + box_h - 4),
            font, font_scale,
            (255, 255, 255), thickness,
            cv2.LINE_AA,
        )

    return img_out


# ─── 并排对比图 ───────────────────────────────────────────────────────────────

def make_comparison_figure(
    image_bgr:  np.ndarray,
    pred_mask:  np.ndarray,
    gt_mask:    np.ndarray | None,
    stem:       str,
    alpha:      float = 0.5,
) -> np.ndarray:
    """
    生成左右并排对比图：
      左：原图
      右：推理彩色 mask（纯色，不叠加原图）

    分辨率缩减为 50%（1360×765）后拼图。
    """
    scale = 0.5
    H_s   = int(image_bgr.shape[0] * scale)
    W_s   = int(image_bgr.shape[1] * scale)

    def _resize_img(img):
        return cv2.resize(img, (W_s, H_s), interpolation=cv2.INTER_LINEAR)

    def _resize_mask(m):
        return cv2.resize(m, (W_s, H_s), interpolation=cv2.INTER_NEAREST)

    def _add_title(img, title):
        """在图像顶部添加白色标题栏。"""
        titled = img.copy()
        cv2.rectangle(titled, (0, 0), (W_s, 32), (255, 255, 255), -1)
        cv2.putText(
            titled, title, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (30, 30, 30), 1, cv2.LINE_AA,
        )
        return titled

    # ── 左图：原图 ───────────────────────────────────────────────
    left_panel = _add_title(_resize_img(image_bgr), "Original")

    # ── 右图：纯彩色 mask ─────────────────────────────────────────
    pred_small    = _resize_mask(pred_mask)
    color_mask    = render_color_mask(pred_small)       # 纯彩色，不叠加原图
    pred_present  = [c for c in FOREGROUND_IDS if (pred_small == c).any()]
    color_mask    = draw_legend(color_mask, pred_present)
    right_panel   = _add_title(color_mask, f"Prediction  [{stem}]")

    # ── 左右拼接 ──────────────────────────────────────────────────
    comparison = np.concatenate([left_panel, right_panel], axis=1)

    # ── 在拼接图中间画一条竖向分隔线 ─────────────────────────────
    mid_x = W_s
    cv2.line(comparison, (mid_x, 0), (mid_x, H_s), (200, 200, 200), 2)

    return comparison

def compute_area_stats(
    mask:  np.ndarray,
    stem:  str,
) -> dict:
    """
    统计单张图像中各类别的病害面积（像素数）及其占图像总像素的百分比。

    Returns
    -------
    dict，包含各类别的 pixel_count 和 area_ratio
    """
    H, W      = mask.shape
    total_pix = H * W

    row = {"stem": stem, "image_pixels": total_pix}
    for cls_id in FOREGROUND_IDS:
        cls_name  = CLASSES[cls_id]
        pix_count = int((mask == cls_id).sum())
        ratio     = pix_count / total_pix
        row[f"{cls_name}_pixels"] = pix_count
        row[f"{cls_name}_ratio"]  = round(ratio, 6)

    # 总病害像素（所有前景类别之和）
    total_crack = sum(
        row[f"{CLASSES[c]}_pixels"] for c in FOREGROUND_IDS
    )
    row["total_crack_pixels"] = total_crack
    row["total_crack_ratio"]  = round(total_crack / total_pix, 6)

    return row


# ─── 主处理循环 ───────────────────────────────────────────────────────────────

def run_postprocess(
    results:    list,
    image_dir:  Path,
    output_dir: Path,
    alpha:      float = 0.5,
    device_str: str   = "cpu",
) -> None:
    """
    对所有推理结果执行后处理。

    Parameters
    ----------
    results    : predict.py 生成的 inference_results.json 内容（list[dict]）
    image_dir  : 原始图像目录
    output_dir : 输出根目录
    alpha      : 叠加透明度
    device_str : 用于 ConfusionMatrix 的设备
    """
    import torch
    device = torch.device(device_str)

    overlay_dir    = output_dir / "overlays"
    comparison_dir = output_dir / "comparisons"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    cm         = ConfusionMatrix(num_classes=NUM_CLASSES, device=device)
    area_rows  = []
    has_gt     = False

    for item in tqdm(results, desc="后处理", unit="img"):
        stem           = item["stem"]
        image_path     = Path(item["image_path"])
        filt_mask_path = Path(item["filt_mask_path"])
        gt_mask_path   = Path(item["gt_mask_path"]) \
                         if item.get("gt_mask_path") else None

        # ── 读取图像和预测 mask ───────────────────────────────────────────────
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  [警告] 无法读取图像：{image_path.name}，跳过")
            continue

        pred_mask = cv2.imread(str(filt_mask_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            print(f"  [警告] 无法读取预测 mask：{filt_mask_path.name}，跳过")
            continue
	# 2类 → 6类细分
	from src.inference.crack_direction import classify_damage_full
	pred_mask = classify_damage_full(pred_mask)
	# 输出像素值：0=bg, 1=Transverse, 2=Longitudinal, 3=Oblique, 4=Alligator, 5=fixpatch



        # ── 读取真值 mask（若有）─────────────────────────────────────────────
        gt_mask = None
        if gt_mask_path and gt_mask_path.exists():
            gt_mask  = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
            has_gt   = True

        # ── 1. 叠加可视化图 ───────────────────────────────────────────────────
        pred_present = [c for c in FOREGROUND_IDS if (pred_mask == c).any()]
        overlay      = render_overlay(image, pred_mask, alpha)
        overlay      = draw_legend(overlay, pred_present)
        cv2.imwrite(str(overlay_dir / (stem + ".jpg")), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ── 2. 并排对比图 ─────────────────────────────────────────────────────
        comparison = make_comparison_figure(
            image, pred_mask, gt_mask, stem, alpha
        )
        cv2.imwrite(str(comparison_dir / (stem + ".jpg")), comparison,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ── 3. 病害面积统计 ───────────────────────────────────────────────────
        area_row = compute_area_stats(pred_mask, stem)
        area_rows.append(area_row)

        # ── 4. 更新混淆矩阵（有真值时）───────────────────────────────────────
        if gt_mask is not None:
            pred_t = torch.from_numpy(pred_mask.astype(np.int64)).unsqueeze(0)
            gt_t   = torch.from_numpy(gt_mask.astype(np.int64)).unsqueeze(0)
            cm.update(pred_t.to(device), gt_t.to(device))

    # ── 保存 CSV 报表 ────────────────────────────────────────────────────────
    _save_area_csv(area_rows, output_dir / "crack_area_report.csv")

    # ── 保存评估指标（有真值时）──────────────────────────────────────────────
    if has_gt:
        metrics    = cm.compute()
        metrics_path = output_dir / "test_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n[✓] 测试集评估指标已保存：{metrics_path}")
        print_metrics_report(metrics)

        # 混淆矩阵热力图
        matrix_np = cm.get_matrix_numpy()
        plot_confusion_matrix(
            matrix_np,
            output_dir / "confusion_matrix.png",
            normalize=True,
        )
    else:
        print("\n[提示] 测试集无真值 mask，跳过指标计算和混淆矩阵绘制")

    # ── 汇总打印 ─────────────────────────────────────────────────────────────
    _print_area_summary(area_rows)

    print(f"\n[✓] Step 12 后处理完成。")
    print(f"    叠加可视化图  ：{overlay_dir}  ({len(area_rows)} 张)")
    print(f"    并排对比图    ：{comparison_dir}")
    print(f"    面积统计报表  ：{output_dir / 'crack_area_report.csv'}")
    if has_gt:
        print(f"    评估指标      ：{output_dir / 'test_metrics.json'}")
        print(f"    混淆矩阵      ：{output_dir / 'confusion_matrix.png'}")


# ─── CSV 报表 ─────────────────────────────────────────────────────────────────

def _save_area_csv(rows: list, csv_path: Path) -> None:
    """
    将病害面积统计写入 CSV。

    列顺序：
      stem | image_pixels
      | Transverse_pixels | Transverse_ratio
      | Longitudinal_pixels | Longitudinal_ratio
      | Oblique_pixels | Oblique_ratio
      | Alligator_pixels | Alligator_ratio
      | fixpatch_pixels | fixpatch_ratio
      | total_crack_pixels | total_crack_ratio
    """
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[✓] 病害面积报表已保存：{csv_path}  ({len(rows)} 行)")


def _print_area_summary(rows: list) -> None:
    """打印测试集各类别病害面积的汇总统计（均值 ± 标准差）。"""
    if not rows:
        return

    print("\n[病害面积汇总统计]")
    print(f"  {'类别':<14} {'均值占比(%)':>12} {'最大占比(%)':>12} {'出现图像数':>10}")
    print(f"  {'-'*52}")

    total_images = len(rows)
    for cls_id in FOREGROUND_IDS:
        cls_name   = CLASSES[cls_id]
        ratio_key  = f"{cls_name}_ratio"
        ratios     = [r[ratio_key] for r in rows if ratio_key in r]
        if not ratios:
            continue
        mean_r     = np.mean(ratios) * 100
        max_r      = np.max(ratios)  * 100
        appear_cnt = sum(1 for r in ratios if r > 0)
        print(f"  {cls_name:<14} {mean_r:>12.4f} {max_r:>12.4f} "
              f"{appear_cnt:>10}/{total_images}")


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="推理结果后处理与可视化")
    parser.add_argument("--results_json", type=str,
                        default="outputs/predictions/inference_results.json")
    parser.add_argument("--image_dir",    type=str,
                        default="data/labeled/images")
    parser.add_argument("--output_dir",   type=str,
                        default="outputs/predictions")
    parser.add_argument("--alpha",        type=float, default=0.5,
                        help="叠加透明度（0~1，默认 0.5）")
    parser.add_argument("--device",       type=str,   default="cpu",
                        help="混淆矩阵计算设备（默认 cpu，推理已完成无需 GPU）")
    args = parser.parse_args()

    results_json = Path(args.results_json)
    if not results_json.exists():
        raise FileNotFoundError(
            f"找不到推理结果文件：{results_json}\n"
            f"请先运行 predict.py"
        )

    with open(results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"[后处理] 读取 {len(results)} 条推理结果")

    run_postprocess(
        results    = results,
        image_dir  = Path(args.image_dir),
        output_dir = Path(args.output_dir),
        alpha      = args.alpha,
        device_str = args.device,
    )


if __name__ == "__main__":
    main()