"""
predict.py
────────────────────────────────────────────────────────────────────────────
Step 11：测试集推理

功能：
  1. 加载 checkpoints/best_miou.pth 权重
  2. 对 data/processed/splits/test.txt 中的测试样本逐张推理
  3. 输出原始预测 mask（施加物理约束之前）和施加约束之后的 mask
  4. 将结果传递给 postprocess.py 完成可视化和统计

推理流程：
  原图 (2720×1530)
    └→ 归一化 → model.forward()
    └→ _get_semantic_logits()   # 从 query logits 转换为语义 logits
    └→ argmax → 语义 mask [H, W]
    └→ apply_physical_constraints()
    └→ 保存 raw_mask.png + filtered_mask.png

注意事项：
  - 推理时不使用 AMP（保证数值精度）
  - 推理时不计算梯度（torch.no_grad）
  - batch_size 推理时可设为 1（A100 单张 2720×1530 推理约 3~5 秒）

运行方式：
  python src/inference/predict.py
  python src/inference/predict.py \
      --ckpt checkpoints/best_miou.pth \
      --test_txt data/processed/splits/test.txt \
      --image_dir data/labeled/images \
      --mask_dir  data/labeled/masks \
      --output_dir outputs/predictions \
      --yaml configs/mask2former_crack.yaml
────────────────────────────────────────────────────────────────────────────

Step 11 几个关键设计说明：
推理时不传 mask_labels / class_labels：HuggingFace 的 Mask2Former forward() 在只传 pixel_values 时不计算 loss，
直接输出 masks_queries_logits 和 class_queries_logits，正好是推理所需的内容。训练时必须传标注才有 loss，推理时传了反而会触发不必要的匈牙利匹配计算。

raw_mask 和 filtered_mask 都保存：两者都存到磁盘，一方面可以在 postprocess.py 中对比物理约束前后的差异，
另一方面如果事后发现某个约束参数过激误剔了真实裂缝，可以直接从 raw_mask 重新用不同参数过滤，不需要重新推理。

inference_results.json：记录每张图的推理耗时、mask 路径、物理约束统计，postprocess.py 直接读取这个文件驱动后续处理，两个模块完全解耦，可以单独重跑 postprocess 而不重跑推理。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model.mask2former_config import build_model, load_config
from src.model.physical_constraints import (
    ConstraintParams,
    apply_physical_constraints,
    summarize_constraint_stats,
)
from src.training.trainer import _get_semantic_logits
from src.dataset.crack_dataset import IMAGENET_MEAN, IMAGENET_STD


# ─── 常量 ─────────────────────────────────────────────────────────────────────

CLASSES = {
    0: "background",
    1: "Transverse",
    2: "Longitudinal",
    3: "Oblique",
    4: "Alligator",
    5: "fixpatch",
}

# 可视化颜色（BGR 格式，供 postprocess.py 使用）
CLASS_COLORS_BGR = {
    0: (64,  64,  64),    # background → 深灰
    1: (60,  76,  231),   # Transverse → 红（BGR）
    2: (219, 152,  52),   # Longitudinal → 蓝（BGR）
    3: (18,  156, 243),   # Oblique → 橙（BGR）
    4: (179,  68, 142),   # Alligator → 紫（BGR）
    5: (39,  174,  96),   # fixpatch → 绿（BGR）
}


# ─── 图像预处理 ───────────────────────────────────────────────────────────────

def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """
    单张图像预处理：BGR uint8 → RGB float32 → ImageNet 归一化 → CHW Tensor。
    与 crack_dataset.py 中的 _normalize() 完全一致。

    Returns
    -------
    tensor : FloatTensor [1, 3, H, W]（含 batch 维度）
    """
    image_rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_f    = image_rgb.astype(np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    image_norm = (image_f - mean) / std           # [H, W, 3]
    image_chw  = np.transpose(image_norm, (2, 0, 1))  # [3, H, W]

    return torch.from_numpy(image_chw).float().unsqueeze(0)  # [1, 3, H, W]


# ─── 单张推理 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_one(
    model:  torch.nn.Module,
    image:  np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    对单张原始图像执行推理，返回语义分割 mask。

    Parameters
    ----------
    model  : 已加载权重的 Mask2Former 模型（eval 模式）
    image  : BGR uint8 ndarray [H, W, 3]
    device : 推理设备

    Returns
    -------
    pred_mask : uint8 ndarray [H, W]，像素值 0~5
    """
    H, W = image.shape[:2]

    # 预处理
    pixel_values = preprocess_image(image).to(device)   # [1, 3, H, W]

    # 前向（不传 mask_labels / class_labels，不计算 loss）
    outputs = model(pixel_values=pixel_values)

    # 转换为语义分割 logits [1, C, H, W]
    semantic_logits = _get_semantic_logits(outputs, (H, W))

    # argmax → 语义 mask
    pred_mask = semantic_logits.argmax(dim=1).squeeze(0)  # [H, W]
    pred_mask = pred_mask.cpu().numpy().astype(np.uint8)

    return pred_mask


# ─── Checkpoint 加载 ──────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    ckpt_path: Path,
    cfg:       dict,
    device:    torch.device,
) -> torch.nn.Module:
    """
    从 checkpoint 加载模型权重。

    Parameters
    ----------
    ckpt_path : best_miou.pth 路径
    cfg       : 来自 yaml 的配置字典
    device    : 推理设备

    Returns
    -------
    model : eval 模式的 Mask2Former
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"找不到 checkpoint：{ckpt_path}\n"
            f"请先完成训练（运行 train.py）"
        )

    print(f"[加载权重] {ckpt_path}")
    model = build_model(cfg)
    model = model.to(device)

    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch     = ckpt.get("epoch", "unknown")
    best_miou = ckpt.get("best_miou", "unknown")
    print(f"  ckpt epoch={epoch}  best_mIoU={best_miou}")

    return model


# ─── 主推理循环 ───────────────────────────────────────────────────────────────

def run_inference(
    model:       torch.nn.Module,
    test_stems:  list,
    image_dir:   Path,
    mask_dir:    Path,
    output_dir:  Path,
    params:      ConstraintParams,
    device:      torch.device,
    image_ext:   str = ".jpg",
) -> list:
    """
    对测试集所有样本执行推理，保存预测结果。

    目录结构：
      output_dir/
        raw_masks/          ← 物理约束前的原始预测 mask（PNG，0~5）
        filtered_masks/     ← 物理约束后的 mask（PNG，0~5）
        overlays/           ← 彩色叠加可视化图（由 postprocess.py 生成）

    Returns
    -------
    results : list[dict]，每个元素包含：
        stem           : 文件名主干
        raw_mask_path  : 原始预测 mask 路径
        filt_mask_path : 过滤后 mask 路径
        gt_mask_path   : 真值 mask 路径（可能为 None）
        infer_time_s   : 单张推理耗时（秒）
        constraint_stats: 物理约束统计
    """
    raw_dir  = output_dir / "raw_masks"
    filt_dir = output_dir / "filtered_masks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    filt_dir.mkdir(parents=True, exist_ok=True)

    results          = []
    constraint_stats = []
    total_infer_time = 0.0
    error_count      = 0

    for stem in tqdm(test_stems, desc="测试集推理", unit="img"):
        # ── 查找图像文件 ──────────────────────────────────────────────────────
        image_path = image_dir / (stem + image_ext)
        if not image_path.exists():
            for ext in [".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                alt = image_dir / (stem + ext)
                if alt.exists():
                    image_path = alt
                    break

        if not image_path.exists():
            print(f"  [警告] 找不到图像：{stem}，已跳过")
            error_count += 1
            continue

        # ── 读取图像 ──────────────────────────────────────────────────────────
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  [警告] 无法读取图像：{image_path.name}，已跳过")
            error_count += 1
            continue

        # ── 推理 ──────────────────────────────────────────────────────────────
        t0 = time.time()
        raw_mask = predict_one(model, image, device)
        infer_time = time.time() - t0
        total_infer_time += infer_time

        # ── 物理约束 ──────────────────────────────────────────────────────────
        filtered_mask, stats = apply_physical_constraints(
            raw_mask, params, verbose=False
        )
        constraint_stats.append(stats)

        # ── 保存 mask ─────────────────────────────────────────────────────────
        raw_mask_path  = raw_dir  / (stem + ".png")
        filt_mask_path = filt_dir / (stem + ".png")
        cv2.imwrite(str(raw_mask_path),  raw_mask)
        cv2.imwrite(str(filt_mask_path), filtered_mask)

        # ── 真值 mask 路径（用于 postprocess.py 评估对比）─────────────────────
        gt_mask_path = mask_dir / (stem + ".png")
        gt_mask_path = gt_mask_path if gt_mask_path.exists() else None

        results.append({
            "stem":            stem,
            "image_path":      str(image_path),
            "raw_mask_path":   str(raw_mask_path),
            "filt_mask_path":  str(filt_mask_path),
            "gt_mask_path":    str(gt_mask_path) if gt_mask_path else None,
            "infer_time_s":    round(infer_time, 3),
            "constraint_stats": stats,
        })

    # ── 统计 ──────────────────────────────────────────────────────────────────
    n = len(results)
    avg_time = total_infer_time / n if n > 0 else 0.0
    print(f"\n  推理完成：{n} 张  |  "
          f"平均耗时：{avg_time:.2f}s/张  |  "
          f"跳过：{error_count} 张")

    summarize_constraint_stats(constraint_stats)

    # ── 保存推理结果元数据 ────────────────────────────────────────────────────
    meta_path = output_dir / "inference_results.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  [✓] 推理元数据已保存：{meta_path}")

    return results


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="测试集推理")
    parser.add_argument("--ckpt",       type=str,
                        default="checkpoints/best_miou.pth")
    parser.add_argument("--test_txt",   type=str,
                        default="data/processed/splits/test.txt")
    parser.add_argument("--image_dir",  type=str,
                        default="data/labeled/images")
    parser.add_argument("--mask_dir",   type=str,
                        default="data/labeled/masks")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/predictions")
    parser.add_argument("--yaml",       type=str,
                        default="configs/mask2former_crack.yaml")
    parser.add_argument("--image_ext",  type=str, default=".jpg")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[推理设备] {device}")

    # ── 1. 加载配置和约束参数 ─────────────────────────────────────────────────
    yaml_path = Path(args.yaml)
    cfg       = load_config(yaml_path)
    params    = ConstraintParams.from_yaml(yaml_path)
    print(params)

    # ── 2. 读取测试集文件列表 ─────────────────────────────────────────────────
    test_txt = Path(args.test_txt)
    if not test_txt.exists():
        raise FileNotFoundError(
            f"找不到测试集列表：{test_txt}\n"
            f"请先运行 dataset_split.py"
        )
    with open(test_txt, "r", encoding="utf-8") as f:
        test_stems = [line.strip() for line in f if line.strip()]
    print(f"[测试集] {len(test_stems)} 张")

    # ── 3. 加载模型 ───────────────────────────────────────────────────────────
    model = load_model_from_checkpoint(Path(args.ckpt), cfg, device)

    # ── 4. 推理 ───────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    results = run_inference(
        model       = model,
        test_stems  = test_stems,
        image_dir   = Path(args.image_dir),
        mask_dir    = Path(args.mask_dir),
        output_dir  = output_dir,
        params      = params,
        device      = device,
        image_ext   = args.image_ext,
    )

    print(f"\n[✓] Step 11 推理完成。")
    print(f"    原始预测 mask ：{output_dir / 'raw_masks'}")
    print(f"    过滤后 mask   ：{output_dir / 'filtered_masks'}")
    print(f"    元数据文件    ：{output_dir / 'inference_results.json'}")
    print(f"\n    下一步：运行 postprocess.py 生成可视化和统计报表\n")


if __name__ == "__main__":
    main()