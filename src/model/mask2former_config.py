"""
mask2former_config.py
────────────────────────────────────────────────────────────────────────────
Step 5：Mask2Former 模型配置与初始化

功能：
  1. 从 configs/mask2former_crack.yaml 读取超参数
  2. 基于 HuggingFace transformers 加载预训练 Mask2Former（Swin-Base backbone）
  3. 将预训练模型的分类头替换为适配本项目 6 类（含 background）的新 head
  4. 配置差异化学习率（backbone 使用较小学习率，head 使用较大学习率）
  5. 提供模型参数量统计与显存估算工具函数

预训练权重说明：
  facebook/mask2former-swin-base-ade-semantic
  - 在 ADE20K（150类语义分割）上预训练
  - Swin-Base backbone 的特征提取能力直接迁移
  - pixel decoder 和 transformer decoder 权重也可复用
  - 仅最终分类线性层（class_embed）需要重新初始化（150类→6类）

差异化学习率策略：
  backbone 参数      : lr × backbone_lr_multiplier（默认 × 0.1）
  pixel_decoder 参数 : lr × 0.5（中间层，适中学习率）
  transformer_decoder
  + 分类头参数       : lr × 1.0（新初始化层，全速学习）

运行方式（独立测试）：
  python src/model/mask2former_config.py
  python src/model/mask2former_config.py --yaml configs/mask2former_crack.yaml
────────────────────────────────────────────────────────────────────────────

Step 5 的几个关键设计说明：
ignore_mismatched_sizes=True 的作用：HuggingFace 在加载预训练权重时，如果某层的 shape 与当前模型不一致会报错。
这个参数告诉 from_pretrained 遇到 shape 不匹配时跳过该层（即 class_embed），其余所有层正常加载预训练权重。
随后 _reinit_class_embed 显式地用 Kaiming 初始化补全这些被跳过的层，确保不遗漏。

三组差异化学习率的设计逻辑：backbone 的 Swin-Base 已在 ImageNet + ADE20K 上充分预训练，特征提取能力已经很强，
只需要小幅微调（lr × 0.1 = 1e-5）；pixel_decoder 是多尺度特征融合层，预训练权重有一定的迁移性，
用中等学习率（lr × 0.5 = 5e-5）；transformer decoder 和 class_embed 中新初始化的层需要全速收敛（lr × 1.0 = 1e-4）。

no-object 类的 +1：Mask2Former 的分类头输出维度是 num_classes + 1，多出的一个维度对应"无目标"查询（即该 query 没有匹配到任何实例），
这是匈牙利匹配机制的必要组成部分，不是 background 类，不要混淆。
"""

import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
)


# ─── 常量 ─────────────────────────────────────────────────────────────────────

PRETRAINED_MODEL_NAME = "facebook/mask2former-swin-base-ade-semantic"

# ADE20K 预训练时的类别数（原始 head 尺寸）
ADE20K_NUM_CLASSES = 150

# 本项目类别数（含 background）
NUM_CLASSES = 6

CLASSES = {
    0: "background",
    1: "Transverse",
    2: "Longitudinal",
    3: "Oblique",
    4: "Alligator",
    5: "fixpatch",
}


# ─── YAML 配置读取 ────────────────────────────────────────────────────────────

def load_config(yaml_path: Path) -> dict:
    """
    读取 configs/mask2former_crack.yaml。
    若文件不存在，返回内置默认配置（适用于首次运行尚未执行 class_statistics.py 的情况）。
    """
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"[✓] 已加载配置文件：{yaml_path}")
    else:
        print(f"[警告] 配置文件不存在：{yaml_path}，使用内置默认配置")
        cfg = _default_config()

    return cfg


def _default_config() -> dict:
    """内置默认配置，class_weights 需在 class_statistics.py 运行后更新。"""
    return {
        "model": {
            "backbone":   "swin-base",
            "pretrained": PRETRAINED_MODEL_NAME,
            "num_classes": NUM_CLASSES,
            "num_queries": 100,
        },
        "training": {
            "image_size":              [2720, 1530],
            "batch_size":              2,
            "accumulation_steps":      4,
            "lr":                      1e-4,
            "backbone_lr_multiplier":  0.1,
            "max_epochs":              100,
            "warmup_epochs":           5,
            "mixed_precision":         True,
            "gradient_checkpointing":  True,
            "num_workers":             4,
            "save_every_n_epochs":     10,
        },
        "physical_constraints": {
            "max_crack_width_px":           20,
            "max_transverse_span_ratio":    0.80,
            "min_contour_complexity":       20.0,
            "min_region_area_px":           50,
            "shadow_aspect_ratio_threshold": 30.0,
        },
        # 占位权重，运行 class_statistics.py 后会被覆盖
        "class_weights": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }


# ─── 模型构建 ─────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> Mask2FormerForUniversalSegmentation:
    """
    加载预训练 Mask2Former，并将分类头替换为适配本项目的新 head。

    关键修改：
      原始 class_embed 线性层：in_features → 150+1 (ADE20K + no-object)
      替换后 class_embed 线性层：in_features → 6+1   (本项目 + no-object)

    替换策略：
      - 加载完整预训练权重（backbone、pixel_decoder、transformer_decoder）
      - 仅对 class_embed 层重新随机初始化（Kaiming uniform）
      - 其余所有权重保持预训练值不变
    """
    num_classes = cfg["model"]["num_classes"]    # 6
    num_queries = cfg["model"]["num_queries"]    # 100
    pretrained  = cfg["model"].get("pretrained", PRETRAINED_MODEL_NAME)

    print(f"\n[Step 1/3] 加载预训练模型：{pretrained}")
    print(f"  预训练类别数：{ADE20K_NUM_CLASSES}  →  本项目类别数：{num_classes}")

    # ── 加载预训练配置并修改类别数 ────────────────────────────────────────────
    model_config = Mask2FormerConfig.from_pretrained(pretrained)

    # 修改类别数：num_labels 控制最终分类头的输出维度
    # Mask2Former 内部将 class_embed 定义为 Linear(hidden_dim, num_labels + 1)
    # +1 是 "no-object" 类（背景查询的输出标签）
    model_config.num_labels = num_classes

    # 保持 num_queries 与配置一致
    model_config.num_queries = num_queries

    # ── 加载预训练权重（ignore_mismatched_sizes=True 跳过 head 尺寸不匹配）────
    print(f"[Step 2/3] 加载预训练权重（忽略 class_embed 尺寸不匹配）...")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        pretrained,
        config=model_config,
        ignore_mismatched_sizes=True,   # 允许 class_embed 尺寸不同
    )

    # ── 验证并重新初始化分类头 ────────────────────────────────────────────────
    print(f"[Step 3/3] 验证分类头维度并重新初始化...")
    _reinit_class_embed(model, num_classes)

    # # ── 可选：启用 gradient checkpointing 节省显存 ───────────────────────────
    # if cfg["training"].get("gradient_checkpointing", True):
    #     model.gradient_checkpointing_enable()
    #     print("  [✓] Gradient checkpointing 已启用")

    if cfg["training"].get("gradient_checkpointing", True):
    # 对 Swin backbone 单独启用 gradient checkpointing
        if hasattr(model.model.pixel_level_module.encoder, "gradient_checkpointing"):
            model.model.pixel_level_module.encoder.gradient_checkpointing = True
            print("  [✓] Swin backbone gradient checkpointing 已启用")
        else:
             print("  [提示] 当前模型不支持 gradient checkpointing，已跳过")

    return model


def _reinit_class_embed(
    model: Mask2FormerForUniversalSegmentation,
    num_classes: int,
) -> None:
    """
    定位并重新初始化所有 class_embed 线性层。

    Mask2Former 的 transformer_decoder 中每一层都有独立的 class_embed，
    形成一个 ModuleList，最终取最后一层的输出作为分类预测。
    全部重新初始化确保输出维度与 num_classes+1 一致。
    """
    reinit_count = 0

    for name, module in model.named_modules():
        if "class_embed" in name and isinstance(module, nn.Linear):
            expected_out = num_classes + 1   # +1 为 no-object 类
            if module.out_features != expected_out:
                # 重新构造线性层
                new_linear = nn.Linear(module.in_features, expected_out)
                nn.init.kaiming_uniform_(new_linear.weight, nonlinearity="relu")
                nn.init.zeros_(new_linear.bias)
                print(f"  [重初始化] {name}: "
                      f"({module.in_features}, {module.out_features}) "
                      f"→ ({module.in_features}, {expected_out})")

                # 找到父模块并替换
                parent_name, child_name = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, new_linear)
                reinit_count += 1
            else:
                print(f"  [已匹配] {name}: out_features={module.out_features} ✓")

    if reinit_count == 0:
        print("  [✓] class_embed 维度已正确（from_pretrained 已处理）")
    else:
        print(f"  [✓] 共重新初始化 {reinit_count} 个 class_embed 层")


# ─── 差异化学习率 ─────────────────────────────────────────────────────────────

def get_param_groups(
    model: Mask2FormerForUniversalSegmentation,
    base_lr: float,
    backbone_lr_multiplier: float = 0.1,
) -> list[dict]:
    """
    将模型参数按模块划分为三组，分配不同的学习率。

    分组策略：
      Group 1 - backbone (Swin)
        学习率 = base_lr × backbone_lr_multiplier（默认 1e-5）
        理由：backbone 已充分预训练，只需微调，过大的学习率会破坏特征

      Group 2 - pixel_decoder（FPN-style 多尺度特征融合模块）
        学习率 = base_lr × 0.5（默认 5e-5）
        理由：pixel_decoder 与 backbone 协同，适中的学习率

      Group 3 - transformer_decoder + class_embed（查询-注意力解码器 + 分类头）
        学习率 = base_lr × 1.0（默认 1e-4）
        理由：class_embed 已重新初始化，需要较大学习率快速收敛

    Returns
    -------
    param_groups : list of dict，可直接传入 torch.optim.AdamW
    """
    backbone_params      = []
    pixel_decoder_params = []
    decoder_head_params  = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "pixel_level_module.encoder" in name or "backbone" in name:
            backbone_params.append(param)

        elif "pixel_level_module.decoder" in name or "pixel_decoder" in name:
            pixel_decoder_params.append(param)

        else:
            # transformer_decoder、class_embed、mask_embed 等
            decoder_head_params.append(param)

    param_groups = [
        {
            "params": backbone_params,
            "lr":     base_lr * backbone_lr_multiplier,
            "name":   "backbone",
        },
        {
            "params": pixel_decoder_params,
            "lr":     base_lr * 0.5,
            "name":   "pixel_decoder",
        },
        {
            "params": decoder_head_params,
            "lr":     base_lr * 1.0,
            "name":   "transformer_decoder_and_head",
        },
    ]

    # 打印参数量统计
    for g in param_groups:
        n_params = sum(p.numel() for p in g["params"])
        print(f"  {g['name']:<35}  "
              f"params={n_params/1e6:6.2f}M  lr={g['lr']:.2e}")

    return param_groups


# ─── 参数量 & 显存估算 ────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """统计模型总参数量、可训练参数量（单位：M）。"""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    return {
        "total_M":     total     / 1e6,
        "trainable_M": trainable / 1e6,
        "frozen_M":    frozen    / 1e6,
    }


def estimate_vram_gb(
    model:      nn.Module,
    batch_size: int   = 2,
    img_h:      int   = 1530,
    img_w:      int   = 2720,
    amp:        bool  = True,
) -> dict:
    """
    粗略估算训练时的显存占用。

    估算公式（经验值）：
      模型权重      ：参数量 × 4 bytes（float32）或 × 2 bytes（float16 AMP）
      梯度缓存      ：= 模型权重大小
      优化器状态    ：= 模型权重大小 × 2（AdamW 的 m、v）
      激活值        ：batch_size × H × W × C × 层数 × bytes（难以精确，取经验系数）
    """
    param_bytes = sum(
        p.numel() * (2 if amp else 4) for p in model.parameters()
    )
    grad_bytes    = param_bytes
    optimizer_bytes = param_bytes * 2   # AdamW

    # 激活值估算（Swin-Base 在 2720×1530 的经验估算）
    pixels = batch_size * img_h * img_w
    activation_bytes = pixels * 256 * 2   # 粗略估算（256通道，float16）

    total_bytes = param_bytes + grad_bytes + optimizer_bytes + activation_bytes
    total_gb    = total_bytes / (1024 ** 3)

    return {
        "model_weights_GB":  param_bytes    / (1024 ** 3),
        "gradients_GB":      grad_bytes     / (1024 ** 3),
        "optimizer_GB":      optimizer_bytes / (1024 ** 3),
        "activations_GB":    activation_bytes / (1024 ** 3),
        "total_estimated_GB": total_gb,
    }


# ─── 独立测试入口 ─────────────────────────────────────────────────────────────

def _test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="configs/mask2former_crack.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.yaml))

    print("\n" + "=" * 60)
    print("  Mask2Former 模型配置测试")
    print("=" * 60)

    # 构建模型
    model = build_model(cfg)

    # 参数量统计
    print("\n[参数量统计]")
    param_info = count_parameters(model)
    print(f"  总参数量    ：{param_info['total_M']:.2f} M")
    print(f"  可训练参数  ：{param_info['trainable_M']:.2f} M")
    print(f"  冻结参数    ：{param_info['frozen_M']:.2f} M")

    # 差异化学习率分组
    print("\n[差异化学习率分组]")
    base_lr = cfg["training"]["lr"]
    backbone_lr_mult = cfg["training"]["backbone_lr_multiplier"]
    param_groups = get_param_groups(model, base_lr, backbone_lr_mult)

    # 显存估算
    print("\n[显存估算（A100 40GB，AMP=True）]")
    vram = estimate_vram_gb(
        model,
        batch_size=cfg["training"]["batch_size"],
        img_h=cfg["training"]["image_size"][1],
        img_w=cfg["training"]["image_size"][0],
        amp=cfg["training"]["mixed_precision"],
    )
    for k, v in vram.items():
        flag = "  ← 注意" if k == "total_estimated_GB" and v > 35 else ""
        print(f"  {k:<25}：{v:.2f} GB{flag}")

    a100_capacity = 40.0
    usage_pct = vram["total_estimated_GB"] / a100_capacity * 100
    print(f"\n  A100 40GB 显存利用率（估算）：{usage_pct:.1f}%")
    if usage_pct > 90:
        print("  [警告] 显存可能不足，建议减小 batch_size 或启用 gradient_checkpointing")
    else:
        print("  [✓] 显存预算充足")

    # 前向传播测试（小尺寸，仅验证接口）
    print("\n[前向传播测试（128×128 小图，验证接口）]")
    model.eval()
    dummy_pixel = torch.randn(1, 3, 128, 128)
    dummy_masks = [torch.zeros(2, 128, 128, dtype=torch.bool)]
    dummy_classes = [torch.tensor([1, 2], dtype=torch.long)]

    with torch.no_grad():
        outputs = model(
            pixel_values=dummy_pixel,
            mask_labels=dummy_masks,
            class_labels=dummy_classes,
        )

    print(f"  输出 keys：{list(outputs.keys())}")
    if hasattr(outputs, "loss") and outputs.loss is not None:
        print(f"  loss：{outputs.loss.item():.4f}  ✓")
    print(f"  masks_queries_logits：{outputs.masks_queries_logits.shape}")
    print(f"  class_queries_logits：{outputs.class_queries_logits.shape}")
    print("\n[✓] Step 5 模型配置测试通过\n")


if __name__ == "__main__":
    _test()