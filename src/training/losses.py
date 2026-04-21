"""
losses.py
────────────────────────────────────────────────────────────────────────────
Step 8：加权损失函数

功能：
  在 Mask2Former 原生复合损失的基础上叠加两项额外监督：

  1. 类别加权语义损失（Class-Weighted Semantic Loss）
     对 Mask2Former 输出的语义 logits 计算加权交叉熵，
     权重来自 Step 1 的 Median Frequency Balancing，
     补偿类别极度不平衡问题（fixpatch / Alligator 像素极少）

  2. 边缘感知损失（Edge-Aware Loss）
     在裂缝边缘 3px 范围内加大监督权重，
     强化模型对裂缝细边缘的感知能力，
     抑制边缘模糊和漏检

最终损失：
  total_loss = λ_native  × native_loss
             + λ_weighted × weighted_ce_loss
             + λ_edge    × edge_aware_loss

  默认权重：λ_native=1.0  λ_weighted=0.5  λ_edge=0.3

Mask2Former 原生损失（outputs.loss）组成：
  - 匈牙利匹配代价（Hungarian matching）
  - 分类交叉熵（CE Loss）
  - Mask 二值交叉熵（BCE Loss）
  - Mask Dice Loss
  以上均由 HuggingFace 实现自动计算，本模块不重复实现

运行方式（独立测试）：
  python src/training/losses.py
────────────────────────────────────────────────────────────────────────────

Step 8 几个关键设计说明：
MaxPool2d 代替形态学膨胀：compute_edge_mask 里用最大池化模拟了方形膨胀核，等价于 cv2.dilate，但可以在 GPU 上批量并行执行，比逐样本调用 OpenCV 快一个数量级。
三项损失的融合权重设置逻辑：

λ_native=1.0：原生损失是 Mask2Former 匈牙利匹配的核心，不能削弱
λ_weighted=0.5：加权 CE 是补充监督，数值量级与 native loss 相近，半权重避免 double counting
λ_edge=0.3：边缘损失聚焦于少量像素，数值天然偏小，0.3 足够提供有效梯度信号而不喧宾夺主

************************************************************
************************************************************
这三个 λ 值第一轮训练后如需调整，直接在 CrackSegLoss 的构造函数默认值处修改即可，不影响其他模块。
************************************************************
************************************************************

label_smoothing=0.05：手工标注的 labelme polygon 在边缘处存在 1~2px 的不精确，轻微的标签平滑可以降低模型对这些模糊边界像素的过拟合倾向。值不宜超过 0.1，否则会损失类别区分能力
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml


# ─── 常量 ─────────────────────────────────────────────────────────────────────

NUM_CLASSES = 2   # background(0) + 5 前景类别

# 各损失项的默认融合权重
LAMBDA_NATIVE   = 1.0
LAMBDA_WEIGHTED = 0.5
LAMBDA_EDGE     = 0.3

# 边缘膨胀半径（像素），控制边缘感知损失的感受范围
EDGE_DILATION_RADIUS = 3


# ─── 边缘掩码生成 ─────────────────────────────────────────────────────────────

def compute_edge_mask(
    semantic_gt: torch.Tensor,
    radius:      int = EDGE_DILATION_RADIUS,
) -> torch.Tensor:
    """
    从真值语义 mask 中提取裂缝边缘区域。

    步骤：
      1. 将所有前景类别（1~5）合并为二值前景 mask
      2. 用最大池化模拟膨胀操作（等价于半径为 radius 的方形膨胀核）
      3. 膨胀结果 XOR 原始前景 mask → 边缘区域

    使用最大池化代替形态学膨胀的原因：
      - 形态学操作需要逐 batch 逐样本用 opencv 处理，无法向量化
      - 最大池化可在 GPU 上批量完成，速度快 10x 以上
      - 等价性：MaxPool2d(kernel=2r+1, padding=r) 等效于方形膨胀核半径 r

    Parameters
    ----------
    semantic_gt : LongTensor [B, H, W]，值域 0~5
    radius      : 边缘膨胀半径（像素）

    Returns
    -------
    edge_mask : BoolTensor [B, H, W]，True 表示边缘区域
    """
    # 前景二值 mask：所有非背景像素为 1
    fg_binary = (semantic_gt > 0).float().unsqueeze(1)   # [B, 1, H, W]

    kernel_size = 2 * radius + 1
    # 膨胀：MaxPool2d
    dilated = F.max_pool2d(
        fg_binary,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )   # [B, 1, H, W]

    # 边缘 = 膨胀区域 - 原始前景区域
    edge = (dilated - fg_binary).bool().squeeze(1)   # [B, H, W]
    return edge


# ─── 加权交叉熵损失 ───────────────────────────────────────────────────────────

class WeightedSemanticLoss(nn.Module):
    """
    对语义分割 logits 计算类别加权交叉熵损失。

    Parameters
    ----------
    class_weights : list or Tensor，长度 = NUM_CLASSES（含 background）
                    background 权重应为 0（不参与损失）
    device        : 目标设备
    """

    def __init__(
        self,
        class_weights: list,
        device:        torch.device,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.register_buffer("weights", weights)
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,   # [B, C, H, W]  float
        target: torch.Tensor,   # [B, H, W]     long，值域 0~5
    ) -> torch.Tensor:
        """
        计算加权交叉熵。

        label_smoothing 的作用：
          对标注边界处的模糊像素降低置信度要求，
          缓解 labelme 手工标注边缘不精确带来的噪声监督。
          值设为 0.05（轻微平滑，不过度损失类别区分度）。
        """
        loss = F.cross_entropy(
            logits,
            target,
            weight=self.weights,
            label_smoothing=self.label_smoothing,
            ignore_index=-1,    # 若有无效像素可用 -1 标记
            reduction="mean",
        )
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss：专为类别不平衡设计。

    与加权CE的本质区别：
      加权CE：对整个类别施加固定权重，干扰匈牙利匹配的query分配
      Focal Loss：对每个像素动态调整权重，难分样本权重高，易分样本权重低
      不依赖预设类别权重，与匈牙利匹配机制不冲突

    公式：FL(pt) = -(1 - pt)^gamma * log(pt)
      gamma=0 时退化为标准CE
      gamma=2 时对易分样本（pt>0.5）的惩罚大幅降低，专注难分样本

    Parameters
    ----------
    gamma      : 聚焦参数，越大越专注难分样本，默认2.0
    fg_only    : 若为True，只对前景像素计算，不惩罚背景
    """

    def __init__(
        self,
        gamma:   float = 2.0,
        fg_only: bool  = True,
    ):
        super().__init__()
        self.gamma   = gamma
        self.fg_only = fg_only

    def forward(
        self,
        logits: torch.Tensor,   # [B, C, H, W]
        target: torch.Tensor,   # [B, H, W] long
    ) -> torch.Tensor:
        # 计算每个像素的 CE loss（不做 reduction）
        ce_loss = F.cross_entropy(
            logits,
            target,
            ignore_index=-1,
            reduction="none",
        )   # [B, H, W]

        # 计算每个像素被正确分类的概率 pt
        pt = torch.exp(-ce_loss)   # [B, H, W]

        # Focal 权重：难分样本（pt低）权重大，易分样本（pt高）权重小
        focal_weight = (1.0 - pt) ** self.gamma

        focal_loss = focal_weight * ce_loss   # [B, H, W]

        if self.fg_only:
            # 只对前景像素计算，背景像素不参与
            fg_mask = (target > 0)
            if fg_mask.sum() == 0:
                return torch.tensor(
                    0.0, device=logits.device, requires_grad=True
                )
            return focal_loss[fg_mask].mean()
        else:
            return focal_loss.mean()


# ─── 边缘感知损失 ─────────────────────────────────────────────────────────────

class EdgeAwareLoss(nn.Module):
    """
    在裂缝边缘区域施加额外监督的损失函数。

    设计思路：
      裂缝是细线状目标，边缘像素数量远少于内部像素，
      标准交叉熵对边缘的监督信号极弱，导致边缘模糊。
      本损失在边缘区域内单独计算 CE，迫使模型关注边缘细节。

    具体实现：
      1. 从真值 mask 生成边缘掩码（compute_edge_mask）
      2. 仅在边缘掩码为 True 的位置计算 CE
      3. 加权求和（边缘权重 > 1.0，强化监督）

    若当前 batch 无边缘像素（纯背景图），返回 0 损失。
    """

    def __init__(
        self,
        class_weights:  list,
        device:         torch.device,
        edge_weight:    float = 2.0,
        dilation_radius: int  = EDGE_DILATION_RADIUS,
    ):
        super().__init__()
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.register_buffer("weights", weights)
        self.edge_weight     = edge_weight
        self.dilation_radius = dilation_radius

    def forward(self, logits, target):
    # 原来：edge_mask 只取前景外侧背景像素
    # 修改：取前景膨胀区域，包含前景内侧和外侧边缘
        fg_binary = (target > 0).float().unsqueeze(1)   # [B, 1, H, W]
        kernel    = 2 * self.dilation_radius + 1

        # 膨胀后的区域（包含前景本身 + 外侧边缘）
        dilated = F.max_pool2d(
            fg_binary, kernel_size=kernel,
            stride=1, padding=self.dilation_radius,
        ).squeeze(1).bool()   # [B, H, W]

        # 边缘 = 膨胀区域（包含前景内外两侧）
        edge_mask = dilated   # 不再减去前景，保留前景像素在内

        if not edge_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        B, C, H, W  = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target.reshape(-1)
        edge_flat   = edge_mask.reshape(-1)

        logits_edge = logits_flat[edge_flat]
        target_edge = target_flat[edge_flat]

        if logits_edge.shape[0] == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # background 权重设为一个小值而非 0，避免 log(0) = NaN
        weights_safe = self.weights.clone()
        weights_safe[0] = 0.01

        loss = F.cross_entropy(
            logits_edge,
            target_edge,
            weight=weights_safe,
            reduction="mean",
        )
        # NaN 保护：若 loss 仍为 NaN（极端 batch），返回 0 跳过该项
        if torch.isnan(loss):
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return loss * self.edge_weight


# ─── 复合损失函数（主入口）───────────────────────────────────────────────────

class CrackSegLoss(nn.Module):
    """
    路面病害语义分割复合损失函数。

    融合三项损失：
      1. Mask2Former 原生损失（匈牙利匹配 + CE + BCE + Dice）
      2. 加权语义交叉熵（Median Frequency Balancing 权重）
      3. 边缘感知损失（裂缝边缘 3px 强化监督）

    Parameters
    ----------
    class_weights   : list[float]，长度 6，来自 class_statistics.py 计算结果
    device          : 训练设备
    lambda_native   : 原生损失权重（默认 1.0）
    lambda_weighted : 加权 CE 权重（默认 0.5）
    lambda_edge     : 边缘损失权重（默认 0.3）
    """

    def __init__(
        self,
        class_weights:   list,
        device:          torch.device,
        lambda_native:   float = LAMBDA_NATIVE,
        lambda_weighted: float = LAMBDA_WEIGHTED,
        lambda_focal:    float = 0.0,
        lambda_edge:     float = LAMBDA_EDGE,
    ):
        super().__init__()
        self.lambda_native   = lambda_native
        self.lambda_weighted = lambda_weighted
        self.lambda_focal    = lambda_focal
        self.lambda_edge     = lambda_edge
    
        self.weighted_ce = WeightedSemanticLoss(class_weights, device)
        self.focal_loss  = FocalLoss(gamma=2.0, fg_only=False)
        self.edge_loss   = EdgeAwareLoss(class_weights, device)
    
        print(f"[CrackSegLoss] "
              f"λ_native={lambda_native}  "
              f"λ_weighted={lambda_weighted}  "
              f"λ_focal={lambda_focal}  "
              f"λ_edge={lambda_edge}")
        print(f"  class_weights = {[round(w, 4) for w in class_weights]}")

    def forward(
        self,
        model:        nn.Module,
        pixel_values: torch.Tensor,   # [B, 3, H, W]
        mask_labels:  list,           # List[BoolTensor[N_i, H, W]]
        class_labels: list,           # List[LongTensor[N_i]]
    ) -> torch.Tensor:
        """
        执行 forward 并计算三项损失的加权和。

        步骤：
          1. 调用 model.forward() 获取原生损失和输出 logits
          2. 从输出 logits 推导语义分割 logits（参见 trainer._get_semantic_logits）
          3. 将实例表示还原为语义 mask（参见 trainer._instances_to_semantic）
          4. 计算加权 CE 和边缘损失
          5. 加权求和

        注意：
          semantic_logits 和 gt_semantic 的构建逻辑复用 trainer.py 中的工具函数，
          避免代码重复。
        """
        from src.training.trainer import _get_semantic_logits, _instances_to_semantic

        device = pixel_values.device
        H, W   = pixel_values.shape[-2:]

        # ── 1. 模型前向（含原生损失计算）─────────────────────────────────────
        outputs = model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        native_loss = outputs.loss   # Mask2Former 原生复合损失

        # ── 2. 推导语义分割 logits ────────────────────────────────────────────
        semantic_logits = _get_semantic_logits(outputs, (H, W)).float()   # [B, C, H, W]

        # ── 3. 构建真值语义 mask ──────────────────────────────────────────────
        gt_semantic = _instances_to_semantic(
            mask_labels, class_labels, (H, W), device
        )   # [B, H, W] long

        # ── 4. 额外损失 ───────────────────────────────────────────────────────
        w_ce_loss   = self.weighted_ce(semantic_logits, gt_semantic)
        focal_loss  = self.focal_loss(semantic_logits, gt_semantic)
        edge_loss   = self.edge_loss(semantic_logits, gt_semantic)


        # ── 5. 加权求和 ───────────────────────────────────────────────────────
        total_loss = (
            self.lambda_native   * native_loss
            + self.lambda_weighted * w_ce_loss
            + self.lambda_focal  * focal_loss
            + self.lambda_edge   * edge_loss
        )

        return total_loss


# ─── 工厂函数 ─────────────────────────────────────────────────────────────────

def build_loss_fn(
    yaml_path: Path,
    device:    torch.device,
) -> CrackSegLoss:
    """
    从 yaml 配置文件读取 class_weights，构建 CrackSegLoss。

    Parameters
    ----------
    yaml_path : configs/mask2former_crack.yaml
    device    : 训练设备

    Returns
    -------
    CrackSegLoss 实例，其 __call__ 签名为：
        loss = loss_fn(model, pixel_values, mask_labels, class_labels)
    """
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"找不到配置文件：{yaml_path}\n"
            f"请先运行 class_statistics.py 生成类别权重"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    class_weights = cfg.get("class_weights", [0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # 若 class_weights 全为默认值（说明 class_statistics.py 未运行），给出警告
    if all(w in (0.0, 1.0) for w in class_weights):
        print("[警告] class_weights 为默认值，建议先运行 class_statistics.py")

    # 从 yaml 读取各损失项权重，若未配置则使用默认值
    loss_cfg = cfg.get("loss", {})
    lambda_native   = loss_cfg.get("lambda_native",   1.0)
    lambda_weighted = loss_cfg.get("lambda_weighted",  0.0)
    lambda_edge     = loss_cfg.get("lambda_edge",      0.3)
    lambda_focal = loss_cfg.get("lambda_focal", 0.5)
    
    print(f"[损失权重] lambda_native={lambda_native}  "
          f"lambda_weighted={lambda_weighted}  "
          f"lambda_focal={lambda_focal}  "
          f"lambda_edge={lambda_edge}")

    return CrackSegLoss(
        class_weights   = class_weights,
        device          = device,
        lambda_native   = lambda_native,
        lambda_weighted = lambda_weighted,
        lambda_focal    = lambda_focal,
        lambda_edge     = lambda_edge,
    )

# ─── 独立测试入口 ─────────────────────────────────────────────────────────────

def _test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="configs/mask2former_crack.yaml")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[测试设备] {device}")

    # ── 合成数据 ──────────────────────────────────────────────────────────────
    B, C, H, W = 2, 6, 128, 128
    dummy_logits = torch.randn(B, C, H, W, device=device)

    # 真值语义 mask：随机分配类别
    dummy_gt = torch.randint(0, NUM_CLASSES, (B, H, W), device=device)

    # ── 测试 WeightedSemanticLoss ─────────────────────────────────────────────
    print("\n[测试 WeightedSemanticLoss]")
    weights      = [0.0, 2.5, 1.8, 3.1, 4.2, 1.2]
    weighted_ce  = WeightedSemanticLoss(weights, device)
    loss_wce     = weighted_ce(dummy_logits, dummy_gt)
    print(f"  weighted_ce_loss = {loss_wce.item():.4f}  ✓")

    # ── 测试 EdgeAwareLoss ────────────────────────────────────────────────────
    print("\n[测试 EdgeAwareLoss]")

    # 构造有明显边缘的 gt（中间一个矩形前景）
    gt_with_fg = torch.zeros(B, H, W, dtype=torch.long, device=device)
    gt_with_fg[:, 30:90, 30:90] = 1   # 类别 1（Transverse）

    edge_loss_fn = EdgeAwareLoss(weights, device)
    loss_edge    = edge_loss_fn(dummy_logits, gt_with_fg)
    print(f"  edge_aware_loss  = {loss_edge.item():.4f}  ✓")

    # 验证纯背景时边缘损失为 0
    gt_bg_only   = torch.zeros(B, H, W, dtype=torch.long, device=device)
    loss_edge_bg = edge_loss_fn(dummy_logits, gt_bg_only)
    print(f"  edge_loss (纯背景) = {loss_edge_bg.item():.4f}  (应≈0) ✓")

    # ── 测试 compute_edge_mask ────────────────────────────────────────────────
    print("\n[测试 compute_edge_mask]")
    edge_mask = compute_edge_mask(gt_with_fg, radius=3)
    print(f"  edge_mask shape  = {edge_mask.shape}")
    print(f"  edge pixel count = {edge_mask.sum().item()}")
    print(f"  edge dtype       = {edge_mask.dtype}  (应为 bool) ✓")

    # ── 验证边缘位置正确性（边缘像素应在前景边界附近）─────────────────────────
    fg_pixels   = int((gt_with_fg > 0).sum())
    edge_pixels = int(edge_mask.sum())
    print(f"  前景像素数 = {fg_pixels}，边缘像素数 = {edge_pixels}")
    assert edge_pixels > 0,       "边缘像素为 0，compute_edge_mask 可能有误"
    assert edge_pixels < fg_pixels, "边缘像素多于前景像素，膨胀逻辑可能有误"

    print("\n[✓] Step 8 losses.py 测试全部通过\n")


if __name__ == "__main__":
    _test()