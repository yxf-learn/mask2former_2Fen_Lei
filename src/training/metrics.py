"""
metrics.py
────────────────────────────────────────────────────────────────────────────
Step 9：评估指标

功能：
  1. 基于混淆矩阵计算所有评估指标，保证数值一致性
  2. 支持 batch 级累积（训练中每 batch 追加，epoch 末统一计算）
  3. 指标列表：
       - 各类别 IoU（Intersection over Union）
       - mIoU（前景类别宏平均）
       - Pixel Accuracy（全类别）
       - Mean Accuracy（各类别召回率的宏平均）
       - F1 Score（各前景类别 + 宏平均）
  4. 训练结束后输出混淆矩阵热力图（PNG）

设计原则：
  - 所有指标均基于同一个累积混淆矩阵计算，避免各指标间数值矛盾
  - 混淆矩阵在 GPU 上累积（用 torch.bincount），epoch 末转移到 CPU 计算指标
  - background 类（cls 0）不计入 mIoU / Mean Accuracy / F1 的宏平均
  - 若某类别在真值中完全不存在，该类 IoU 记为 -1（无效值），不参与均值

运行方式（独立测试）：
  python src/training/metrics.py
────────────────────────────────────────────────────────────────────────────

SStep9 几个关键设计说明：
torch.bincount 构建混淆矩阵：用 gt × C + pred 将二维类别对编码为一维索引，再用 bincount 统计频次，最后 reshape 为 [C, C]。
整个过程在 GPU 上批量完成，比逐像素循环快几十倍，也比 sklearn.confusion_matrix 快（后者需要 CPU numpy）。

两套 IoU 接口的设计意图：compute_batch_iou 是轻量函数，仅用于 tqdm 进度条的实时显示，每个 batch 独立计算，存在小样本偏差。
ConfusionMatrix.compute() 是 epoch 末的精确计算，基于全量累积矩阵，这才是写入训练历史和判断 best_miou 的数据来源。
两者在 build_metric_fns 的 wrapper 里被统一封装，trainer 调用一个函数即可同时完成两件事。

-1 标记无效 IoU 的意义：若某类在验证集真值中完全不出现，分母为 0，IoU 无定义。用 -1 标记后，宏平均只对有效类别求均值，避免人为引入 0 值拉低 mIoU。

行归一化混淆矩阵：每行除以该行总数，对角线变为各类的召回率。第一轮训练结束后看这张图，
能直观发现哪些类别之间存在混淆（如 Transverse 行的 Oblique 列值偏高，说明斜向裂缝被误判为横向）。
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ─── 常量 ─────────────────────────────────────────────────────────────────────

NUM_CLASSES = 2

CLASSES = {
    0: "background",
    1: "damage",
}

FOREGROUND_IDS   = [1]
FOREGROUND_NAMES = ["damage"]


# ─── 混淆矩阵累积器 ───────────────────────────────────────────────────────────

class ConfusionMatrix:
    """
    语义分割混淆矩阵，支持 batch 级 GPU 累积。

    矩阵定义：
      matrix[i, j] = 真值为类别 i、预测为类别 j 的像素数
      对角线 = 正确预测；非对角线 = 错误预测

    使用方式：
        cm = ConfusionMatrix(num_classes=6, device=device)
        for batch in loader:
            cm.update(pred_mask, gt_mask)
        metrics = cm.compute()
        cm.reset()
    """

    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device      = device
        self.matrix      = torch.zeros(
            num_classes, num_classes,
            dtype=torch.long,
            device=device,
        )

    def update(
        self,
        pred: torch.Tensor,   # [B, H, W] long，预测类别 ID
        gt:   torch.Tensor,   # [B, H, W] long，真值类别 ID
    ) -> None:
        """
        将一个 batch 的预测与真值累积进混淆矩阵。

        实现：
          将 pred 和 gt 展平后，用 gt * num_classes + pred 编码为一维索引，
          再用 bincount 统计各组合出现次数，reshape 为 [C, C] 矩阵。
          全程在 GPU 上完成，无需 CPU 转移。
        """
        with torch.no_grad():
            pred_flat = pred.reshape(-1)
            gt_flat   = gt.reshape(-1)

            # 过滤掉无效值（-1 标记的忽略区域）
            valid_mask = (gt_flat >= 0) & (gt_flat < self.num_classes) \
                       & (pred_flat >= 0) & (pred_flat < self.num_classes)
            pred_flat  = pred_flat[valid_mask]
            gt_flat    = gt_flat[valid_mask]

            combined   = gt_flat * self.num_classes + pred_flat
            bincount   = torch.bincount(
                combined,
                minlength=self.num_classes * self.num_classes,
            )
            self.matrix += bincount.reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        """清零矩阵，每 epoch 开始前调用。"""
        self.matrix.zero_()

    def compute(self) -> dict:
        """
        基于当前累积矩阵计算所有指标。

        Returns
        -------
        dict 包含以下键：
          iou_per_class   : dict {class_name: float}，-1 表示该类不存在
          mean_iou        : float，前景类 IoU 的宏平均
          pixel_accuracy  : float
          mean_accuracy   : float，前景类召回率的宏平均
          f1_per_class    : dict {class_name: float}
          mean_f1         : float
        """
        mat = self.matrix.cpu().float()   # [C, C]

        # ── TP / FP / FN ─────────────────────────────────────────────────────
        tp = mat.diag()                        # [C]  对角线
        fp = mat.sum(dim=0) - tp               # [C]  列和 - TP（预测为 c 但真值不是 c）
        fn = mat.sum(dim=1) - tp               # [C]  行和 - TP（真值为 c 但预测不是 c）
        gt_count = mat.sum(dim=1)              # [C]  每类真值像素总数

        # ── IoU ──────────────────────────────────────────────────────────────
        iou = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for cls_id in range(NUM_CLASSES):
            denom = tp[cls_id] + fp[cls_id] + fn[cls_id]
            if gt_count[cls_id] == 0:
                iou[cls_id] = -1.0    # 该类在真值中不存在，标记为无效
            elif denom == 0:
                iou[cls_id] = 0.0
            else:
                iou[cls_id] = tp[cls_id] / denom

        # 前景 IoU 宏平均（排除 background 和无效值）
        valid_fg_ious = [
            iou[i].item()
            for i in FOREGROUND_IDS
            if iou[i].item() >= 0
        ]
        mean_iou = sum(valid_fg_ious) / len(valid_fg_ious) if valid_fg_ious else 0.0

        iou_per_class = {
            CLASSES[i]: round(iou[i].item(), 6)
            for i in range(NUM_CLASSES)
        }

        # ── Pixel Accuracy ────────────────────────────────────────────────────
        total_correct = tp.sum().item()
        total_pixels  = mat.sum().item()
        pixel_accuracy = total_correct / total_pixels if total_pixels > 0 else 0.0

        # ── Mean Accuracy（各类召回率的宏平均，仅前景）───────────────────────
        recall = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for cls_id in range(NUM_CLASSES):
            if gt_count[cls_id] > 0:
                recall[cls_id] = tp[cls_id] / gt_count[cls_id]

        valid_fg_recalls = [
            recall[i].item()
            for i in FOREGROUND_IDS
            if gt_count[i] > 0
        ]
        mean_accuracy = (
            sum(valid_fg_recalls) / len(valid_fg_recalls)
            if valid_fg_recalls else 0.0
        )

        # ── F1 Score ──────────────────────────────────────────────────────────
        f1 = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        for cls_id in range(NUM_CLASSES):
            precision_denom = tp[cls_id] + fp[cls_id]
            recall_denom    = tp[cls_id] + fn[cls_id]

            if precision_denom == 0 or recall_denom == 0:
                f1[cls_id] = 0.0
            else:
                precision     = tp[cls_id] / precision_denom
                recall_val    = tp[cls_id] / recall_denom
                f1_denom      = precision + recall_val
                f1[cls_id]    = (
                    2 * precision * recall_val / f1_denom
                    if f1_denom > 0 else 0.0
                )

        valid_fg_f1s = [
            f1[i].item()
            for i in FOREGROUND_IDS
            if gt_count[i] > 0
        ]
        mean_f1 = sum(valid_fg_f1s) / len(valid_fg_f1s) if valid_fg_f1s else 0.0

        f1_per_class = {
            CLASSES[i]: round(f1[i].item(), 6)
            for i in range(NUM_CLASSES)
        }

        return {
            "iou_per_class":  iou_per_class,
            "mean_iou":       round(mean_iou, 6),
            "pixel_accuracy": round(pixel_accuracy, 6),
            "mean_accuracy":  round(mean_accuracy, 6),
            "f1_per_class":   f1_per_class,
            "mean_f1":        round(mean_f1, 6),
        }

    def get_matrix_numpy(self) -> np.ndarray:
        """返回混淆矩阵的 numpy 版本，供可视化使用。"""
        return self.matrix.cpu().numpy().astype(np.int64)


# ─── Batch 级 IoU（供 trainer.py validate_one_epoch 使用）──────────────────

def compute_batch_iou(
    pred: torch.Tensor,   # [B, H, W] long
    gt:   torch.Tensor,   # [B, H, W] long
) -> dict:
    """
    计算单个 batch 的各前景类别 IoU。

    此函数供 trainer.py 的 validate_one_epoch 调用，
    返回值为每个前景类别的 IoU 字典，-1 表示该类在本 batch 不存在。

    注意：batch 级 IoU 存在统计偏差（样本数少时方差大），
    epoch 末的最终指标应使用 ConfusionMatrix.compute() 的结果。
    此处返回值仅用于 trainer 中的 tqdm 进度条实时显示。

    Returns
    -------
    dict {class_name: float}，-1 表示该类在当前 batch 不存在
    """
    with torch.no_grad():
        result = {}
        for cls_id in FOREGROUND_IDS:
            cls_name = CLASSES[cls_id]
            pred_c   = (pred == cls_id)
            gt_c     = (gt   == cls_id)

            if not gt_c.any():
                result[cls_name] = -1.0
                continue

            intersection = (pred_c & gt_c).sum().item()
            union        = (pred_c | gt_c).sum().item()
            result[cls_name] = intersection / union if union > 0 else 0.0

    return result


# ─── 混淆矩阵可视化 ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    matrix:     np.ndarray,
    output_path: Path,
    normalize:  bool = True,
) -> None:
    """
    绘制并保存混淆矩阵热力图。

    Parameters
    ----------
    matrix      : ndarray [C, C]，来自 ConfusionMatrix.get_matrix_numpy()
    output_path : 保存路径
    normalize   : 若为 True，对每行归一化（显示召回率），便于识别各类误分方向
    """
    class_names = [CLASSES[i] for i in range(NUM_CLASSES)]

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)   # 防止除零
        matrix_plot = matrix.astype(float) / row_sums
        fmt         = ".2f"
        title       = "混淆矩阵（行归一化，显示各类召回率）"
        vmin, vmax  = 0.0, 1.0
    else:
        matrix_plot = matrix
        fmt         = "d"
        title       = "混淆矩阵（像素数）"
        vmin, vmax  = None, None

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        matrix_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="lightgray",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        annot_kws={"size": 9},
    )

    ax.set_xlabel("预测类别", fontsize=11, labelpad=8)
    ax.set_ylabel("真值类别", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=12, pad=12)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] 混淆矩阵已保存：{output_path}")


# ─── 指标报告打印 ─────────────────────────────────────────────────────────────

def print_metrics_report(metrics: dict) -> None:
    """格式化打印完整评估报告。"""
    print("\n" + "=" * 58)
    print("  评估指标报告")
    print("=" * 58)
    print(f"  Pixel Accuracy : {metrics['pixel_accuracy']:.4f}")
    print(f"  Mean Accuracy  : {metrics['mean_accuracy']:.4f}  (前景类宏平均)")
    print(f"  Mean IoU       : {metrics['mean_iou']:.4f}  (前景类宏平均)")
    print(f"  Mean F1        : {metrics['mean_f1']:.4f}  (前景类宏平均)")
    print(f"\n  {'类别':<14} {'IoU':>8} {'F1':>8}")
    print(f"  {'-'*32}")
    for cls_id in range(NUM_CLASSES):
        name     = CLASSES[cls_id]
        iou_val  = metrics["iou_per_class"].get(name, -1)
        f1_val   = metrics["f1_per_class"].get(name, 0.0)
        iou_str  = f"{iou_val:.4f}" if iou_val >= 0 else "  N/A "
        flag     = "  (background)" if cls_id == 0 else ""
        print(f"  {name:<14} {iou_str:>8} {f1_val:>8.4f}{flag}")
    print("=" * 58 + "\n")


# ─── 工厂函数（供 trainer.py 使用）──────────────────────────────────────────

def build_metric_fns(device: torch.device) -> tuple:
    """
    构建评估所需的函数和对象，返回给 Trainer 使用。

    Returns
    -------
    cm             : ConfusionMatrix 实例（epoch 末调用 compute()）
    compute_iou_fn : 函数，签名 (pred, gt) -> dict，供 validate_one_epoch 使用
    """
    cm = ConfusionMatrix(num_classes=NUM_CLASSES, device=device)

    def compute_iou_fn(
        pred: torch.Tensor,
        gt:   torch.Tensor,
    ) -> dict:
        """
        wrapper：更新混淆矩阵并返回 batch 级 IoU（用于进度条显示）。
        epoch 末通过 cm.compute() 获取精确指标。
        """
        cm.update(pred, gt)
        return compute_batch_iou(pred, gt)

    return cm, compute_iou_fn


# ─── 独立测试入口 ─────────────────────────────────────────────────────────────

def _test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[测试设备] {device}\n")

    B, H, W = 4, 256, 256
    cm = ConfusionMatrix(num_classes=NUM_CLASSES, device=device)

    # ── 合成 10 个 batch，模拟一个 epoch ─────────────────────────────────────
    print("[模拟训练 epoch，累积 10 个 batch]")
    for batch_i in range(10):
        # 真值：随机类别，但确保每个类别都有出现
        gt = torch.randint(0, NUM_CLASSES, (B, H, W), device=device)

        # 预测：在真值基础上引入 20% 随机错误，模拟不完美预测
        noise_mask = torch.rand(B, H, W, device=device) < 0.20
        pred       = gt.clone()
        pred[noise_mask] = torch.randint(
            0, NUM_CLASSES, (noise_mask.sum().item(),), device=device
        )

        cm.update(pred, gt)

        # batch 级 IoU（进度条用）
        batch_iou = compute_batch_iou(pred, gt)
        if batch_i == 0:
            print(f"  batch[0] IoU 示例：", end="")
            for name, val in batch_iou.items():
                disp = f"{val:.3f}" if val >= 0 else "N/A"
                print(f"{name}={disp}", end="  ")
            print()

    # ── 计算 epoch 级指标 ─────────────────────────────────────────────────────
    print("\n[计算 epoch 级指标]")
    metrics = cm.compute()
    print_metrics_report(metrics)

    # ── 验证指标数值合理性 ────────────────────────────────────────────────────
    assert 0 < metrics["pixel_accuracy"] <= 1.0,  "pixel_accuracy 超出范围"
    assert 0 < metrics["mean_iou"]       <= 1.0,  "mean_iou 超出范围"
    assert 0 < metrics["mean_f1"]        <= 1.0,  "mean_f1 超出范围"
    # 80% 准确率下，IoU 应在合理范围
    assert metrics["mean_iou"] > 0.5, \
        f"mean_iou={metrics['mean_iou']:.4f} 偏低，检查累积逻辑"

    # ── 混淆矩阵可视化 ────────────────────────────────────────────────────────
    print("[绘制混淆矩阵]")
    matrix_np = cm.get_matrix_numpy()

    plot_confusion_matrix(
        matrix_np,
        output_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    plot_confusion_matrix(
        matrix_np,
        output_dir / "confusion_matrix_counts.png",
        normalize=False,
    )

    # ── 验证 reset ────────────────────────────────────────────────────────────
    cm.reset()
    assert cm.matrix.sum().item() == 0, "reset() 未清零混淆矩阵"
    print("[✓] reset() 验证通过")

    print("\n[✓] Step 9 metrics.py 全部测试通过\n")


if __name__ == "__main__":
    _test()