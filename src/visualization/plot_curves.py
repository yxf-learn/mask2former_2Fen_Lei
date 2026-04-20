"""
plot_curves.py
────────────────────────────────────────────────────────────────────────────
Step 10：实时训练曲线绘制

功能：
  每 epoch 结束后由 trainer.py 调用，自动更新并保存：
    outputs/training_history.png

  包含 4 个子图：
    1. Train Loss vs Val Loss
    2. Mean IoU（前景类宏平均）
    3. 各前景类别 IoU（5 条彩色曲线）
    4. 学习率变化曲线

设计要点：
  - 使用 matplotlib Agg 后端，兼容无显示器的服务器环境
  - 每次调用重新绘制完整图像（非增量更新），确保断点续训后曲线完整
  - 若历史数据不足 2 个点，跳过绘制（避免单点图报错）
  - 所有子图共享 x 轴（epoch），便于对比各指标的同步变化

运行方式（独立测试）：
  python src/visualization/plot_curves.py
────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 必须在 import pyplot 之前设置，服务器环境无 display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ─── 常量 ─────────────────────────────────────────────────────────────────────

# 各前景类别的固定颜色（与 class_statistics.py 保持一致）
CLASS_COLORS = {
    "damage":   "#E74C3C",   # 红
}

FOREGROUND_NAMES = list(CLASS_COLORS.keys())

# 图像整体尺寸
FIG_WIDTH  = 18
FIG_HEIGHT = 12


# ─── 核心绘制函数 ─────────────────────────────────────────────────────────────

def plot_training_history(
    history,
    save_path: Path,
) -> None:
    """
    绘制完整训练曲线并保存为 PNG。

    Parameters
    ----------
    history   : TrainingHistory 实例（来自 trainer.py）
                或等效的含以下属性的对象：
                  .epochs       list[int]
                  .train_losses list[float]
                  .val_losses   list[float]
                  .mean_ious    list[float]
                  .class_ious   dict {class_name: list[float]}
                  .lrs          list[float]
    save_path : 输出路径，如 outputs/training_history.png
    """
    epochs      = history.epochs
    train_losses= history.train_losses
    val_losses  = history.val_losses
    mean_ious   = history.mean_ious
    class_ious  = history.class_ious
    lrs         = history.lrs

    # 数据不足时跳过（避免单点折线图）
    if len(epochs) < 2:
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.suptitle(
        f"训练过程监控  |  Epoch {epochs[-1]}  |  "
        f"Best mIoU = {max(mean_ious):.4f}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    _plot_loss(axes[0, 0], epochs, train_losses, val_losses)
    _plot_mean_iou(axes[0, 1], epochs, mean_ious)
    _plot_class_ious(axes[1, 0], epochs, class_ious)
    _plot_lr(axes[1, 1], epochs, lrs)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── 子图绘制函数 ─────────────────────────────────────────────────────────────

def _plot_loss(
    ax,
    epochs:       list,
    train_losses: list,
    val_losses:   list,
) -> None:
    """子图 1：Train Loss vs Val Loss。"""
    ax.plot(epochs, train_losses,
            color="#2980B9", linewidth=1.8, label="Train Loss",
            marker="o", markersize=3, markevery=_markevery(epochs))
    ax.plot(epochs, val_losses,
            color="#E74C3C", linewidth=1.8, label="Val Loss",
            marker="s", markersize=3, markevery=_markevery(epochs))

    # 标注最低 val loss 点
    best_val_idx = int(np.argmin(val_losses))
    ax.annotate(
        f"min={val_losses[best_val_idx]:.4f}",
        xy=(epochs[best_val_idx], val_losses[best_val_idx]),
        xytext=(10, 10), textcoords="offset points",
        fontsize=8, color="#E74C3C",
        arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.0),
    )

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Train Loss vs Val Loss", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))


def _plot_mean_iou(
    ax,
    epochs:    list,
    mean_ious: list,
) -> None:
    """子图 2：Mean IoU 曲线。"""
    ax.plot(epochs, mean_ious,
            color="#27AE60", linewidth=2.0, label="Mean IoU",
            marker="D", markersize=3, markevery=_markevery(epochs))
    ax.fill_between(epochs, mean_ious, alpha=0.12, color="#27AE60")

    # 标注最高 mIoU 点
    best_idx = int(np.argmax(mean_ious))
    ax.annotate(
        f"best={mean_ious[best_idx]:.4f}\n@epoch {epochs[best_idx]}",
        xy=(epochs[best_idx], mean_ious[best_idx]),
        xytext=(10, -20), textcoords="offset points",
        fontsize=8, color="#27AE60",
        arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.0),
    )

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("mIoU", fontsize=10)
    ax.set_title("Mean IoU（前景类宏平均）", fontsize=11, fontweight="bold")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))


def _plot_class_ious(
    ax,
    epochs:     list,
    class_ious: dict,   # {class_name: list[float]}
) -> None:
    """子图 3：各前景类别 IoU 曲线（5 条线）。"""
    for cls_name in FOREGROUND_NAMES:
        iou_vals = class_ious.get(cls_name, [])
        if not iou_vals or len(iou_vals) != len(epochs):
            continue

        # 将 -1（类别不存在的 epoch）替换为 NaN，折线图自动断开
        iou_plot = [v if v >= 0 else float("nan") for v in iou_vals]

        ax.plot(
            epochs, iou_plot,
            color=CLASS_COLORS[cls_name],
            linewidth=1.6,
            label=cls_name,
            marker="o",
            markersize=2.5,
            markevery=_markevery(epochs),
        )

    # 标注每条线的最终 epoch 值（右侧标签）
    if epochs:
        last_epoch = epochs[-1]
        for cls_name in FOREGROUND_NAMES:
            iou_vals = class_ious.get(cls_name, [])
            if iou_vals:
                last_val = iou_vals[-1]
                if last_val >= 0:
                    ax.annotate(
                        f"{last_val:.3f}",
                        xy=(last_epoch, last_val),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=7.5,
                        color=CLASS_COLORS[cls_name],
                        va="center",
                    )

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("IoU", fontsize=10)
    ax.set_title("各类别 IoU", fontsize=11, fontweight="bold")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))


def _plot_lr(
    ax,
    epochs: list,
    lrs:    list,
) -> None:
    """子图 4：学习率变化曲线（对数纵坐标）。"""
    ax.plot(epochs, lrs,
            color="#8E44AD", linewidth=1.8, label="Learning Rate",
            marker="^", markersize=3, markevery=_markevery(epochs))

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Learning Rate", fontsize=10)
    ax.set_title("学习率调度（Warmup + Cosine Annealing）",
                 fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.1e}")
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle="--", which="both")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _markevery(epochs: list) -> int:
    """
    根据总 epoch 数动态决定 marker 间隔，避免点过密导致图像混乱。
    epoch 数 ≤ 20 时每点都画；更多时每 5 或 10 个 epoch 画一个。
    """
    n = len(epochs)
    if n <= 20:
        return 1
    elif n <= 50:
        return 5
    else:
        return 10


# ─── 从 JSON 恢复历史并绘图（断点续训 / 离线重绘）──────────────────────────

def plot_from_json(
    json_path: Path,
    save_path: Path,
) -> None:
    """
    从 training_history.json 加载历史数据并重绘曲线。
    用于训练中断后离线重绘，或在训练完成后生成高分辨率版本。
    """
    if not json_path.exists():
        raise FileNotFoundError(f"找不到历史文件：{json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 构造一个轻量的伪 history 对象
    class _HistoryProxy:
        pass

    h = _HistoryProxy()
    h.epochs       = data["epochs"]
    h.train_losses = data["train_losses"]
    h.val_losses   = data["val_losses"]
    h.mean_ious    = data["mean_ious"]
    h.class_ious   = data["class_ious"]
    h.lrs          = data["lrs"]

    plot_training_history(h, save_path)
    print(f"[✓] 已从 JSON 重绘训练曲线：{save_path}")


# ─── 独立测试入口 ─────────────────────────────────────────────────────────────

def _test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--json_path",   default=None,
                        help="若指定，从已有 JSON 文件重绘曲线")
    parser.add_argument("--num_epochs",  type=int, default=60,
                        help="合成数据的模拟 epoch 数")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 若指定 JSON 则从 JSON 重绘 ────────────────────────────────────────────
    if args.json_path:
        plot_from_json(
            Path(args.json_path),
            output_dir / "training_history.png",
        )
        return

    # ── 合成模拟训练数据 ──────────────────────────────────────────────────────
    print(f"[测试] 生成 {args.num_epochs} 个 epoch 的模拟训练数据...")

    N = args.num_epochs
    rng = np.random.default_rng(42)

    # 模拟 loss：指数衰减 + 轻微噪声
    base_train = 2.5 * np.exp(-np.linspace(0, 3, N)) + 0.15
    base_val   = 2.8 * np.exp(-np.linspace(0, 2.8, N)) + 0.20
    train_losses = (base_train + rng.normal(0, 0.04, N)).clip(0.1).tolist()
    val_losses   = (base_val   + rng.normal(0, 0.06, N)).clip(0.1).tolist()

    # 模拟 mIoU：从低到高，后期趋于平稳
    mean_ious = (
        0.75 * (1 - np.exp(-np.linspace(0, 4, N)))
        + rng.normal(0, 0.01, N)
    ).clip(0, 0.95).tolist()

    # 模拟各类别 IoU（各类收敛速度不同，反映类别不平衡）
    class_ious = {}
    start_vals = {"Transverse": 0.5, "Longitudinal": 0.6,
                  "Oblique": 0.45, "Alligator": 0.3, "fixpatch": 0.35}
    end_vals   = {"Transverse": 0.82, "Longitudinal": 0.85,
                  "Oblique": 0.74, "Alligator": 0.62, "fixpatch": 0.70}
    for cls_name in FOREGROUND_NAMES:
        s, e  = start_vals[cls_name], end_vals[cls_name]
        curve = s + (e - s) * (1 - np.exp(-np.linspace(0, 4, N)))
        curve += rng.normal(0, 0.012, N)
        class_ious[cls_name] = curve.clip(0, 1).tolist()

    # 模拟学习率（warmup 5 epoch + cosine 衰减）
    warmup = np.linspace(1e-7, 1e-4, 5)
    cosine = 1e-4 * 0.5 * (1 + np.cos(np.linspace(0, np.pi, N - 5)))
    lrs    = np.concatenate([warmup, cosine]).tolist()

    # ── 构建伪 history 对象 ───────────────────────────────────────────────────
    class _MockHistory:
        pass

    h = _MockHistory()
    h.epochs       = list(range(1, N + 1))
    h.train_losses = train_losses
    h.val_losses   = val_losses
    h.mean_ious    = mean_ious
    h.class_ious   = class_ious
    h.lrs          = lrs

    # ── 绘制 ─────────────────────────────────────────────────────────────────
    save_path = output_dir / "training_history.png"
    plot_training_history(h, save_path)
    print(f"[✓] 训练曲线已保存：{save_path}")

    # ── 验证文件是否生成 ──────────────────────────────────────────────────────
    assert save_path.exists(), "training_history.png 未生成"
    file_size_kb = save_path.stat().st_size / 1024
    print(f"[✓] 文件大小：{file_size_kb:.1f} KB")
    assert file_size_kb > 50, "文件过小，图像可能为空"

    # ── 测试 plot_from_json ───────────────────────────────────────────────────
    import json
    json_data = {
        "epochs":       h.epochs,
        "train_losses": h.train_losses,
        "val_losses":   h.val_losses,
        "mean_ious":    h.mean_ious,
        "class_ious":   h.class_ious,
        "lrs":          h.lrs,
    }
    json_path = output_dir / "training_history.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f)

    plot_from_json(json_path, output_dir / "training_history_from_json.png")
    assert (output_dir / "training_history_from_json.png").exists()
    print("[✓] plot_from_json 验证通过")

    print(f"\n[✓] Step 10 plot_curves.py 全部测试通过\n")


if __name__ == "__main__":
    _test()