"""
trainer.py
────────────────────────────────────────────────────────────────────────────
Step 7：训练主逻辑

功能：
  1. AMP 混合精度训练（torch.cuda.amp）
  2. 梯度累积（accumulation_steps=4，等效 batch=8）
  3. Cosine Annealing + Linear Warmup 学习率调度
  4. 梯度裁剪（max_norm=1.0）
  5. 每 epoch 记录：train loss / val loss / mIoU / 各类别 IoU
  6. 实时调用 plot_curves.py 更新 training_history.png
  7. checkpoint 保存策略：
       best_miou.pth       ← val mIoU 最优时覆盖
       latest.pth          ← 每 epoch 结束覆盖
       epoch_{N:03d}.pth   ← 每 save_every_n_epochs 存档一次

混淆矩阵管理：
  Trainer 接收一个 ConfusionMatrix 实例（self.cm），
  在每个 epoch 的 validate 开始前调用 cm.reset()，
  validate 结束后调用 cm.compute() 获取精确指标，
  替换 validate_one_epoch 返回的 batch 级近似值。
  train.py 只需构造一个 cm 传入，无需任何额外包装。
────────────────────────────────────────────────────────────────────────────
"""

import time
import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from tqdm import tqdm


# ─── 训练历史数据结构 ─────────────────────────────────────────────────────────

class TrainingHistory:
    """
    记录并持久化训练过程中的所有指标。
    每 epoch 结束后追加一条记录，同时序列化为 JSON 供断点续训使用。
    """

    FOREGROUND_CLASSES = {
        1: "damage",
    }

    def __init__(self, save_path: Path):
        self.save_path      = save_path
        self.epochs:        list = []
        self.train_losses:  list = []
        self.val_losses:    list = []
        self.mean_ious:     list = []
        self.class_ious:    dict = {
            name: [] for name in self.FOREGROUND_CLASSES.values()
        }
        self.lrs:           list = []

    def append(
        self,
        epoch:      int,
        train_loss: float,
        val_loss:   float,
        mean_iou:   float,
        class_iou:  dict,
        lr:         float,
    ) -> None:
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.mean_ious.append(mean_iou)
        self.lrs.append(lr)
        for name in self.FOREGROUND_CLASSES.values():
            self.class_ious[name].append(class_iou.get(name, 0.0))

    def save_json(self) -> None:
        data = {
            "epochs":       self.epochs,
            "train_losses": self.train_losses,
            "val_losses":   self.val_losses,
            "mean_ious":    self.mean_ious,
            "class_ious":   self.class_ious,
            "lrs":          self.lrs,
        }
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, save_path: Path) -> "TrainingHistory":
        history = cls(save_path)
        if not save_path.exists():
            return history
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        history.epochs       = data.get("epochs",       [])
        history.train_losses = data.get("train_losses", [])
        history.val_losses   = data.get("val_losses",   [])
        history.mean_ious    = data.get("mean_ious",    [])
        history.class_ious   = data.get("class_ious",   history.class_ious)
        history.lrs          = data.get("lrs",          [])
        return history

    @property
    def best_miou(self) -> float:
        return max(self.mean_ious) if self.mean_ious else 0.0

    @property
    def last_epoch(self) -> int:
        return self.epochs[-1] if self.epochs else 0


# ─── 学习率调度器 ─────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer:          AdamW,
    warmup_epochs:      int,
    max_epochs:         int,
    steps_per_epoch:    int,
    accumulation_steps: int,
) -> SequentialLR:
    """
    构建 Linear Warmup + Cosine Annealing 复合调度器。
    调度粒度：每次 optimizer.step() 触发一次（非每 epoch）。
    """
    opt_steps_per_epoch = max(1, steps_per_epoch // accumulation_steps)
    warmup_opt_steps    = warmup_epochs * opt_steps_per_epoch
    total_opt_steps     = max_epochs   * opt_steps_per_epoch
    cosine_opt_steps    = total_opt_steps - warmup_opt_steps

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_opt_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_opt_steps,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_opt_steps],
    )

    print(f"[调度器] warmup_opt_steps={warmup_opt_steps}  "
          f"cosine_opt_steps={cosine_opt_steps}  "
          f"total_opt_steps={total_opt_steps}")

    return scheduler


# ─── 单 epoch 训练 ────────────────────────────────────────────────────────────

def train_one_epoch(
    model:              nn.Module,
    loader:             DataLoader,
    optimizer:          AdamW,
    scheduler,
    scaler:             GradScaler,
    compute_loss_fn:    Callable,
    accumulation_steps: int,
    device:             torch.device,
    epoch:              int,
) -> float:
    """执行一个 epoch 的训练，返回平均训练 loss。"""
    model.train()
    total_loss    = 0.0
    num_opt_steps = 0
    accum_loss    = 0.0

    pbar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch:03d} [train]",
        leave=False,
    )

    optimizer.zero_grad()

    for step, batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        mask_labels  = [m.to(device) for m in batch["mask_labels"]]
        class_labels = [c.to(device) for c in batch["class_labels"]]

        with autocast(device_type="cuda"):
            loss        = compute_loss_fn(model, pixel_values, mask_labels, class_labels)
            # NaN/inf 保护：跳过问题 batch，不污染梯度
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
                
            loss_scaled = loss / accumulation_steps

        scaler.scale(loss_scaled).backward()
        accum_loss += loss.item()

        is_last_step      = (step == len(loader) - 1)
        is_accum_complete = ((step + 1) % accumulation_steps == 0)

        if is_accum_complete or is_last_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            actual_steps   = (
                accumulation_steps if is_accum_complete
                else ((step % accumulation_steps) + 1)
            )
            avg_block_loss = accum_loss / actual_steps
            total_loss    += avg_block_loss
            num_opt_steps += 1
            accum_loss     = 0.0

            current_lr = scheduler.get_last_lr()[-1]
            pbar.set_postfix({
                "loss": f"{avg_block_loss:.4f}",
                "lr":   f"{current_lr:.2e}",
            })
            # 定期释放显存碎片（每 200 个 optimizer step 清理一次）
            if num_opt_steps % 200 == 0:
                torch.cuda.empty_cache()
    return total_loss / max(num_opt_steps, 1)


# ─── 单 epoch 验证 ────────────────────────────────────────────────────────────

@torch.no_grad()
def validate_one_epoch(
    model:           nn.Module,
    loader:          DataLoader,
    compute_loss_fn: Callable,
    compute_iou_fn:  Callable,
    device:          torch.device,
    epoch:           int,
) -> float:
    """
    执行一个 epoch 的验证，返回 val_loss。

    IoU 不在此处返回——调用方在此函数返回后通过 cm.compute() 获取精确指标。
    compute_iou_fn 内部持有 cm 引用，每次调用时自动更新 cm。
    """
    model.eval()
    total_loss  = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False)

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        mask_labels  = [m.to(device) for m in batch["mask_labels"]]
        class_labels = [c.to(device) for c in batch["class_labels"]]

        loss        = compute_loss_fn(model, pixel_values, mask_labels, class_labels)
        # NaN 保护：跳过问题 batch，不计入统计
        if not torch.isfinite(loss):
            continue
        total_loss += loss.item()
        num_batches += 1

        # 推理并更新混淆矩阵
        outputs         = model(pixel_values=pixel_values)
        semantic_logits = _get_semantic_logits(outputs, pixel_values.shape[-2:])
        pred_masks      = semantic_logits.argmax(dim=1)
        gt_masks        = _instances_to_semantic(
            mask_labels, class_labels, pixel_values.shape[-2:], device
        )
        compute_iou_fn(pred_masks, gt_masks)   # 更新 cm，返回值不使用

        pbar.set_postfix({"val_loss": f"{total_loss/num_batches:.4f}"})

    return total_loss / max(num_batches, 1)


# ─── 辅助：Mask2Former 输出转语义 logits ─────────────────────────────────────

def _get_semantic_logits(outputs, target_size: tuple) -> torch.Tensor:
    """
    将 Mask2Former 的 query 输出转换为语义分割 logits [B, C, H, W]。

    公式：
      semantic_logits[b,c,h,w] =
          sum_q( softmax(class_logits)[b,q,c] × sigmoid(mask_logits)[b,q,h,w] )
    """
    import torch.nn.functional as F

    masks_logits = outputs.masks_queries_logits.float()   # [B, Q, H', W']
    class_logits = outputs.class_queries_logits.float()   # [B, Q, C+1]
    H, W         = target_size

    class_prob    = F.softmax(class_logits, dim=-1)[:, :, :-1]   # [B, Q, C]
    masks_up      = F.interpolate(
        masks_logits.float(), size=(H, W),
        mode="bilinear", align_corners=False,
    )
    masks_sigmoid = torch.sigmoid(masks_up)                       # [B, Q, H, W]

    return torch.einsum("bqc,bqhw->bchw", class_prob, masks_sigmoid)


def _instances_to_semantic(
    mask_labels:  list,
    class_labels: list,
    spatial_size: tuple,
    device:       torch.device,
) -> torch.Tensor:
    """将 Mask2Former 实例格式还原为单通道语义 mask [B, H, W]。"""
    H, W     = spatial_size
    B        = len(mask_labels)
    semantic = torch.zeros(B, H, W, dtype=torch.long, device=device)
    for b in range(B):
        m, c = mask_labels[b], class_labels[b]
        for i in range(m.shape[0]):
            cls_id = int(c[i])
            if cls_id != 0:
                semantic[b][m[i].bool()] = cls_id
    return semantic


# ─── Checkpoint 管理 ──────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module, optimizer: AdamW, scheduler,
    scaler: GradScaler, history: TrainingHistory,
    epoch: int, ckpt_dir: Path, tag: str,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "best_miou":            history.best_miou,
        },
        ckpt_dir / f"{tag}.pth",
    )


def load_checkpoint(
    ckpt_path: Path, model: nn.Module, optimizer: AdamW,
    scheduler, scaler: GradScaler, device: torch.device,
) -> int:
    if not ckpt_path.exists():
        print(f"[断点续训] 未找到 {ckpt_path}，从头开始训练")
        return 0
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"[断点续训] 从 epoch {ckpt['epoch']} 恢复，"
          f"历史最优 mIoU={ckpt['best_miou']:.4f}")
    return ckpt["epoch"]


# ─── 主训练器 ─────────────────────────────────────────────────────────────────

class Trainer:
    """
    封装完整训练流程。

    confusion_matrix 参数说明：
      传入 Step 9 的 ConfusionMatrix 实例。
      Trainer 在每个 epoch 的 validate 开始前调用 cm.reset()，
      validate 结束后调用 cm.compute() 获取精确 IoU，
      无需任何外部包装类。
    """

    def __init__(
        self,
        cfg:             dict,
        model:           nn.Module,
        train_loader:    DataLoader,
        val_loader:      DataLoader,
        param_groups:    list,
        device:          torch.device,
        compute_loss_fn: Callable,
        compute_iou_fn:  Callable,
        confusion_matrix,                    # ConfusionMatrix 实例
        plot_fn:         Callable,
        ckpt_dir:        Path = Path("checkpoints"),
        output_dir:      Path = Path("outputs"),
        resume:          bool = False,
    ):
        self.cfg          = cfg
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.ckpt_dir     = ckpt_dir
        self.output_dir   = output_dir
        self.cm           = confusion_matrix

        self.compute_loss = compute_loss_fn
        self.compute_iou  = compute_iou_fn
        self.plot         = plot_fn

        train_cfg               = cfg["training"]
        self.max_epochs         = train_cfg["max_epochs"]
        self.accumulation_steps = train_cfg["accumulation_steps"]
        self.save_every         = train_cfg["save_every_n_epochs"]
        self.use_amp            = train_cfg["mixed_precision"]

        self.optimizer = AdamW(param_groups, weight_decay=1e-4, betas=(0.9, 0.999))
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_epochs=train_cfg["warmup_epochs"],
            max_epochs=self.max_epochs,
            steps_per_epoch=len(train_loader),
            accumulation_steps=self.accumulation_steps,
        )
        self.scaler    = GradScaler(device="cuda", enabled=self.use_amp)

        history_path   = output_dir / "training_history.json"
        self.history   = TrainingHistory(history_path)
        self.best_miou = 0.0

        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 0
        if resume:
            self.start_epoch = load_checkpoint(
                ckpt_dir / "latest.pth",
                self.model, self.optimizer,
                self.scheduler, self.scaler, device,
            )
            self.history   = TrainingHistory.load_json(history_path)
            self.best_miou = self.history.best_miou

    # ─── 主训练循环 ───────────────────────────────────────────────────────────

    def train(self) -> None:
        batch_size = self.cfg["training"]["batch_size"]
        print(f"\n{'='*65}")
        print(f"  开始训练  |  总 epochs: {self.max_epochs}"
              f"  |  设备: {self.device}  |  AMP: {self.use_amp}")
        print(f"  accumulation_steps: {self.accumulation_steps}"
              f"  |  等效 batch: {batch_size * self.accumulation_steps}")
        print(f"{'='*65}\n")

        for epoch in range(self.start_epoch + 1, self.max_epochs + 1):
            epoch_start = time.time()

            # ── 训练 ─────────────────────────────────────────────────────────
            train_loss = train_one_epoch(
                model=self.model, loader=self.train_loader,
                optimizer=self.optimizer, scheduler=self.scheduler,
                scaler=self.scaler, compute_loss_fn=self.compute_loss,
                accumulation_steps=self.accumulation_steps,
                device=self.device, epoch=epoch,
            )

            # ── 验证（reset → validate → compute 精确指标）────────────────────
            self.cm.reset()
            val_loss = validate_one_epoch(
                model=self.model, loader=self.val_loader,
                compute_loss_fn=self.compute_loss,
                compute_iou_fn=self.compute_iou,   # 内部自动更新 cm
                device=self.device, epoch=epoch,
            )

            # cm 已累积完整 val 集，compute() 给出精确指标
            precise  = self.cm.compute()
            mean_iou = precise["mean_iou"]
            class_iou = {
                k: v for k, v in precise["iou_per_class"].items()
                if k != "background"
            }

            epoch_time = time.time() - epoch_start
            current_lr = max(g["lr"] for g in self.optimizer.param_groups)

            # ── 记录 & 保存历史 ───────────────────────────────────────────────
            self.history.append(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                mean_iou=mean_iou, class_iou=class_iou, lr=current_lr,
            )
            self.history.save_json()
            self._print_epoch_log(
                epoch, train_loss, val_loss, mean_iou,
                class_iou, current_lr, epoch_time,
            )

            # ── Checkpoint ────────────────────────────────────────────────────
            is_best = mean_iou > self.best_miou
            if is_best:
                self.best_miou = mean_iou
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.scaler, self.history, epoch,
                    self.ckpt_dir, tag="best_miou",
                )
                print(f"  [★] 新最优 mIoU={mean_iou:.4f}，已保存 best_miou.pth")

            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                self.scaler, self.history, epoch,
                self.ckpt_dir, tag="latest",
            )
            if epoch % self.save_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    self.scaler, self.history, epoch,
                    self.ckpt_dir, tag=f"epoch_{epoch:03d}",
                )

            # ── 实时训练曲线 ──────────────────────────────────────────────────
            self.plot(
                self.history,
                save_path=self.output_dir / "training_history.png",
            )

        print(f"\n{'='*65}")
        print(f"  训练完成！最优 val mIoU = {self.best_miou:.4f}")
        print(f"  最优模型权重：{self.ckpt_dir / 'best_miou.pth'}")
        print(f"  训练曲线    ：{self.output_dir / 'training_history.png'}")
        print(f"{'='*65}\n")

    # ─── 日志 ─────────────────────────────────────────────────────────────────

    def _print_epoch_log(
        self, epoch, train_loss, val_loss,
        mean_iou, class_iou, lr, elapsed,
    ) -> None:
        print(
            f"\nEpoch [{epoch:03d}/{self.max_epochs}]  "
            f"time={elapsed:.1f}s  lr={lr:.2e}\n"
            f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"mIoU={mean_iou:.4f}  best_mIoU={self.best_miou:.4f}"
        )
        print("  per-class IoU:")
        for cls_name, iou_val in class_iou.items():
            bar = "█" * int(iou_val * 30) + "░" * (30 - int(iou_val * 30))
            print(f"    {cls_name:<14} {bar} {iou_val:.4f}")