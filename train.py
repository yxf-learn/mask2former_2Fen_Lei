"""
train.py
────────────────────────────────────────────────────────────────────────────
主训练入口

串联所有模块，一键启动完整训练流程：
  Step 4  → CrackDataset / DataLoader
  Step 5  → Mask2Former 模型 + 差异化学习率分组
  Step 8  → CrackSegLoss
  Step 9  → ConfusionMatrix + compute_iou_fn
  Step 10 → plot_training_history
  Step 7  → Trainer（接收 cm，内部管理 reset / compute）

运行方式：
  # 从头训练
  python train.py

  # 断点续训
  python train.py --resume

  # 指定配置文件
  python train.py --yaml configs/mask2former_crack.yaml
────────────────────────────────────────────────────────────────────────────
"""
import os
import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

plt.rcParams['font.sans-serif'] = [
    'WenQuanYi Zen Hei',
    'WenQuanYi Micro Hei',
    'Noto Sans CJK JP',
    'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False


# ─── 全局随机种子 ─────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="路面病害语义分割训练")
    parser.add_argument("--yaml",
                        default="configs/mask2former_crack.yaml")
    parser.add_argument("--split_meta",
                        default="data/processed/splits/split_meta.json")
    parser.add_argument("--aug_txt",
                        default="data/processed/splits/train_augmented.txt")
    parser.add_argument("--labeled_images",
                        default="data/labeled/images")
    parser.add_argument("--labeled_masks",
                        default="data/labeled/masks")
    parser.add_argument("--aug_images",
                        default="data/augmented/images")
    parser.add_argument("--aug_masks",
                        default="data/augmented/masks")
    parser.add_argument("--ckpt_dir",    default="checkpoints")
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--image_ext",   default=".jpg")
    parser.add_argument("--resume",      action="store_true",
                        help="从 checkpoints/latest.pth 断点续训")
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


# ─── 预检查 ───────────────────────────────────────────────────────────────────

def preflight_check(args) -> None:
    """训练前检查所有必要文件是否存在，提前暴露配置问题。"""
    errors = []
    required = {
        "配置文件":       args.yaml,
        "划分元数据":     args.split_meta,
        "增强训练集列表": args.aug_txt,
        "标注图像目录":   args.labeled_images,
        "标注 mask 目录": args.labeled_masks,
        "增强图像目录":   args.aug_images,
        "增强 mask 目录": args.aug_masks,
    }
    for name, path in required.items():
        if not Path(path).exists():
            errors.append(f"  ✗ {name} 不存在：{path}")

    if args.resume and not (Path(args.ckpt_dir) / "latest.pth").exists():
        errors.append(f"  ✗ 指定了 --resume 但找不到 {args.ckpt_dir}/latest.pth")

    if errors:
        print("\n[预检查失败] 以下文件 / 目录缺失：\n")
        for e in errors:
            print(e)
        print("\n  正确的运行顺序：")
        print("    1. python src/data_processing/class_statistics.py")
        print("    2. python src/data_processing/dataset_split.py")
        print("    3. python src/data_processing/augmentation.py")
        print("    4. python train.py")
        raise SystemExit(1)

    print("[✓] 预检查通过")


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    
    
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[设备] {device}")
    if device.type == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  显存 : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── 1. 预检查 ─────────────────────────────────────────────────────────────
    preflight_check(args)

    # ── 2. 延迟导入（预检查通过后再导入）─────────────────────────────────────
    from src.model.mask2former_config import build_model, load_config, get_param_groups
    from src.dataset.crack_dataset     import build_dataloader
    from src.training.losses           import build_loss_fn
    from src.training.metrics          import build_metric_fns
    from src.training.trainer          import Trainer
    from src.visualization.plot_curves import plot_training_history

    # ── 3. 加载配置 ───────────────────────────────────────────────────────────
    yaml_path = Path(args.yaml)
    cfg       = load_config(yaml_path)
    train_cfg = cfg["training"]

    print(f"\n[配置]")
    print(f"  image_size         : {train_cfg['image_size']}")
    print(f"  batch_size         : {train_cfg['batch_size']}")
    print(f"  accumulation_steps : {train_cfg['accumulation_steps']}")
    print(f"  等效 batch_size    : "
          f"{train_cfg['batch_size'] * train_cfg['accumulation_steps']}")
    print(f"  max_epochs         : {train_cfg['max_epochs']}")
    print(f"  lr                 : {train_cfg['lr']}")
    print(f"  mixed_precision    : {train_cfg['mixed_precision']}")

    # ── 4. DataLoader ─────────────────────────────────────────────────────────
    print(f"\n[数据] 构建 DataLoader...")
    loader_kwargs = dict(
        split_meta_path   = Path(args.split_meta),
        labeled_image_dir = Path(args.labeled_images),
        labeled_mask_dir  = Path(args.labeled_masks),
        aug_image_dir     = Path(args.aug_images),
        aug_mask_dir      = Path(args.aug_masks),
        aug_txt_path      = Path(args.aug_txt),
        batch_size        = train_cfg["batch_size"],
        num_workers       = train_cfg["num_workers"],
        image_ext         = args.image_ext,
    )
    train_loader = build_dataloader(split="train", **loader_kwargs)
    val_loader   = build_dataloader(split="val",   **loader_kwargs)
    print(f"  train batches : {len(train_loader)}")
    print(f"  val   batches : {len(val_loader)}")

    # ── 5. 模型 ───────────────────────────────────────────────────────────────
    print(f"\n[模型] 加载 Mask2Former（Swin-Base）...")
    model = build_model(cfg)

    print(f"\n[优化器] 差异化学习率分组：")
    param_groups = get_param_groups(
        model,
        base_lr=train_cfg["lr"],
        backbone_lr_multiplier=train_cfg["backbone_lr_multiplier"],
    )

    # ── 6. 损失函数 ───────────────────────────────────────────────────────────
    print(f"\n[损失函数] 构建 CrackSegLoss...")
    loss_fn = build_loss_fn(yaml_path, device)

    # ── 7. 评估函数 + 混淆矩阵 ───────────────────────────────────────────────
    print(f"\n[指标] 构建评估函数与混淆矩阵...")
    cm, compute_iou_fn = build_metric_fns(device)
    # cm 传入 Trainer，Trainer 在每个 epoch 的 validate 前后自动 reset / compute

    # ── 8. 绘图函数 ───────────────────────────────────────────────────────────
    def plot_fn(history, save_path):
        plot_training_history(history, save_path)

    # ── 9. 启动训练 ───────────────────────────────────────────────────────────
    print(f"\n[训练器] 初始化 Trainer...")
    trainer = Trainer(
        cfg             = cfg,
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        param_groups    = param_groups,
        device          = device,
        compute_loss_fn = loss_fn,
        compute_iou_fn  = compute_iou_fn,
        confusion_matrix= cm,          # Trainer 内部管理 reset / compute
        plot_fn         = plot_fn,
        ckpt_dir        = Path(args.ckpt_dir),
        output_dir      = Path(args.output_dir),
        resume          = args.resume,
    )

    trainer.train()


if __name__ == "__main__":
    main()