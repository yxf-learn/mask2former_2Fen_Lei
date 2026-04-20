"""
crack_dataset.py
────────────────────────────────────────────────────────────────────────────
Step 4：PyTorch Dataset 类

功能：
  1. 封装训练集（augmented）、验证集、测试集的数据读取逻辑
  2. 将图像和 mask 转换为 Mask2Former 所需的输入格式：
       pixel_values  : FloatTensor [3, H, W]，归一化图像
       mask_labels   : BoolTensor  [N, H, W]，每个前景实例的二值 mask
       class_labels  : LongTensor  [N]，每个实例对应的类别 ID
  3. 提供 collate_fn 处理同一 batch 内实例数量不等的情况
  4. 支持通过 split_meta.json 或纯 txt 文件两种方式指定样本列表

关于 Mask2Former 的输入格式：
  Mask2Former 是实例感知的全景/语义分割模型，其输入要求与传统语义分割不同：
  - 不接受单通道语义 mask（每像素一个类别 ID）
  - 接受一组二值 mask + 对应类别 ID 的列表
  - 对于语义分割任务，同一类别的所有像素合并为一个实例
  - 即：每个类别对应一个二值 mask，最多 5 个前景实例（5 类病害）

类别映射：
  0: background   → 不作为实例，忽略
  1: Transverse   → 实例 0（若存在）
  2: Longitudinal → 实例 1（若存在）
  3: Oblique      → 实例 2（若存在）
  4: Alligator    → 实例 3（若存在）
  5: fixpatch     → 实例 4（若存在）

运行方式（独立测试）：
  python src/dataset/crack_dataset.py

Step 4 的几个关键设计说明：
_mask_to_instances 的转换逻辑：Mask2Former 本质上是一个实例分割模型，即使做语义分割任务，也要把每个类别拆成独立的二值 mask。
这里遍历 5 个前景类别，凡是在当前图像中出现过像素的类别，就生成一个 bool [H, W] 的二值 mask，最终打包为 [N, H, W]，N 在不同图像之间是不同的（0~5）。

crack_collate_fn 的必要性：正因为 N 在不同样本之间不相等，标准的 default_collate 无法处理，必须自定义。
pixel_values 可以直接 stack（所有图像尺寸相同），而 mask_labels 和 class_labels 保持列表形式传给 Mask2Former，与其 forward 接口完全匹配。

纯背景图的处理：数据集中可能存在没有任何病害标注的图（全是 background），此时返回一个占位 dummy tensor 避免空 tensor 引发的 shape 错误，训练时该样本对损失的贡献自然为 0。
────────────────────────────────────────────────────────────────────────────
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


# ─── 全局常量 ────────────────────────────────────────────────────────────────

CLASSES = {
    0: "background",
    1: "damage",
}
NUM_CLASSES     = 2
FOREGROUND_IDS  = [1]   # 不含 background

# ImageNet 归一化参数（Swin backbone 预训练时使用）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── 主 Dataset 类 ────────────────────────────────────────────────────────────

class CrackDataset(Dataset):
    """
    路面病害语义分割 Dataset。

    Parameters
    ----------
    image_dir : Path
        图像根目录。
        训练集传入 data/augmented/images，val/test 传入 data/labeled/images。
    mask_dir : Path
        mask 根目录。
        训练集传入 data/augmented/masks，val/test 传入 data/labeled/masks。
    stems : list[str]
        样本的文件名主干列表（不含后缀），由 split_meta.json 或 txt 文件提供。
    image_ext : str
        图像文件后缀，默认 ".jpg"。
    mode : str
        "train" | "val" | "test"，当前仅用于日志信息，无行为差异。
    """

    def __init__(
        self,
        image_dir: Path,
        mask_dir:  Path,
        stems:     list,
        image_ext: str = ".jpg",
        mode:      str = "train",
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.stems     = stems
        self.image_ext = image_ext
        self.mode      = mode

        # 验证目录存在
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录不存在：{self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"mask 目录不存在：{self.mask_dir}")

        print(f"[CrackDataset] mode={mode}  样本数={len(stems)}")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        # ── 1. 读取图像与 mask ────────────────────────────────────────────────
        image, mask = self._load(stem)

        # ── 2. 图像归一化：BGR → RGB → float32 → ImageNet normalize ──────────
        pixel_values = self._normalize(image)   # [3, H, W] float32 tensor

        # ── 3. 将语义 mask 转换为 Mask2Former 所需格式 ────────────────────────
        mask_labels, class_labels = self._mask_to_instances(mask)
        # mask_labels  : BoolTensor  [N, H, W]，N = 该图中实际出现的前景类别数
        # class_labels : LongTensor  [N]

        return {
            "pixel_values": pixel_values,   # [3, H, W]
            "mask_labels":  mask_labels,    # [N, H, W]
            "class_labels": class_labels,   # [N]
            "stem":         stem,           # str，用于推理时对应输出文件名
        }

    # ─── 私有方法 ─────────────────────────────────────────────────────────────

    def _load(self, stem: str) -> tuple:
        """读取图像（BGR uint8）和 mask（灰度 uint8，值域 0~5）。"""
        image_path = self.image_dir / (stem + self.image_ext)
        mask_path  = self.mask_dir  / (stem + ".png")

        # 兼容增强样本（stem 如 "000001_aug003"）和原始样本（stem 如 "000001"）
        # 如果按 .jpg 找不到，尝试 .JPG / .jpeg
        if not image_path.exists():
            for ext in [".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                alt = self.image_dir / (stem + ext)
                if alt.exists():
                    image_path = alt
                    break

        if not image_path.exists():
            raise FileNotFoundError(f"找不到图像：{image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"找不到 mask：{mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask  = cv2.imread(str(mask_path),  cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise IOError(f"cv2 无法解码图像：{image_path}")
        if mask is None:
            raise IOError(f"cv2 无法解码 mask：{mask_path}")

        # 严格校验 mask 像素值范围
        max_val = int(mask.max())
        if max_val > 5:
            raise ValueError(
                f"{mask_path.name} 中存在非法像素值 {max_val}，"
                f"合法范围为 0~5"
            )

        return image, mask

    def _normalize(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        BGR uint8 → RGB float32 → ImageNet 归一化 → CHW tensor。

        步骤：
          1. BGR → RGB（Swin 预训练时使用 RGB 输入）
          2. /255.0 映射到 [0, 1]
          3. 减均值 / 除标准差（ImageNet 统计量）
          4. HWC → CHW
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_f   = image_rgb.astype(np.float32) / 255.0

        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std  = np.array(IMAGENET_STD,  dtype=np.float32)
        image_norm = (image_f - mean) / std   # [H, W, 3]

        # HWC → CHW
        image_chw = np.transpose(image_norm, (2, 0, 1))   # [3, H, W]
        return torch.from_numpy(image_chw).float()

    def _mask_to_instances(
        self, mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        将单通道语义 mask（值域 0~5）转换为 Mask2Former 实例格式。

        对于语义分割任务，每个前景类别视为一个独立实例：
          - 遍历 5 个前景类别（1~5）
          - 若该类别在 mask 中存在像素，则生成对应的二值 mask
          - 收集所有存在的类别 ID

        Returns
        -------
        mask_labels  : BoolTensor [N, H, W]
            N = 该图中实际出现的前景类别数（0 ≤ N ≤ 5）
        class_labels : LongTensor [N]
            每个实例的类别 ID（1~5）

        边界情况：
          若图像中无任何前景类别（纯背景图），
          返回空 tensor：mask_labels=[1,H,W](全False), class_labels=[0]
          避免 DataLoader collate 时因空 tensor 导致维度不一致。
        """
        h, w = mask.shape
        instance_masks  = []
        instance_labels = []

        for cls_id in FOREGROUND_IDS:
            binary = (mask == cls_id)   # bool ndarray [H, W]
            if binary.any():
                instance_masks.append(binary)
                instance_labels.append(cls_id)

        if len(instance_masks) == 0:
            # 纯背景图：返回占位 tensor，训练时该样本的 loss 贡献为 0
            dummy_mask  = torch.zeros((1, h, w), dtype=torch.float32)
            dummy_label = torch.tensor([0], dtype=torch.long)
            return dummy_mask, dummy_label

        mask_tensor  = torch.from_numpy(    
            np.stack(instance_masks, axis=0).astype(np.float32)   # [N, H, W] float32
        )
        label_tensor = torch.tensor(instance_labels, dtype=torch.long)

        return mask_tensor, label_tensor


# ─── Collate 函数 ─────────────────────────────────────────────────────────────

def crack_collate_fn(batch: list[dict]) -> dict:
    """
    自定义 collate 函数，处理同一 batch 内各样本实例数量 N 不同的情况。

    标准 torch.utils.data.default_collate 要求 batch 内所有 tensor
    在每个维度上完全一致，但不同图像含有的病害类别数 N 不同（0~5），
    因此必须使用自定义 collate：

      pixel_values : stack → [B, 3, H, W]（所有图像尺寸相同，可直接 stack）
      mask_labels  : list  → List[Tensor[N_i, H, W]]（保持列表，不 stack）
      class_labels : list  → List[Tensor[N_i]]
      stems        : list  → List[str]

    Mask2Former 的 forward() 接受 mask_labels 和 class_labels 为列表形式，
    与此 collate 输出完全兼容。
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)
    mask_labels  = [item["mask_labels"]  for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    stems        = [item["stem"]         for item in batch]

    return {
        "pixel_values": pixel_values,   # [B, 3, H, W]
        "mask_labels":  mask_labels,    # List[Tensor[N_i, H, W]]
        "class_labels": class_labels,   # List[Tensor[N_i]]
        "stems":        stems,          # List[str]
    }


# ─── 工厂函数 ─────────────────────────────────────────────────────────────────

def build_dataloader(
    split:          str,
    split_meta_path: Path,
    labeled_image_dir: Path,
    labeled_mask_dir:  Path,
    aug_image_dir:     Path,
    aug_mask_dir:      Path,
    aug_txt_path:      Path,
    batch_size:        int  = 2,
    num_workers:       int  = 4,
    image_ext:         str  = ".jpg",
) -> DataLoader:
    """
    根据 split 名称构建对应的 DataLoader。

    Parameters
    ----------
    split            : "train" | "val" | "test"
    split_meta_path  : split_meta.json 路径
    labeled_image_dir: data/labeled/images（val/test 使用）
    labeled_mask_dir : data/labeled/masks（val/test 使用）
    aug_image_dir    : data/augmented/images（train 使用）
    aug_mask_dir     : data/augmented/masks（train 使用）
    aug_txt_path     : data/processed/splits/train_augmented.txt（train 使用）
    """
    if not split_meta_path.exists():
        raise FileNotFoundError(
            f"找不到 split_meta.json：{split_meta_path}\n"
            f"请先运行 dataset_split.py"
        )

    with open(split_meta_path, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    if split == "train":
        if not aug_txt_path.exists():
            raise FileNotFoundError(
                f"找不到 train_augmented.txt：{aug_txt_path}\n"
                f"请先运行 augmentation.py"
            )
        with open(aug_txt_path, "r", encoding="utf-8") as f:
            stems = [line.strip() for line in f if line.strip()]
    
        dataset = CrackDataset(
            image_dir=aug_image_dir,
            mask_dir=aug_mask_dir,
            stems=stems,
            image_ext=image_ext,
            mode="train",
        )
        shuffle = True

    elif split in ("val", "test"):
        records = split_meta[split]
        stems   = [r["stem"] for r in records]

        dataset = CrackDataset(
            image_dir=labeled_image_dir,
            mask_dir=labeled_mask_dir,
            stems=stems,
            image_ext=image_ext,
            mode=split,
        )
        shuffle = False

    else:
        raise ValueError(f"split 必须为 'train' / 'val' / 'test'，收到：{split}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=crack_collate_fn,
        pin_memory=True,         # A100 环境下加速 CPU→GPU 传输
        persistent_workers=True if num_workers > 0 else False,
        drop_last=(split == "train"),   # 训练集丢弃最后不完整的 batch
    )

    return loader


# ─── 独立测试入口 ─────────────────────────────────────────────────────────────

def _test():
    """
    快速验证 Dataset 和 DataLoader 是否正常工作。
    运行：python src/dataset/crack_dataset.py
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_meta",    default="data/processed/splits/split_meta.json")
    parser.add_argument("--labeled_images",default="data/labeled/images")
    parser.add_argument("--labeled_masks", default="data/labeled/masks")
    parser.add_argument("--aug_images",    default="data/augmented/images")
    parser.add_argument("--aug_masks",     default="data/augmented/masks")
    parser.add_argument("--aug_txt",       default="data/processed/splits/train_augmented.txt")
    parser.add_argument("--split",         default="val", choices=["train","val","test"])
    parser.add_argument("--batch_size",    type=int, default=2)
    parser.add_argument("--num_workers",   type=int, default=0)
    args = parser.parse_args()

    loader = build_dataloader(
        split           = args.split,
        split_meta_path = Path(args.split_meta),
        labeled_image_dir = Path(args.labeled_images),
        labeled_mask_dir  = Path(args.labeled_masks),
        aug_image_dir     = Path(args.aug_images),
        aug_mask_dir      = Path(args.aug_masks),
        aug_txt_path      = Path(args.aug_txt),
        batch_size        = args.batch_size,
        num_workers       = args.num_workers,
    )

    print(f"\n[测试] split={args.split}  batch_size={args.batch_size}  "
          f"batches={len(loader)}")

    for i, batch in enumerate(loader):
        pv = batch["pixel_values"]
        ml = batch["mask_labels"]
        cl = batch["class_labels"]

        print(f"\n  batch [{i}]")
        print(f"    pixel_values : {pv.shape}  dtype={pv.dtype}  "
              f"min={pv.min():.3f}  max={pv.max():.3f}")

        for b_idx, (m, c) in enumerate(zip(ml, cl)):
            class_names = [CLASSES[int(x)] for x in c.tolist()]
            print(f"    sample[{b_idx}] mask_labels={m.shape} "
                  f"class_labels={c.tolist()} → {class_names}")

        if i >= 2:   # 只测前 3 个 batch
            print("\n  ... 已验证前 3 个 batch，数据加载正常 ✓")
            break


if __name__ == "__main__":
    _test()