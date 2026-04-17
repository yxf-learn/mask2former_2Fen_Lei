"""
augmentation.py
────────────────────────────────────────────────────────────────────────────
Step 3：训练集数据增强

功能：
  1. 读取 data/processed/splits/split_meta.json 中的 train 子集
  2. 对每张训练图像及其 mask 执行多种增强变换
  3. 将增强后的图像/mask 写入 data/augmented/images/ 和 data/augmented/masks/
  4. 同时将原始训练样本复制到 augmented 目录（增强集包含原图）
  5. 更新 data/processed/splits/train_augmented.txt

增强规则：
  ✓ 允许：90° / 180° / 270° 旋转（k=1,2,3）
  ✓ 允许：水平翻转
  ✓ 允许：亮度/对比度扰动、高斯噪声、Coarse Dropout
  ✗ 禁止：任意角度旋转（Transverse 旋转 30° 后特征接近 Oblique）
  ✗ 禁止：缩放、裁剪（必须保持 2720×1530 原分辨率）
  ✗ 禁止：透视变换、弹性形变

增强策略：
  - 每张原始图生成 N_AUG 个增强版本（默认 6），加上原图共 7 份
  - 旋转与翻转：确定性枚举，保证类别语义不变
  - 光度增强（亮度/对比度/噪声/Dropout）：随机组合，不影响 mask

命名规则：
  原图  000001.jpg  →  增强图  000001_aug001.jpg ... 000001_aug006.jpg
  原mask 000001.png →  增强mask 000001_aug001.png ... 000001_aug006.png

运行方式：
  python src/data_processing/augmentation.py
  python src/data_processing/augmentation.py \
      --image_dir data/labeled/images \
      --mask_dir  data/labeled/masks  \
      --split_meta data/processed/splits/split_meta.json \
      --out_image_dir data/augmented/images \
      --out_mask_dir  data/augmented/masks  \
      --n_aug 6 --seed 42
────────────────────────────────────────────────────────────────────────────

Step 3 数据增强的几个关键点说明：
restore_resolution 的必要性：90°/270° 旋转后图像变为 1530×2720（宽高互换），若不处理，DataLoader 在同一个 batch 内会遇到不同尺寸的 tensor 而报错。
这里采用等比缩放 + 中心裁剪的策略，既保持宽高比尽量一致，又强制回到 2720×1530。
几何变换与光度增强的分离：几何变换（旋转/翻转）是确定性的，image 和 mask 必须完全同步执行。
光度增强（亮度/噪声等）只改变像素值不改变语义，mask 不受影响，但仍同步传入 albumentations 的 additional_targets 接口以保持代码接口统一。
photo_seed 可复现性：每个增强配置持有独立的 photo_seed，确保相同 seed 下整个增强过程完全可复现，便于 debug 和消融实验。
输出规模：默认 n_aug=6，7 种几何组合循环分配，最终训练集 = 原始 965 张 × 7 = 6755 张

4种旋转 × 2种翻转 = 8种组合，去掉「不旋转+不翻转」（那就是原图），剩余 7种：
编号  rotate_k    do_hflip    实际效果
1       0           True    不旋转 + 水平翻转
2       1           False   顺时针90° + 不翻转
3       1           True    顺时针90° + 水平翻转
4       2           False   180° + 不翻转
5       2           True    180° + 水平翻转
6       3           False   顺时针270° + 不翻转
7       3           True顺时针270° + 水平翻转
加上原图本身，每张原始图共产生 8份（原图 + 7个增强版本）。
因此训练集规模 = 965 × 8 = 7720张。每种几何组合上还会叠加独立的光度增强（随机亮度/高斯噪声等），所以实际上这7个版本彼此之间在光度上也不完全相同。

"""

import os
import shutil
import argparse
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# ─── 常量 ─────────────────────────────────────────────────────────────────────

# 增强后图像的目标分辨率（严格保持原分辨率，此处仅作校验用）
IMG_H, IMG_W = 1530, 2720

# 允许的旋转角度（仅 90° 的整数倍）
ROTATE_Ks = [1, 2, 3]   # cv2.rotate: k=1→90°CW, k=2→180°, k=3→270°CW

# 每张原始图生成的增强版本数量（不含原图本身）
DEFAULT_N_AUG = 3


# ─── 增强 Pipeline 定义 ───────────────────────────────────────────────────────

def build_photometric_pipeline(seed: int) -> A.Compose:
    """
    纯光度增强管线（不改变几何形状，mask 不受影响但仍同步传入保持接口一致）。

    变换说明：
      RandomBrightnessContrast : 模拟不同时段/季节的光照差异
      GaussNoise               : 模拟无人机传感器噪声
      CoarseDropout            : 模拟路面污渍、落叶等局部遮挡
      RandomGamma              : 模拟曝光补偿差异
      ImageCompression         : 模拟 JPEG 压缩伪影（无人机传输损耗）

    所有变换的 p 值均 < 1.0，以 p=0.5 为主确保样本多样性。
    """
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7,
            ),
            A.GaussNoise(
                var_limit=(100.0, 625.0),   # sigma ≈ 10~25
                mean=0,
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=10,
                max_height=64,
                max_width=64,
                min_holes=5,
                min_height=16,
                min_width=16,
                fill_value=0,
                p=0.4,
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.4,
            ),
            A.ImageCompression(
                quality_lower=75,
                quality_upper=100,
                p=0.3,
            ),
        ],
        additional_targets={"mask": "mask"},
    )


# ─── 几何变换（确定性）────────────────────────────────────────────────────────

def rotate_90(image: np.ndarray, mask: np.ndarray, k: int):
    """
    顺时针旋转 k×90°。
    k=1 → 90°CW,  k=2 → 180°,  k=3 → 270°CW

    旋转后若图像变为竖向（1530宽×2720高），
    需 pad/crop 恢复到 2720×1530，此处采用等比缩放至长边=2720 后补黑边。

    注意：90°/270° 旋转后原始 2720×1530 变为 1530×2720（宽高互换），
    必须 resize 或 pad 回 2720×1530，否则后续 DataLoader 报错。
    策略：等比缩放（长边对齐 2720），短边两侧补零填充。
    """
    cv_codes = {
        1: cv2.ROTATE_90_CLOCKWISE,
        2: cv2.ROTATE_180,
        3: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    code = cv_codes[k]
    img_rot  = cv2.rotate(image, code)
    mask_rot = cv2.rotate(mask,  code)

    # 旋转后恢复到 IMG_W × IMG_H (2720 × 1530)
    img_rot  = restore_resolution(img_rot,  IMG_W, IMG_H, is_mask=False)
    mask_rot = restore_resolution(mask_rot, IMG_W, IMG_H, is_mask=True)

    return img_rot, mask_rot


def restore_resolution(
    arr: np.ndarray,
    target_w: int,
    target_h: int,
    is_mask: bool = False,
) -> np.ndarray:
    """
    将任意尺寸的图像等比缩放，使其短边对齐目标尺寸，
    然后中心裁剪（或补零填充）至 target_w × target_h。

    对于 mask 使用最近邻插值，对于图像使用双线性插值，
    确保 mask 中的像素值（类别 ID）不因插值而产生非整数值。
    """
    h, w = arr.shape[:2]

    if w == target_w and h == target_h:
        return arr

    # 等比缩放：以使缩放后 w=target_w 或 h=target_h，选覆盖目标区域的那个
    scale_w = target_w / w
    scale_h = target_h / h
    scale   = max(scale_w, scale_h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    arr_resized = cv2.resize(arr, (new_w, new_h), interpolation=interp)

    # 中心裁剪至 target_w × target_h
    crop_y = (new_h - target_h) // 2
    crop_x = (new_w - target_w) // 2
    arr_cropped = arr_resized[crop_y:crop_y + target_h, crop_x:crop_x + target_w]

    return arr_cropped


def hflip(image: np.ndarray, mask: np.ndarray):
    """水平翻转。不改变 Transverse / Longitudinal / Oblique 的语义。"""
    return cv2.flip(image, 1), cv2.flip(mask, 1)


# ─── 增强序列生成 ─────────────────────────────────────────────────────────────

def generate_aug_configs(n_aug: int, seed: int) -> list[dict]:
    """
    生成 n_aug 个确定性的增强配置，每个配置包含：
      rotate_k  : 旋转参数（0=不旋转，1/2/3=旋转90/180/270°）
      do_hflip  : 是否水平翻转
      photo_seed: 光度增强的随机种子（保证可复现）

    设计原则：
      - 旋转和翻转采用均匀枚举（保证各角度均被覆盖）
      - 光度增强完全随机（每次不同的噪声/亮度组合）
    """
    rng = random.Random(seed)

    # 所有可能的几何变换组合（4旋转 × 2翻转 = 8种，去掉 k=0+不翻转=原图）
    geo_combos = [
        (k, flip)
        for k in [0, 2]
        for flip in [False, True]
        if not (k == 0 and flip == False)   # 排除"无变换"（那就是原图）
    ]  # 共 3 种几何变换（去掉90°和270°旋转）

    configs = []
    for i in range(n_aug):
        geo = geo_combos[i % len(geo_combos)]
        configs.append({
            "aug_idx":    i + 1,
            "rotate_k":   geo[0],
            "do_hflip":   geo[1],
            "photo_seed": rng.randint(0, 99999),
        })

    return configs


# ─── 单样本增强 ───────────────────────────────────────────────────────────────

def augment_one(
    image:     np.ndarray,
    mask:      np.ndarray,
    config:    dict,
    photo_aug: A.Compose,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对单张 image + mask 执行一次完整增强（几何 + 光度）。

    流程：
      1. 几何变换（旋转 → 翻转），image 和 mask 完全同步
      2. 光度增强（仅作用于 image，mask 不变）
    """
    img_out  = image.copy()
    mask_out = mask.copy()

    # ── 1. 几何变换（确定性）──────────────────────────────────────────────────
    k = config["rotate_k"]
    if k > 0:
        img_out, mask_out = rotate_90(img_out, mask_out, k)

    if config["do_hflip"]:
        img_out, mask_out = hflip(img_out, mask_out)

    # ── 2. 光度增强（随机，仅影响图像）──────────────────────────────────────
    random.seed(config["photo_seed"])
    np.random.seed(config["photo_seed"])

    result = photo_aug(image=img_out, mask=mask_out)
    img_out  = result["image"]
    mask_out = result["mask"]

    # ── 3. 校验分辨率与 mask 像素值 ──────────────────────────────────────────
    assert img_out.shape[:2]  == (IMG_H, IMG_W), \
        f"增强后图像尺寸异常：{img_out.shape}"
    assert mask_out.shape[:2] == (IMG_H, IMG_W), \
        f"增强后 mask 尺寸异常：{mask_out.shape}"
    assert mask_out.max() <= 5, \
        f"增强后 mask 出现非法像素值：{mask_out.max()}"

    return img_out, mask_out


# ─── I/O 工具 ─────────────────────────────────────────────────────────────────

def load_image_mask(
    image_path: Path,
    mask_path:  Path,
) -> tuple[np.ndarray, np.ndarray]:
    """读取图像（BGR）和 mask（单通道灰度）。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"无法读取图像：{image_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"无法读取 mask：{mask_path}")

    return image, mask


def save_augmented(
    image:          np.ndarray,
    mask:           np.ndarray,
    stem:           str,
    aug_idx:        int,
    out_image_dir:  Path,
    out_mask_dir:   Path,
    image_ext:      str = ".jpg",
) -> tuple[str, str]:
    """
    保存增强后的图像和 mask，命名格式：{stem}_aug{aug_idx:03d}。
    Returns: (image_filename, mask_filename)
    """
    img_name  = f"{stem}_aug{aug_idx:03d}{image_ext}"
    mask_name = f"{stem}_aug{aug_idx:03d}.png"

    # 图像用 JPEG 质量 95 保存（平衡体积与质量）
    if image_ext.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(out_image_dir / img_name), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(out_image_dir / img_name), image)

    # mask 必须以 PNG 无损保存，保证像素值 0~5 不失真
    cv2.imwrite(str(out_mask_dir / mask_name), mask)

    return img_name, mask_name


# ─── 主增强循环 ───────────────────────────────────────────────────────────────

def run_augmentation(
    image_dir:     Path,
    mask_dir:      Path,
    train_records: list[dict],
    out_image_dir: Path,
    out_mask_dir:  Path,
    n_aug:         int = DEFAULT_N_AUG,
    seed:          int = 42,
) -> list[str]:
    """
    对所有训练样本执行增强，同时将原图复制到 augmented 目录。

    Returns
    -------
    augmented_stems : list[str]，所有增强后样本的 stem（含原图 stem）
    """
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    photo_aug = build_photometric_pipeline(seed)
    aug_configs = generate_aug_configs(n_aug, seed)

    all_stems      = []
    error_count    = 0
    skipped_count  = 0

    for record in tqdm(train_records, desc="数据增强", unit="img"):
        stem       = record["stem"]
        image_name = record["image_name"]
        mask_name  = record["mask_name"]

        image_path = image_dir / image_name
        mask_path  = mask_dir  / mask_name

        # ── 检查原始文件是否存在 ──────────────────────────────────────────────
        if not image_path.exists() or not mask_path.exists():
            print(f"  [警告] 找不到文件：{image_name} 或 {mask_name}，已跳过")
            skipped_count += 1
            continue

        try:
            image, mask = load_image_mask(image_path, mask_path)
        except IOError as e:
            print(f"  [错误] {e}")
            error_count += 1
            continue

        image_ext = image_path.suffix.lower()

        # ── 1. 原图复制到 augmented 目录 ──────────────────────────────────────
        dst_img  = out_image_dir / image_name
        dst_mask = out_mask_dir  / mask_name
        if not dst_img.exists():
            shutil.copy2(str(image_path), str(dst_img))
        if not dst_mask.exists():
            shutil.copy2(str(mask_path),  str(dst_mask))
        all_stems.append(stem)

        # ── 2. 逐配置增强 ─────────────────────────────────────────────────────
        for cfg in aug_configs:
            try:
                aug_img, aug_mask = augment_one(image, mask, cfg, photo_aug)
                save_augmented(
                    aug_img, aug_mask, stem, cfg["aug_idx"],
                    out_image_dir, out_mask_dir, image_ext
                )
                aug_stem = f"{stem}_aug{cfg['aug_idx']:03d}"
                all_stems.append(aug_stem)
            except Exception as e:
                print(f"  [错误] {stem} aug{cfg['aug_idx']:03d}: {e}")
                error_count += 1

    print(f"\n  原始训练样本数  ：{len(train_records)}")
    print(f"  每张生成增强数  ：{n_aug}")
    print(f"  增强后总样本数  ：{len(all_stems)}")
    print(f"  跳过（缺文件）  ：{skipped_count}")
    print(f"  错误数          ：{error_count}")

    return all_stems


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="路面病害训练集数据增强")
    parser.add_argument("--image_dir",     type=str, default="data/labeled/images")
    parser.add_argument("--mask_dir",      type=str, default="data/labeled/masks")
    parser.add_argument("--split_meta",    type=str,
                        default="data/processed/splits/split_meta.json")
    parser.add_argument("--out_image_dir", type=str, default="data/augmented/images")
    parser.add_argument("--out_mask_dir",  type=str, default="data/augmented/masks")
    parser.add_argument("--split_dir",     type=str,
                        default="data/processed/splits")
    parser.add_argument("--n_aug",         type=int, default=DEFAULT_N_AUG,
                        help=f"每张原图生成的增强版本数（默认 {DEFAULT_N_AUG}）")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    image_dir     = Path(args.image_dir)
    mask_dir      = Path(args.mask_dir)
    split_meta_p  = Path(args.split_meta)
    out_image_dir = Path(args.out_image_dir)
    out_mask_dir  = Path(args.out_mask_dir)
    split_dir     = Path(args.split_dir)

    # ── 1. 读取 split_meta.json ───────────────────────────────────────────────
    print(f"\n[Step 1/3] 读取划分元数据：{split_meta_p}")
    if not split_meta_p.exists():
        raise FileNotFoundError(
            f"找不到 {split_meta_p}，请先运行 dataset_split.py"
        )
    with open(split_meta_p, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    train_records = split_meta["train"]
    val_records   = split_meta["val"]
    test_records  = split_meta["test"]

    print(f"  train：{len(train_records)} 张")
    print(f"  val  ：{len(val_records)} 张（不增强，原样使用）")
    print(f"  test ：{len(test_records)} 张（不增强，原样使用）")

    # ── 2. 执行增强 ───────────────────────────────────────────────────────────
    print(f"\n[Step 2/3] 开始数据增强（n_aug={args.n_aug}）...")
    all_stems = run_augmentation(
        image_dir, mask_dir, train_records,
        out_image_dir, out_mask_dir,
        n_aug=args.n_aug, seed=args.seed,
    )

    # ── 3. 写入 train_augmented.txt ───────────────────────────────────────────
    print(f"\n[Step 3/3] 写入增强后训练集文件列表...")
    split_dir.mkdir(parents=True, exist_ok=True)
    aug_txt = split_dir / "train_augmented.txt"
    with open(aug_txt, "w", encoding="utf-8") as f:
        for stem in sorted(all_stems):
            f.write(stem + "\n")
    print(f"  [✓] train_augmented.txt  {len(all_stems)} 条  →  {aug_txt}")

    print(f"\n[✓] Step 3 数据增强全部完成。")
    print(f"    增强图像目录：{out_image_dir}  ({len(all_stems)} 张)")
    print(f"    增强 mask 目录：{out_mask_dir}")
    print(f"    增强后训练集列表：{aug_txt}\n")


if __name__ == "__main__":
    main()

