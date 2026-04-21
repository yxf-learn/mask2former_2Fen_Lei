"""
crack_direction.py
────────────────────────────────────────────────────────────────────────────
线状裂缝方向分类模块
2026-04-15新增模块
负责把 linear_crack 细分为横/纵/斜
输入：语义分割输出的 linear_crack mask（像素值=1）
输出：细分后的方向 mask，像素值定义如下：
  0 → background（不变）
  1 → Transverse（横向，θ ∈ [0°,30°] 或 [150°,180°]）
  2 → Longitudinal（纵向，θ ∈ [60°,120°]）
  3 → Oblique（斜向，θ ∈ (30°,60°) 或 (120°,150°)）
  4 → Alligator（不变）
  5 → fixpatch（不变）

方向角 θ 定义：
  裂缝连通域主轴与水平方向的夹角，范围 [0°, 180°]
  使用 cv2.minAreaRect 的旋转角度计算
  
分类规则：
  横向：θ ∈ [0°, 30°] ∪ [150°, 180°]
  纵向：θ ∈ [60°, 120°]
  斜向：θ ∈ (30°, 60°) ∪ (120°, 150°)
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import cv2
from pathlib import Path


# ─── 常量 ─────────────────────────────────────────────────────────────────────

# 输入 mask 中各类别的像素值
LINEAR_CRACK_ID = 1
ALLIGATOR_ID    = 2
FIXPATCH_ID     = 3

# 输出 mask 中细分后的像素值
OUTPUT_CLASSES = {
    0: "background",
    1: "Transverse",
    2: "Longitudinal",
    3: "Oblique",
    4: "Alligator",
    5: "fixpatch",
}

# 方向分类角度阈值（单位：度）
TRANSVERSE_RANGES   = [(0, 30), (150, 180)]
LONGITUDINAL_RANGES = [(60, 120)]
# 其余区间自动归为 Oblique：(30,60) 和 (120,150)

# 最小连通域面积（过小的连通域跳过方向判断，直接归入横向）
MIN_AREA_FOR_DIRECTION = 50


# ─── 方向角计算 ───────────────────────────────────────────────────────────────

def compute_orientation(component_mask: np.ndarray) -> float:
    """
    计算单个连通域的主方向角 θ（与水平方向的夹角，范围 [0°, 180°]）。

    方法：使用 cv2.minAreaRect 获取最小旋转外接矩形，
    从矩形的角度推算主轴方向。

    cv2.minAreaRect 返回的角度范围是 [-90°, 0°]，需要转换：
      - 若矩形宽 >= 高：主轴水平，θ = |angle|
      - 若矩形宽 < 高：主轴垂直，θ = 90° + |angle|

    Returns
    -------
    theta : float，范围 [0°, 180°]
    """
    contours, _ = cv2.findContours(
        component_mask.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return 0.0

    main_contour = max(contours, key=cv2.contourArea)

    if len(main_contour) < 5:
        # 点数不足时用边界框估算
        x, y, w, h = cv2.boundingRect(main_contour)
        return 0.0 if w >= h else 90.0

    rect  = cv2.minAreaRect(main_contour)
    angle = rect[2]        # [-90°, 0°]
    w, h  = rect[1]        # 宽和高

    if w >= h:
        theta = abs(angle)           # 主轴接近水平
    else:
        theta = 90.0 + abs(angle)    # 主轴接近垂直

    # 确保在 [0°, 180°] 范围内
    theta = theta % 180.0
    return theta


def classify_direction(theta: float) -> int:
    """
    根据方向角 θ 返回裂缝方向类别的输出像素值。

    Returns
    -------
    1 → Transverse
    2 → Longitudinal
    3 → Oblique
    """
    for lo, hi in TRANSVERSE_RANGES:
        if lo <= theta <= hi:
            return 1   # Transverse

    for lo, hi in LONGITUDINAL_RANGES:
        if lo <= theta <= hi:
            return 2   # Longitudinal

    return 3   # Oblique


# ─── 主处理函数 ───────────────────────────────────────────────────────────────

def classify_crack_direction(
    seg_mask:       np.ndarray,
    min_area:       int = MIN_AREA_FOR_DIRECTION,
    verbose:        bool = False,
) -> tuple[np.ndarray, dict]:
    """
    将语义分割输出的 linear_crack 连通域按方向细分。

    Parameters
    ----------
    seg_mask : uint8 ndarray [H, W]
        语义分割输出，像素值：
          0=background, 1=linear_crack, 2=Alligator, 3=fixpatch
    min_area : int
        最小连通域面积，低于此值直接归为 Transverse（细小裂缝默认横向）
    verbose  : bool
        是否打印每个连通域的方向判断结果

    Returns
    -------
    direction_mask : uint8 ndarray [H, W]
        细分后的 mask，像素值：
          0=background, 1=Transverse, 2=Longitudinal,
          3=Oblique, 4=Alligator, 5=fixpatch
    stats : dict
        统计信息：各方向连通域数量
    """
    H, W           = seg_mask.shape
    direction_mask = np.zeros((H, W), dtype=np.uint8)

    # ── 直接复制非线状裂缝类别（像素值重映射）────────────────────────────────
    direction_mask[seg_mask == 0] = 0   # background
    direction_mask[seg_mask == 2] = 4   # Alligator → 4
    direction_mask[seg_mask == 3] = 5   # fixpatch  → 5

    # ── 对 linear_crack 连通域逐个判断方向 ───────────────────────────────────
    linear_binary = (seg_mask == LINEAR_CRACK_ID).astype(np.uint8)

    if linear_binary.sum() == 0:
        return direction_mask, {
            "total": 0, "Transverse": 0,
            "Longitudinal": 0, "Oblique": 0
        }

    num_labels, label_map, stats_cv, _ = cv2.connectedComponentsWithStats(
        linear_binary, connectivity=8
    )

    counts = {"total": 0, "Transverse": 0, "Longitudinal": 0, "Oblique": 0}

    for lbl in range(1, num_labels):
        component = (label_map == lbl)
        area      = int(component.sum())

        if area < min_area:
            # 过小连通域默认归为横向
            direction_mask[component] = 1
            counts["Transverse"] += 1
            counts["total"]      += 1
            continue

        # 计算方向角
        theta      = compute_orientation(component.astype(np.uint8))
        direction  = classify_direction(theta)
        direction_mask[component] = direction

        dir_name = OUTPUT_CLASSES[direction]
        counts[dir_name] += 1
        counts["total"]  += 1

        if verbose:
            print(f"  label={lbl:>4}  area={area:>7}  "
                  f"θ={theta:>6.1f}°  → {dir_name}")

    return direction_mask, counts


def classify_damage_full(
    seg_mask: np.ndarray,
    alligator_area_threshold: int  = 5000,   # 面积大于此值且形状不规则 → Alligator
    fixpatch_rect_threshold:  float = 0.6,   # 矩形度大于此值 → fixpatch
    min_area: int = MIN_AREA_FOR_DIRECTION,
) -> np.ndarray:
    """
    将2类分割结果（background/damage）细分为6类。

    细分规则：
      对每个 damage 连通域提取几何特征：
      1. 长细比（aspect_ratio = 长轴/短轴）
         < 5  → 面状病害（Alligator 或 fixpatch）
         >= 5 → 线状裂缝（Transverse / Longitudinal / Oblique）

      2. 面状病害的进一步区分：
         矩形度（rectangularity = 面积/外接矩形面积）> fixpatch_rect_threshold
         → fixpatch（修补区域边界规整）
         否则 → Alligator（龟裂边界不规整）

      3. 线状裂缝按方向角细分（复用 classify_direction）

    输出像素值：
      0=background, 1=Transverse, 2=Longitudinal,
      3=Oblique, 4=Alligator, 5=fixpatch
    """
    H, W           = seg_mask.shape
    direction_mask = np.zeros((H, W), dtype=np.uint8)

    damage_binary = (seg_mask == 1).astype(np.uint8)
    if damage_binary.sum() == 0:
        return direction_mask

    num_labels, label_map, stats_cv, _ = cv2.connectedComponentsWithStats(
        damage_binary, connectivity=8
    )

    for lbl in range(1, num_labels):
        component = (label_map == lbl)
        area      = int(component.sum())

        if area < min_area:
            direction_mask[component] = 1   # 过小默认横向
            continue

        # 计算几何特征
        coords  = np.where(component)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        bbox_w  = x_max - x_min + 1
        bbox_h  = y_max - y_min + 1

        contours, _ = cv2.findContours(
            component.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            direction_mask[component] = 1
            continue

        main_contour = max(contours, key=cv2.contourArea)

        # 最小外接矩形
        if len(main_contour) >= 5:
            rect        = cv2.minAreaRect(main_contour)
            rw, rh      = rect[1]
            major_axis  = float(max(rw, rh))
            minor_axis  = float(min(rw, rh))
        else:
            major_axis = float(max(bbox_w, bbox_h))
            minor_axis = float(min(bbox_w, bbox_h))

        aspect_ratio  = major_axis / minor_axis if minor_axis > 0 else 999
        rect_area     = major_axis * minor_axis
        rectangularity = area / rect_area if rect_area > 0 else 0

        # 分类判断
        if aspect_ratio >= 5:
            # 线状裂缝 → 按方向细分
            theta     = compute_orientation(component.astype(np.uint8))
            direction = classify_direction(theta)
            direction_mask[component] = direction
        else:
            # 面状病害 → 区分 fixpatch 和 Alligator
            if rectangularity > fixpatch_rect_threshold:
                direction_mask[component] = 5   # fixpatch
            else:
                direction_mask[component] = 4   # Alligator

    return direction_mask


# ─── Batch 处理接口 ───────────────────────────────────────────────────────────

def classify_direction_batch(
    seg_masks: np.ndarray,
    min_area:  int  = MIN_AREA_FOR_DIRECTION,
    verbose:   bool = False,
) -> tuple[np.ndarray, list]:
    """
    对一个 batch 的 mask 逐张处理。

    Parameters
    ----------
    seg_masks : uint8 ndarray [B, H, W]

    Returns
    -------
    direction_masks : ndarray [B, H, W]
    batch_stats     : List[dict]
    """
    B = seg_masks.shape[0]
    direction_masks = np.zeros_like(seg_masks)
    batch_stats     = []

    for i in range(B):
        dm, stats = classify_crack_direction(
            seg_masks[i], min_area, verbose
        )
        direction_masks[i] = dm
        batch_stats.append(stats)

    return direction_masks, batch_stats


# ─── 独立测试入口 ─────────────────────────────────────────────────────────────

def _test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法读取：{args.mask}")
            return
        dm, stats = classify_crack_direction(mask, verbose=args.verbose)
        print(f"\n方向分类统计：{stats}")
        out_path = Path(args.mask).stem + "_direction.png"
        cv2.imwrite(out_path, dm)
        print(f"已保存：{out_path}")
        return

    # 合成测试
    print("[合成数据测试]")
    H, W    = 1530, 2720
    test_mask = np.zeros((H, W), dtype=np.uint8)

    # 模拟横向裂缝（接近水平）
    test_mask[300, 100:800] = 1

    # 模拟纵向裂缝（接近垂直）
    test_mask[200:900, 1200] = 1

    # 模拟斜向裂缝（45°）
    for i in range(300):
        test_mask[600 + i, 1500 + i] = 1

    # 模拟龟裂
    test_mask[800:900, 2000:2200] = 2

    # 模拟修补
    test_mask[400:500, 400:600] = 3

    dm, stats = classify_crack_direction(test_mask, verbose=True)
    print(f"\n方向分类统计：{stats}")

    # 验证输出像素值
    assert (dm[300, 100:800] == 1).all(),  "横向裂缝应为 Transverse(1)"
    assert (dm[200:900, 1200] == 2).all(), "纵向裂缝应为 Longitudinal(2)"
    assert (dm[800:900, 2000:2200] == 4).all(), "龟裂应为 Alligator(4)"
    assert (dm[400:500, 400:600] == 5).all(),   "修补应为 fixpatch(5)"

    print("\n[✓] crack_direction.py 测试通过\n")


if __name__ == "__main__":
    _test()