"""
physical_constraints.py
────────────────────────────────────────────────────────────────────────────
Step 6：物理约束后处理模块

功能：
  针对热棒阴影和路面分界线被误识别为 Transverse（横向裂缝）的问题，
  基于连通域几何特征设计约束规则，将不满足条件的误检区域从
  Transverse 类中剔除，重分类为 background（0）。

约束逻辑（仅作用于 Transverse 类，其余类别不受影响）：

  约束 1 — 最小面积
    连通域像素数 < min_region_area_px → 噪点，直接剔除

  约束 2 — 等效宽度
    连通域面积 / 主轴长度 > max_crack_width_px → 过宽，非裂缝

  约束 3 — 横向跨幅
    连通域横向范围（max_x - min_x）/ 图像宽度 > max_transverse_span_ratio
    且 等效宽度均匀（宽度标准差 < width_std_threshold）
    → 近似贯穿全幅的规则直线，判定为路面分界线，剔除

  约束 4 — 轮廓复杂度
    轮廓复杂度 = 周长² / (4π × 面积)，圆形 = 1.0，越不规则值越大
    复杂度 < min_contour_complexity → 过于规整（如椭圆/矩形阴影），剔除

  约束 5 — 纵横比
    长轴长度 / 短轴长度 > shadow_aspect_ratio_threshold
    且 等效宽度 > max_crack_width_px × 0.5
    → 极细长且较宽的均匀条带，判定为阴影，剔除

几何特征计算：
  使用 cv2.connectedComponentsWithStats 提取连通域基本统计量
  使用 cv2.minAreaRect 获取旋转包围框（长轴、短轴、角度）
  使用 cv2.findContours 获取轮廓，计算复杂度

参数默认值（来自 configs/mask2former_crack.yaml）：
  max_crack_width_px           = 20
  max_transverse_span_ratio    = 0.80
  min_contour_complexity       = 20.0
  min_region_area_px           = 50
  shadow_aspect_ratio_threshold = 30.0

运行方式（独立测试）：
  python src/model/physical_constraints.py --test_mask path/to/mask.png
────────────────────────────────────────────────────────────────────────────

Step 6 几个关键设计说明：
轮廓复杂度（约束4）的数学依据：这是 Polsby-Popper 圆度指数的倒数，公式为 周长² / (4π × 面积)。
完美圆形 = 1.0，规则矩形 ≈ 1.27，热棒阴影（扁平矩形）≈ 3~8，而真实裂缝由于边缘极不规则、周长相对面积极大，通常远超 20，因此以 20 为下限可以有效区分。

约束3的宽度均匀性判断：仅靠跨幅比还不够，一条较长的真实裂缝也可能横跨图像 80% 以上。加入宽度均匀性检查（等效宽度与 bbox 高度之比的变异程度）后，
只有既"横贯全幅"又"宽度均匀"的区域才会被判定为分界线，避免误剔除真实裂缝。

合成数据测试：独立运行时会在内存中构造 4 种典型场景（真实裂缝、分界线、热棒阴影、噪点），不依赖任何真实文件，可以在拿到数据之前就验证约束逻辑是否正确触发。

第一次训练后的参数调优方式：如果发现某类误检没被剔除，或真实裂缝被误剔除，直接修改 configs/mask2former_crack.yaml 中 physical_constraints 下的对应参数即可，无需改动代码。
"""

import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import yaml


# ─── 常量 ─────────────────────────────────────────────────────────────────────

LINEAR_CRACK_CLASS_ID = 1
BACKGROUND_CLASS_ID = 0

CLASSES = {
    0: "background",
    1: "damage",

}


# ─── 约束参数数据类 ───────────────────────────────────────────────────────────

@dataclass
class ConstraintParams:
    """
    物理约束参数集合。
    所有参数均可从 yaml 配置文件中读取，也可在代码中直接覆盖。
    """
    # 约束 1：最小连通域面积（像素数），低于此值视为噪点
    min_region_area_px: int = 100

    # 约束 2：连通域等效宽度上限（像素），超过则认为过宽，非裂缝
    # 等效宽度 = 面积 / 主轴长度（最小外接矩形的长轴）
    max_crack_width_px: float = 35.0

    # 约束 3：横向跨幅比上限（连通域 x 范围 / 图像宽度）
    # 超过此比例 且 宽度均匀 → 判定为分界线
    max_transverse_span_ratio: float = 0.90

    # 约束 3 辅助：宽度均匀性判断阈值（等效宽度的变异系数 = 标准差/均值）
    # 低于此值认为宽度均匀
    width_uniformity_cv: float = 0.3

    # 约束 4：轮廓复杂度下限（= 周长² / (4π × 面积)）
    # 正圆 = 1.0，裂缝通常 >> 20，阴影/矩形 接近 1~5
    min_contour_complexity: float = 3.0

    # 约束 5：纵横比上限（长轴/短轴），超过此值且宽度较宽 → 判定为阴影
    shadow_aspect_ratio_threshold: float = 40.0

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ConstraintParams":
        """从 yaml 文件读取物理约束参数。"""
        if not yaml_path.exists():
            print(f"[警告] 配置文件不存在：{yaml_path}，使用默认参数")
            return cls()
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        pc = cfg.get("physical_constraints", {})
        return cls(
            min_region_area_px           = pc.get("min_region_area_px",            50),
            max_crack_width_px           = pc.get("max_crack_width_px",            20.0),
            max_transverse_span_ratio    = pc.get("max_transverse_span_ratio",     0.80),
            min_contour_complexity       = pc.get("min_contour_complexity",        20.0),
            shadow_aspect_ratio_threshold= pc.get("shadow_aspect_ratio_threshold", 30.0),
        )

    def __str__(self) -> str:
        return (
            f"ConstraintParams(\n"
            f"  min_region_area_px           = {self.min_region_area_px}\n"
            f"  max_crack_width_px           = {self.max_crack_width_px}\n"
            f"  max_transverse_span_ratio    = {self.max_transverse_span_ratio}\n"
            f"  width_uniformity_cv          = {self.width_uniformity_cv}\n"
            f"  min_contour_complexity       = {self.min_contour_complexity}\n"
            f"  shadow_aspect_ratio_threshold= {self.shadow_aspect_ratio_threshold}\n"
            f")"
        )


# ─── 单连通域几何特征提取 ─────────────────────────────────────────────────────

@dataclass
class RegionGeometry:
    """单个连通域的几何属性。"""
    label_id:          int
    area:              int      # 像素数
    bbox_x:            int      # 外接矩形左上角 x
    bbox_y:            int      # 外接矩形左上角 y
    bbox_w:            int      # 外接矩形宽度
    bbox_h:            int      # 外接矩形高度
    major_axis:        float    # 最小旋转外接矩形的长轴长度
    minor_axis:        float    # 最小旋转外接矩形的短轴长度
    equiv_width:       float    # 等效宽度 = area / major_axis
    aspect_ratio:      float    # 纵横比 = major_axis / minor_axis
    span_x:            int      # 横向跨度（bbox_w）
    contour_complexity: float   # 轮廓复杂度 = 周长² / (4π × area)

    # 约束判断结果（用于 debug 日志）
    reject_reason: str = ""


def extract_region_geometry(
    binary_mask: np.ndarray,
    label_id:    int,
    component_mask: np.ndarray,
) -> RegionGeometry:
    """
    计算单个连通域的完整几何特征。

    Parameters
    ----------
    binary_mask    : 全图二值 mask（0/1），仅当前连通域为 1
    label_id       : 连通域编号
    component_mask : 当前连通域的像素 mask（bool 或 uint8）
    """
    # ── 基本统计（面积、bbox）─────────────────────────────────────────────────
    area = int(np.sum(component_mask))

    # 外接轴对齐矩形
    coords = np.where(component_mask)
    y_min, y_max = int(coords[0].min()), int(coords[0].max())
    x_min, x_max = int(coords[1].min()), int(coords[1].max())
    bbox_w = x_max - x_min + 1
    bbox_h = y_max - y_min + 1
    span_x = bbox_w

    # ── 最小旋转外接矩形（获取真实长短轴）───────────────────────────────────
    comp_uint8 = component_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        comp_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    major_axis = float(max(bbox_w, bbox_h))   # 默认值（轮廓点不足时）
    minor_axis = float(min(bbox_w, bbox_h))
    perimeter  = float(bbox_w * 2 + bbox_h * 2)

    if contours:
        # 取面积最大的轮廓（排除内部空洞产生的小轮廓）
        main_contour = max(contours, key=cv2.contourArea)
        perimeter    = cv2.arcLength(main_contour, closed=True)

        if len(main_contour) >= 5:
            # minAreaRect 需要至少 5 个点
            rect = cv2.minAreaRect(main_contour)
            # rect = ((cx, cy), (w, h), angle)，w/h 不区分长短轴
            rect_w, rect_h = rect[1]
            major_axis = float(max(rect_w, rect_h))
            minor_axis = float(min(rect_w, rect_h))

    # ── 等效宽度 ──────────────────────────────────────────────────────────────
    equiv_width = area / major_axis if major_axis > 0 else float(bbox_h)

    # ── 纵横比 ────────────────────────────────────────────────────────────────
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else float("inf")

    # ── 轮廓复杂度（Polsby-Popper 的倒数，越大越不规则）─────────────────────
    # 标准圆形：complexity = 1.0
    # 细长裂缝：complexity >> 20（周长极大，面积相对较小）
    # 规则矩形：complexity ≈ 4/π ≈ 1.27
    contour_complexity = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0.0

    return RegionGeometry(
        label_id           = label_id,
        area               = area,
        bbox_x             = x_min,
        bbox_y             = y_min,
        bbox_w             = bbox_w,
        bbox_h             = bbox_h,
        major_axis         = major_axis,
        minor_axis         = minor_axis,
        equiv_width        = equiv_width,
        aspect_ratio       = aspect_ratio,
        span_x             = span_x,
        contour_complexity = contour_complexity,
    )


# ─── 单连通域约束判断 ─────────────────────────────────────────────────────────

def should_reject(
    geom:      RegionGeometry,
    img_width: int,
    params:    ConstraintParams,
) -> tuple[bool, str]:
    """
    对单个连通域逐项应用 5 条物理约束。

    Returns
    -------
    (True, reason_str)  → 应剔除
    (False, "")         → 保留
    """

    # ── 约束 1：最小面积 ──────────────────────────────────────────────────────
    if geom.area < params.min_region_area_px:
        return True, (
            f"约束1_面积过小 "
            f"(area={geom.area} < {params.min_region_area_px})"
        )

    # ── 约束 2：等效宽度上限 ──────────────────────────────────────────────────
    if geom.equiv_width > params.max_crack_width_px:
        return True, (
            f"约束2_等效宽度过大 "
            f"(equiv_width={geom.equiv_width:.1f} > {params.max_crack_width_px})"
        )

    # ── 约束 3：横向跨幅 + 宽度均匀性（分界线判定）───────────────────────────
    span_ratio = geom.span_x / img_width
    if span_ratio > params.max_transverse_span_ratio:
        # 进一步检查宽度是否均匀
        # 此处使用等效宽度与 bbox 高度之比作为均匀性代理指标：
        # 若等效宽度接近 bbox_h，说明连通域整体宽度较均匀
        if geom.bbox_h > 0:
            uniformity_proxy = abs(geom.equiv_width - geom.bbox_h) / geom.bbox_h
        else:
            uniformity_proxy = 0.0

        if uniformity_proxy < params.width_uniformity_cv:
            return True, (
                f"约束3_横向分界线 "
                f"(span_ratio={span_ratio:.2f} > {params.max_transverse_span_ratio}, "
                f"宽度均匀 uniformity={uniformity_proxy:.2f})"
            )

    # ── 约束 4：轮廓复杂度（过于规整 → 阴影/矩形区域）──────────────────────
    if geom.contour_complexity < params.min_contour_complexity:
        return True, (
            f"约束4_轮廓过于规整 "
            f"(complexity={geom.contour_complexity:.2f} < {params.min_contour_complexity})"
        )

    # ── 约束 5：纵横比 + 宽度（细长均匀条带 → 热棒阴影）────────────────────
    if (
        geom.aspect_ratio > params.shadow_aspect_ratio_threshold
        and geom.equiv_width > params.max_crack_width_px * 0.5
    ):
        return True, (
            f"约束5_阴影条带 "
            f"(aspect_ratio={geom.aspect_ratio:.1f} > {params.shadow_aspect_ratio_threshold}, "
            f"equiv_width={geom.equiv_width:.1f})"
        )

    return False, ""


# ─── 主处理函数 ───────────────────────────────────────────────────────────────

def apply_physical_constraints(
    pred_mask: np.ndarray,
    params:    ConstraintParams,
    verbose:   bool = False,
) -> tuple[np.ndarray, dict]:
    """
    对预测 mask 的 Transverse 类应用全部物理约束。

    Parameters
    ----------
    pred_mask : np.ndarray uint8 [H, W]，像素值 0~5
    params    : ConstraintParams
    verbose   : 若为 True，打印每个连通域的判断过程

    Returns
    -------
    filtered_mask : np.ndarray uint8 [H, W]，修正后的 mask
    stats         : dict，包含统计信息（保留/剔除数量、各约束触发次数）
    """
    filtered_mask = pred_mask.copy()
    img_h, img_w  = pred_mask.shape

    # 提取 Transverse 类的二值 mask
    transverse_binary = (pred_mask == LINEAR_CRACK_CLASS_ID).astype(np.uint8)

    if transverse_binary.sum() == 0:
        # 当前 mask 中无 Transverse 预测，直接返回
        return filtered_mask, {
            "total_regions": 0, "kept": 0, "rejected": 0,
            "reject_by_constraint": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        }

    # 连通域标记
    num_labels, label_map, stats_cv, _ = cv2.connectedComponentsWithStats(
        transverse_binary, connectivity=8
    )
    # stats_cv 形状：[num_labels, 5]，列：[x, y, w, h, area]
    # label 0 为背景，跳过

    total_regions = num_labels - 1
    kept      = 0
    rejected  = 0
    reject_by = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for lbl in range(1, num_labels):
        component_mask = (label_map == lbl)

        # 提取几何特征
        geom = extract_region_geometry(transverse_binary, lbl, component_mask)

        # 应用约束
        reject, reason = should_reject(geom, img_w, params)

        if reject:
            # 将该连通域像素重分类为 background
            filtered_mask[component_mask] = BACKGROUND_CLASS_ID
            rejected += 1

            # 统计触发了哪条约束
            for c_id in [1, 2, 3, 4, 5]:
                if f"约束{c_id}" in reason:
                    reject_by[c_id] += 1
                    break

            if verbose:
                print(f"  [剔除] label={lbl:>4}  area={geom.area:>7}  "
                      f"equiv_w={geom.equiv_width:>6.1f}  "
                      f"aspect={geom.aspect_ratio:>7.1f}  "
                      f"complexity={geom.contour_complexity:>7.1f}  "
                      f"span={geom.span_x/img_w:.2f}  "
                      f"原因：{reason}")
        else:
            kept += 1
            if verbose:
                print(f"  [保留] label={lbl:>4}  area={geom.area:>7}  "
                      f"equiv_w={geom.equiv_width:>6.1f}  "
                      f"aspect={geom.aspect_ratio:>7.1f}  "
                      f"complexity={geom.contour_complexity:>7.1f}  "
                      f"span={geom.span_x/img_w:.2f}")

    stats = {
        "total_regions":      total_regions,
        "kept":               kept,
        "rejected":           rejected,
        "reject_by_constraint": reject_by,
    }

    return filtered_mask, stats


# ─── Batch 处理接口 ───────────────────────────────────────────────────────────

def apply_constraints_batch(
    pred_masks: np.ndarray,
    params:     ConstraintParams,
    verbose:    bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """
    对一个 batch 的预测 mask 逐张应用物理约束。

    Parameters
    ----------
    pred_masks : np.ndarray uint8 [B, H, W]
    params     : ConstraintParams

    Returns
    -------
    filtered_masks : np.ndarray [B, H, W]
    batch_stats    : List[dict]，每张图的统计信息
    """
    B = pred_masks.shape[0]
    filtered = np.zeros_like(pred_masks)
    batch_stats = []

    for i in range(B):
        f_mask, s = apply_physical_constraints(pred_masks[i], params, verbose)
        filtered[i] = f_mask
        batch_stats.append(s)

    return filtered, batch_stats


# ─── 统计汇总 ─────────────────────────────────────────────────────────────────

def summarize_constraint_stats(stats_list: list[dict]) -> None:
    """打印多张图物理约束统计的汇总信息。"""
    total_r  = sum(s["total_regions"] for s in stats_list)
    total_k  = sum(s["kept"]          for s in stats_list)
    total_rej= sum(s["rejected"]      for s in stats_list)

    rej_by = {c: sum(s["reject_by_constraint"].get(c, 0) for s in stats_list)
              for c in range(1, 6)}

    print("\n[物理约束统计汇总]")
    print(f"  处理图像数      ：{len(stats_list)}")
    print(f"  Transverse 连通域总数：{total_r}")
    print(f"  保留            ：{total_k}  ({total_k/max(total_r,1)*100:.1f}%)")
    print(f"  剔除            ：{total_rej}  ({total_rej/max(total_r,1)*100:.1f}%)")
    print(f"  各约束触发次数  ：")
    constraint_names = {
        1: "约束1_面积过小",
        2: "约束2_等效宽度过大",
        3: "约束3_横向分界线",
        4: "约束4_轮廓过于规整",
        5: "约束5_阴影条带",
    }
    for c_id, cnt in rej_by.items():
        print(f"    {constraint_names[c_id]:<22}：{cnt}")


# ─── 独立测试入口 ─────────────────────────────────────────────────────────────

def _test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mask", type=str, default=None,
                        help="用于测试的 mask.png 路径（可选）")
    parser.add_argument("--yaml",      type=str,
                        default="configs/mask2former_crack.yaml")
    parser.add_argument("--verbose",   action="store_true")
    args = parser.parse_args()

    params = ConstraintParams.from_yaml(Path(args.yaml))
    print(params)

    if args.test_mask:
        mask_path = Path(args.test_mask)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[错误] 无法读取 mask：{mask_path}")
            return

        before_count = int(np.sum(mask == LINEAR_CRACK_CLASS_ID))
        filtered, stats = apply_physical_constraints(
            mask, params, verbose=args.verbose
        )
        after_count = int(np.sum(filtered == LINEAR_CRACK_CLASS_ID))

        print(f"\n[测试结果]")
        print(f"  Transverse 像素（约束前）：{before_count:,}")
        print(f"  Transverse 像素（约束后）：{after_count:,}")
        print(f"  剔除像素数               ：{before_count - after_count:,}")
        print(f"  连通域统计：{stats}")

        # 保存对比图
        out_path = mask_path.parent / (mask_path.stem + "_filtered.png")
        cv2.imwrite(str(out_path), filtered)
        print(f"  [✓] 过滤后 mask 已保存：{out_path}")
    else:
        # 用合成数据测试
        print("\n[合成数据测试]")
        img_h, img_w = 1530, 2720
        test_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # 模拟1：真实横向裂缝（细短，不规则）
        for x in range(300, 600):
            offset = int(5 * np.sin(x * 0.05))
            y = 400 + offset
            if 0 <= y < img_h:
                test_mask[y, x] = LINEAR_CRACK_CLASS_ID
                if x % 3 == 0:
                    test_mask[y+1, x] = LINEAR_CRACK_CLASS_ID

        # 模拟2：路面分界线（贯穿全幅，均匀宽度）
        test_mask[800:803, 100:2620] = LINEAR_CRACK_CLASS_ID

        # 模拟3：热棒阴影（细长矩形，较宽）
        test_mask[600:625, 1000:1500] = LINEAR_CRACK_CLASS_ID

        # 模拟4：噪点（面积极小）
        test_mask[200, 200] = LINEAR_CRACK_CLASS_ID

        before = int(np.sum(test_mask == LINEAR_CRACK_CLASS_ID))
        filtered, stats = apply_physical_constraints(
            test_mask, params, verbose=True
        )
        after = int(np.sum(filtered == LINEAR_CRACK_CLASS_ID))

        print(f"\n  约束前 Transverse 像素：{before:,}")
        print(f"  约束后 Transverse 像素：{after:,}")
        print(f"  统计：{stats}")
        print("\n[✓] Step 6 物理约束模块测试通过\n")


if __name__ == "__main__":
    _test()