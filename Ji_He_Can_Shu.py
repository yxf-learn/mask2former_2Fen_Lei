from __future__ import annotations

import csv
import copy
import json
from pathlib import Path
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


LINEAR_FAMILY = "linear_crack"
PATCH_FAMILY = "patch"
NETWORK_FAMILY = "alligator"


@dataclass(slots=True)
class DiseaseClassSpec:
    class_id: int
    name: str
    family: str
    enabled: bool = True


@dataclass(slots=True)
class GeometryExtractionConfig:
    """Shared configuration for class-aware pavement distress analysis."""

    mm_per_pixel: float = 1.0
    min_area: int = 50
    binarize_threshold: float = 0.5
    crack_class_ids: tuple[int, ...] | None = None
    opening_kernel_size: int = 3
    closing_kernel_size: int = 3
    prune_length: int = 10
    enable_hole_filling: bool = True
    class_specs: tuple[DiseaseClassSpec, ...] = field(
        default_factory=lambda: (
            DiseaseClassSpec(0, "background", "background", enabled=False),
            DiseaseClassSpec(1, "Transverse", LINEAR_FAMILY),
            DiseaseClassSpec(2, "Longitudinal", LINEAR_FAMILY),
            DiseaseClassSpec(3, "fixpatch", PATCH_FAMILY),
            DiseaseClassSpec(4, "Alligator", NETWORK_FAMILY),
        )
    )
    metadata: dict[str, Any] = field(default_factory=dict)


class CrackGeometryExtractor:
    """
    Class-aware stage scaffold for pavement distress geometry extraction.

    Current responsibilities:
    - validate the mask
    - keep multiclass information
    - route each class to the correct analysis family
    - return a stable result schema for future stages
    """

    def __init__(
        self,
        calibration: float | None = None,
        min_area: int = 50,
        config: GeometryExtractionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None and calibration is not None:
            raise ValueError("Use either 'config' or 'calibration', not both.")

        if config is None:
            mm_per_pixel = 1.0 if calibration is None else float(calibration)
            config = GeometryExtractionConfig(
                mm_per_pixel=mm_per_pixel,
                min_area=min_area,
                **kwargs,
            )

        if config.mm_per_pixel <= 0:
            raise ValueError("mm_per_pixel must be positive.")
        if config.min_area < 0:
            raise ValueError("min_area must be non-negative.")

        self.config = config
        self.d = config.mm_per_pixel
        self.min_area = config.min_area
        self.class_specs = {spec.class_id: spec for spec in config.class_specs}

    def extract(self, mask: np.ndarray, analysis_profile: str = "full") -> dict[str, Any]:
        """
        Main entry point.

        For multiclass masks, each disease category is preserved and routed to
        its own analysis family. Later stages will progressively replace the
        family stubs with real computations.
        """
        if analysis_profile not in {"full", "fast", "basic"}:
            raise ValueError("analysis_profile must be 'full', 'fast', or 'basic'.")

        start_time = time.perf_counter()

        normalized_mask, mask_mode = self._normalize_mask(mask)
        image_shape = tuple(int(v) for v in normalized_mask.shape)

        class_results = []
        preprocessed_masks: dict[str, np.ndarray] = {}
        component_labels: dict[str, np.ndarray] = {}
        skeletons: dict[str, Any] = {}
        distance_transforms: dict[str, np.ndarray | None] = {}
        graphs: dict[str, Any] = {}
        enabled_specs = [spec for spec in self.class_specs.values() if spec.enabled]

        for spec in enabled_specs:
            class_mask = (normalized_mask == spec.class_id).astype(np.uint8)
            preprocessed_mask = self._preprocess_mask(class_mask, spec)
            preprocessed_masks[spec.name] = preprocessed_mask
            label_map, components = self._extract_components(preprocessed_mask, spec)
            component_labels[spec.name] = label_map
            skeletons[spec.name] = None
            if analysis_profile != "basic":
                skeletons[spec.name] = self._extract_skeletons(label_map, components, spec)
            raw_pixel_count = int(class_mask.sum())
            pixel_count = int(preprocessed_mask.sum())
            class_result = self._build_class_stub(
                spec=spec,
                mask_shape=image_shape,
                pixel_count=pixel_count,
                raw_pixel_count=raw_pixel_count,
                preprocessed_mask=preprocessed_mask,
                components=components,
            )
            class_result["skeleton"] = skeletons[spec.name]
            distance_transform = self._compute_distance_transform(preprocessed_mask)
            distance_transforms[spec.name] = distance_transform
            self._populate_basic_geometry(
                class_result=class_result,
                class_mask=preprocessed_mask,
                components=components,
                label_map=label_map,
                distance_transform=distance_transform,
                image_shape=image_shape,
            )
            graph_summary = None
            if analysis_profile == "full":
                graph_summary = self._build_topology_graphs(
                    class_result=class_result,
                    label_map=label_map,
                    components=components,
                    spec=spec,
                )
            graphs[spec.name] = graph_summary
            class_result["graph"] = graph_summary
            self._populate_complexity_and_engineering(
                class_result=class_result,
                class_mask=preprocessed_mask,
                components=components,
                label_map=label_map,
                image_shape=image_shape,
            )
            class_results.append(class_result)

        result = {
            "stage": "stage_7_output_packaging_and_visualization",
            "status": "ready_for_integration_and_real_data_validation",
            "config": self._serialize_config(),
            "image": {
                "shape": image_shape,
                "dtype": str(mask.dtype),
                "mask_mode": mask_mode,
                "class_ids_present": self._unique_ids(normalized_mask),
            },
            "intermediate": {
                "normalized_mask": normalized_mask,
                "preprocessed_masks": preprocessed_masks,
                "component_labels": component_labels,
                "skeletons": skeletons,
                "distance_transforms": distance_transforms,
                "graphs": graphs,
            },
            "classes": class_results,
            "summary": self._build_summary(class_results),
        }
        result["exports"] = self._build_export_package(
            result=result,
            normalized_mask=normalized_mask,
        )
        result["validation"] = self._build_validation_summary(
            result=result,
            runtime_sec=(time.perf_counter() - start_time),
            analysis_profile=analysis_profile,
        )
        return result

    def _normalize_mask(self, mask: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Normalize different mask formats.

        Returns:
        - normalized 2D mask
        - mask mode: 'multiclass', 'binary_from_integer', or 'binary_from_float'
        """
        array = np.asarray(mask)

        if array.ndim == 3:
            if array.shape[-1] == 1:
                array = array[..., 0]
            else:
                raise ValueError(
                    "Expected a 2D mask or a single-channel 3D mask, "
                    f"but received shape {array.shape}."
                )

        if array.ndim != 2:
            raise ValueError(f"Mask must be 2D, but received shape {array.shape}.")

        if array.dtype == np.bool_:
            return array.astype(np.uint8), "binary_from_integer"

        if np.issubdtype(array.dtype, np.floating):
            threshold = float(self.config.binarize_threshold)
            binary = (array >= threshold).astype(np.uint8)
            return binary, "binary_from_float"

        if np.issubdtype(array.dtype, np.integer):
            unique_ids = set(np.unique(array).tolist())
            known_ids = set(self.class_specs.keys())
            if unique_ids.issubset(known_ids):
                return array.astype(np.int32), "multiclass"

            class_ids = self.config.crack_class_ids
            if class_ids:
                binary_mask = np.isin(array, np.array(class_ids, dtype=array.dtype))
            else:
                binary_mask = array > 0
            return binary_mask.astype(np.uint8), "binary_from_integer"

        raise TypeError(f"Unsupported mask dtype: {array.dtype}")

    def _serialize_config(self) -> dict[str, Any]:
        config_dict = asdict(self.config)
        config_dict["class_specs"] = [asdict(spec) for spec in self.config.class_specs]
        return config_dict

    def _unique_ids(self, mask: np.ndarray) -> list[int]:
        return [int(v) for v in np.unique(mask).tolist()]

    def _build_class_stub(
        self,
        spec: DiseaseClassSpec,
        mask_shape: tuple[int, int],
        pixel_count: int,
        raw_pixel_count: int,
        preprocessed_mask: np.ndarray,
        components: list[dict[str, Any]],
    ) -> dict[str, Any]:
        area_mm2 = pixel_count * (self.d ** 2)
        return {
            "class_id": spec.class_id,
            "class_name": spec.name,
            "family": spec.family,
            "mask_shape": mask_shape,
            "raw_pixel_count": raw_pixel_count,
            "pixel_count": pixel_count,
            "area_mm2": float(area_mm2),
            "component_count": len(components),
            "analysis_route": self._analysis_route_for_family(spec.family),
            "preprocessing": self._build_preprocess_summary(
                spec=spec,
                raw_pixel_count=raw_pixel_count,
                cleaned_pixel_count=pixel_count,
                preprocessed_mask=preprocessed_mask,
            ),
            "common_geometry": self._build_common_geometry_stub(pixel_count),
            "family_geometry": self._build_family_stub(spec.family),
            "complexity": self._build_complexity_stub(),
            "engineering": self._build_engineering_stub(),
            "skeleton": None,
            "graph": None,
            "components": components,
        }

    def _analysis_route_for_family(self, family: str) -> str:
        if family == LINEAR_FAMILY:
            return "component -> skeleton -> width/length/topology"
        if family == PATCH_FAMILY:
            return "component -> contour -> region geometry"
        if family == NETWORK_FAMILY:
            return "component -> region + skeleton network"
        return "unsupported"

    def _build_common_geometry_stub(self, pixel_count: int) -> dict[str, Any]:
        return {
            "area_px": pixel_count,
            "area_mm2": float(pixel_count * (self.d ** 2)),
            "perimeter_px": None,
            "perimeter_mm": None,
            "bbox": None,
            "centroid": None,
            "aspect_ratio": None,
            "compactness": None,
            "solidity": None,
            "dominant_angle_deg": None,
        }

    def _build_family_stub(self, family: str) -> dict[str, Any]:
        if family == LINEAR_FAMILY:
            return {
                "length_px": None,
                "length_mm": None,
                "width_max_px": None,
                "width_max_mm": None,
                "width_mean_px": None,
                "width_mean_mm": None,
                "width_median_px": None,
                "width_median_mm": None,
                "width_std_px": None,
                "width_std_mm": None,
                "n_endpoints": None,
                "n_junctions": None,
                "n_branches": None,
                "tortuosity": None,
                "direction_entropy": None,
            }

        if family == PATCH_FAMILY:
            return {
                "major_axis_px": None,
                "major_axis_mm": None,
                "minor_axis_px": None,
                "minor_axis_mm": None,
                "rectangularity": None,
                "elongation": None,
                "boundary_roughness": None,
                "patch_bandwidth_px": None,
                "patch_bandwidth_mm": None,
            }

        if family == NETWORK_FAMILY:
            return {
                "skeleton_length_px": None,
                "skeleton_length_mm": None,
                "crack_density": None,
                "junction_density": None,
                "n_endpoints": None,
                "n_junctions": None,
                "n_branches": None,
                "n_loops": None,
                "width_max_px": None,
                "width_max_mm": None,
                "width_mean_px": None,
                "width_mean_mm": None,
                "direction_entropy": None,
                "anisotropy_index": None,
                "fractal_dimension": None,
                "cell_area_mean_px": None,
            }

        return {"note": "No family-specific metrics defined."}

    def _build_summary(self, class_results: list[dict[str, Any]]) -> dict[str, Any]:
        active_classes = [item for item in class_results if item["pixel_count"] > 0]
        total_pixels = sum(item["pixel_count"] for item in class_results)
        total_raw_pixels = sum(item["raw_pixel_count"] for item in class_results)
        total_components = sum(item["component_count"] for item in class_results)

        return {
            "n_active_classes": len(active_classes),
            "active_class_names": [item["class_name"] for item in active_classes],
            "total_components": int(total_components),
            "total_raw_foreground_pixels": int(total_raw_pixels),
            "total_foreground_pixels": int(total_pixels),
            "total_foreground_area_mm2": float(total_pixels * (self.d ** 2)),
            "notes": [
                "Linear cracks will use skeleton-based geometry.",
                "Fixpatch regions will use contour and region geometry.",
                "Alligator cracks will use regional and network metrics.",
            ],
        }

    def _build_complexity_stub(self) -> dict[str, Any]:
        return {
            "fractal_dimension": None,
            "fill_ratio": None,
            "branch_density": None,
            "node_density": None,
            "complexity_index": None,
        }

    def _build_engineering_stub(self) -> dict[str, Any]:
        return {
            "hazard_level": None,
            "reference_standard": None,
            "hazard_basis": None,
            "recommended_action": None,
        }

    def _build_export_package(
        self,
        result: dict[str, Any],
        normalized_mask: np.ndarray,
    ) -> dict[str, Any]:
        serializable = self.to_serializable_result(result, include_intermediate=False, include_exports=False)
        return {
            "json_ready": serializable,
            "class_rows": self.to_class_rows(result),
            "component_rows": self.to_component_rows(result),
            "visualization_keys": [
                "class_overlay",
                "skeleton_overlay",
                "node_overlay",
            ],
            "mask_snapshot": {
                "shape": [int(v) for v in normalized_mask.shape],
                "class_ids_present": [int(v) for v in np.unique(normalized_mask).tolist()],
            },
        }

    def _build_validation_summary(
        self,
        result: dict[str, Any],
        runtime_sec: float,
        analysis_profile: str,
    ) -> dict[str, Any]:
        image_shape = result["image"]["shape"]
        image_area = max(1, int(image_shape[0] * image_shape[1]))
        warnings: list[str] = []

        for class_item in result.get("classes", []):
            class_name = class_item["class_name"]
            coverage = float(class_item.get("pixel_count", 0) / image_area)
            if coverage > 0.8:
                warnings.append(f"{class_name} covers more than 80% of the image.")
            if class_name == "fixpatch" and coverage > 0.5:
                warnings.append("fixpatch dominates the image; verify the predicted class map.")

            family_geometry = class_item.get("family_geometry", {})
            width_max_mm = family_geometry.get("width_max_mm")
            if width_max_mm is not None and width_max_mm > 1000:
                warnings.append(f"{class_name} width_max_mm is unusually large ({width_max_mm:.2f}).")

        if runtime_sec > 30:
            warnings.append("Processing time exceeds 30 seconds; consider downsampling or ROI analysis.")

        return {
            "analysis_profile": analysis_profile,
            "runtime_sec": float(runtime_sec),
            "image_area_px": int(image_area),
            "warning_count": len(warnings),
            "warnings": warnings,
        }

    def extract_from_file(
        self,
        mask_path: str | Path,
        max_dim: int | None = None,
        analysis_profile: str = "fast",
    ) -> dict[str, Any]:
        mask_path = Path(mask_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask file: {mask_path}")

        effective_extractor = self
        processed_mask = mask
        scale = 1.0

        if max_dim is not None:
            processed_mask, scale = self._resize_mask_if_needed(mask, int(max_dim))
            if scale != 1.0:
                effective_extractor = self._clone_with_mm_per_pixel(self.d / scale)

        result = effective_extractor.extract(processed_mask, analysis_profile=analysis_profile)
        result["source"] = {
            "mask_path": str(mask_path),
            "original_shape": [int(v) for v in mask.shape[:2]],
            "processed_shape": [int(v) for v in processed_mask.shape[:2]],
            "resize_scale": float(scale),
        }
        return result

    def save_result_bundle(
        self,
        result: dict[str, Any],
        output_dir: str | Path,
        base_image: np.ndarray | None = None,
        save_visualizations: bool = True,
    ) -> dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        json_path = output_path / "geometry_result.json"
        class_csv_path = output_path / "class_rows.csv"
        component_csv_path = output_path / "component_rows.csv"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                self.to_serializable_result(result, include_intermediate=False, include_exports=True),
                f,
                ensure_ascii=False,
                indent=2,
            )

        self._write_rows_to_csv(class_csv_path, result["exports"]["class_rows"])
        self._write_rows_to_csv(component_csv_path, result["exports"]["component_rows"])

        saved_paths = {
            "json": str(json_path),
            "class_csv": str(class_csv_path),
            "component_csv": str(component_csv_path),
        }

        if save_visualizations:
            visuals = self.render_visualizations(result, base_image=base_image)
            for key, image in visuals.items():
                image_path = output_path / f"{key}.png"
                cv2.imwrite(str(image_path), image)
                saved_paths[key] = str(image_path)

        return saved_paths

    def _write_rows_to_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            with path.open("w", encoding="utf-8", newline="") as f:
                f.write("")
            return

        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _resize_mask_if_needed(
        self,
        mask: np.ndarray,
        max_dim: int,
    ) -> tuple[np.ndarray, float]:
        h, w = mask.shape[:2]
        current_max = max(h, w)
        if max_dim <= 0 or current_max <= max_dim:
            return mask, 1.0

        scale = float(max_dim / current_max)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return resized, scale

    def _clone_with_mm_per_pixel(self, mm_per_pixel: float) -> CrackGeometryExtractor:
        new_config = copy.deepcopy(self.config)
        new_config.mm_per_pixel = float(mm_per_pixel)
        return CrackGeometryExtractor(config=new_config)

    def _preprocess_mask(self, mask: np.ndarray, spec: DiseaseClassSpec) -> np.ndarray:
        binary = np.asarray(mask, dtype=np.uint8)
        binary = (binary > 0).astype(np.uint8)
        if not binary.any():
            return binary

        opening_size, closing_size, fill_holes = self._preprocess_policy(spec.family)

        if opening_size > 1:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                self._square_structure(opening_size),
            )

        if closing_size > 1:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                self._square_structure(closing_size),
            )

        if fill_holes and self.config.enable_hole_filling:
            binary = self._fill_holes(binary)

        binary = self._remove_small_objects(binary, min_size=self.min_area)
        return binary

    def _preprocess_policy(self, family: str) -> tuple[int, int, bool]:
        base_open = self._normalize_kernel_size(self.config.opening_kernel_size)
        base_close = self._normalize_kernel_size(self.config.closing_kernel_size)

        if family == LINEAR_FAMILY:
            return 1, max(3, base_close), False
        if family == PATCH_FAMILY:
            return max(3, base_open), max(3, base_close), True
        if family == NETWORK_FAMILY:
            return max(1, base_open), max(3, base_close), True
        return base_open, base_close, self.config.enable_hole_filling

    def _normalize_kernel_size(self, kernel_size: int) -> int:
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def _square_structure(self, kernel_size: int) -> np.ndarray:
        kernel_size = self._normalize_kernel_size(kernel_size)
        return np.ones((kernel_size, kernel_size), dtype=np.uint8)

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        if not binary.any():
            return binary

        flood = (binary * 255).copy()
        h, w = flood.shape
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, seedPoint=(0, 0), newVal=255)
        flood_inv = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(binary * 255, flood_inv)
        return (filled > 0).astype(np.uint8)

    def _remove_small_objects(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        min_size = max(0, int(min_size))
        if min_size <= 1 or not mask.any():
            return mask

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        if num_labels <= 1:
            return mask

        cleaned = np.zeros_like(mask, dtype=np.uint8)
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area >= min_size:
                cleaned[labels == label_idx] = 1
        return cleaned

    def _build_preprocess_summary(
        self,
        spec: DiseaseClassSpec,
        raw_pixel_count: int,
        cleaned_pixel_count: int,
        preprocessed_mask: np.ndarray,
    ) -> dict[str, Any]:
        opening_size, closing_size, fill_holes = self._preprocess_policy(spec.family)
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            preprocessed_mask.astype(np.uint8),
            connectivity=8,
        )
        num_components = max(0, int(num_labels) - 1)

        return {
            "opening_kernel_size": opening_size,
            "closing_kernel_size": closing_size,
            "hole_filling": bool(fill_holes and self.config.enable_hole_filling),
            "min_area": int(self.min_area),
            "raw_pixel_count": int(raw_pixel_count),
            "cleaned_pixel_count": int(cleaned_pixel_count),
            "removed_pixel_count": int(raw_pixel_count - cleaned_pixel_count),
            "pixel_retention_ratio": (
                float(cleaned_pixel_count / raw_pixel_count) if raw_pixel_count > 0 else None
            ),
            "n_connected_regions_after_cleaning": int(num_components),
        }

    def _extract_components(
        self,
        mask: np.ndarray,
        spec: DiseaseClassSpec,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if not binary.any():
            return np.zeros_like(binary, dtype=np.int32), []

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        components: list[dict[str, Any]] = []
        relabeled = np.zeros_like(labels, dtype=np.int32)
        component_idx = 0

        for label_idx in range(1, num_labels):
            area_px = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area_px < self.min_area:
                continue

            component_idx += 1
            relabeled[labels == label_idx] = component_idx
            components.append(
                self._build_component_stub(
                    component_id=component_idx,
                    spec=spec,
                    stats_row=stats[label_idx],
                    centroid=centroids[label_idx],
                )
            )

        return relabeled, components

    def _build_component_stub(
        self,
        component_id: int,
        spec: DiseaseClassSpec,
        stats_row: np.ndarray,
        centroid: np.ndarray,
    ) -> dict[str, Any]:
        x = int(stats_row[cv2.CC_STAT_LEFT])
        y = int(stats_row[cv2.CC_STAT_TOP])
        w = int(stats_row[cv2.CC_STAT_WIDTH])
        h = int(stats_row[cv2.CC_STAT_HEIGHT])
        area_px = int(stats_row[cv2.CC_STAT_AREA])

        return {
            "component_id": component_id,
            "class_id": spec.class_id,
            "class_name": spec.name,
            "family": spec.family,
            "pixel_count": area_px,
            "area_mm2": float(area_px * (self.d ** 2)),
            "bbox": {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "x2": x + w - 1,
                "y2": y + h - 1,
            },
            "centroid": {
                "x": float(centroid[0]),
                "y": float(centroid[1]),
            },
            "analysis_route": self._analysis_route_for_family(spec.family),
            "common_geometry": self._build_common_geometry_stub(area_px),
            "family_geometry": self._build_family_stub(spec.family),
            "complexity": self._build_complexity_stub(),
            "engineering": self._build_engineering_stub(),
            "skeleton": None,
            "graph": None,
        }

    def _compute_distance_transform(self, mask: np.ndarray) -> np.ndarray | None:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if not binary.any():
            return None
        padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
        return distance[1:-1, 1:-1]

    def _populate_basic_geometry(
        self,
        class_result: dict[str, Any],
        class_mask: np.ndarray,
        components: list[dict[str, Any]],
        label_map: np.ndarray,
        distance_transform: np.ndarray | None,
        image_shape: tuple[int, int],
    ) -> None:
        class_result["common_geometry"].update(self._compute_common_geometry(class_mask))
        class_result["family_geometry"].update(
            self._compute_family_geometry(
                family=class_result["family"],
                mask=class_mask,
                skeleton_info=class_result.get("skeleton"),
                distance_transform=distance_transform,
                image_shape=image_shape,
            )
        )

        for component in components:
            component_id = int(component["component_id"])
            component_mask = (label_map == component_id).astype(np.uint8)
            component["common_geometry"].update(self._compute_common_geometry(component_mask))
            component_distance = None
            if distance_transform is not None:
                component_distance = distance_transform * component_mask.astype(distance_transform.dtype)
            component["family_geometry"].update(
                self._compute_family_geometry(
                    family=component["family"],
                    mask=component_mask,
                    skeleton_info=component.get("skeleton"),
                    distance_transform=component_distance,
                    image_shape=image_shape,
                )
            )

    def _compute_common_geometry(self, mask: np.ndarray) -> dict[str, Any]:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        pixel_count = int(binary.sum())
        if pixel_count == 0:
            return {}

        ys, xs = np.where(binary > 0)
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter_px = float(sum(cv2.arcLength(contour, True) for contour in contours))

        points = np.column_stack((xs, ys)).astype(np.float32)
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())

        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        dominant_angle_deg = self._compute_pca_angle(points)
        compactness = None
        if perimeter_px > 0:
            compactness = float((4.0 * np.pi * pixel_count) / (perimeter_px ** 2))

        solidity = self._compute_solidity(points, pixel_count)

        return {
            "area_px": pixel_count,
            "area_mm2": float(pixel_count * (self.d ** 2)),
            "perimeter_px": perimeter_px,
            "perimeter_mm": float(perimeter_px * self.d),
            "bbox": {
                "x": x_min,
                "y": y_min,
                "w": width,
                "h": height,
                "x2": x_max,
                "y2": y_max,
            },
            "centroid": {
                "x": centroid_x,
                "y": centroid_y,
            },
            "aspect_ratio": aspect_ratio,
            "compactness": compactness,
            "solidity": solidity,
            "dominant_angle_deg": dominant_angle_deg,
        }

    def _compute_family_geometry(
        self,
        family: str,
        mask: np.ndarray,
        skeleton_info: dict[str, Any] | None,
        distance_transform: np.ndarray | None,
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        if family == LINEAR_FAMILY:
            return self._compute_linear_geometry(mask, skeleton_info, distance_transform)
        if family == PATCH_FAMILY:
            return self._compute_fixpatch_geometry(mask, distance_transform)
        if family == NETWORK_FAMILY:
            return self._compute_network_geometry(mask, skeleton_info, distance_transform, image_shape)
        return {}

    def _compute_linear_geometry(
        self,
        mask: np.ndarray,
        skeleton_info: dict[str, Any] | None,
        distance_transform: np.ndarray | None,
    ) -> dict[str, Any]:
        skeleton_mask = self._extract_skeleton_mask(skeleton_info)
        length_px = self._estimate_skeleton_length(skeleton_mask)
        width_stats = self._compute_width_statistics(skeleton_mask, distance_transform)

        return {
            "length_px": length_px,
            "length_mm": float(length_px * self.d) if length_px is not None else None,
            "width_max_px": width_stats["max_px"],
            "width_max_mm": self._to_mm(width_stats["max_px"]),
            "width_mean_px": width_stats["mean_px"],
            "width_mean_mm": self._to_mm(width_stats["mean_px"]),
            "width_median_px": width_stats["median_px"],
            "width_median_mm": self._to_mm(width_stats["median_px"]),
            "width_std_px": width_stats["std_px"],
            "width_std_mm": self._to_mm(width_stats["std_px"]),
        }

    def _compute_fixpatch_geometry(
        self,
        mask: np.ndarray,
        distance_transform: np.ndarray | None,
    ) -> dict[str, Any]:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if not binary.any():
            return {}

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {}

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        (major_axis, minor_axis) = sorted(rect[1], reverse=True)
        rect_area = float(max(major_axis * minor_axis, 1e-6))
        perimeter_px = float(cv2.arcLength(contour, True))
        hull = cv2.convexHull(contour)
        hull_perimeter = float(cv2.arcLength(hull, True))

        patch_bandwidth_px = float(minor_axis) if minor_axis > 0 else None
        if distance_transform is not None and binary.any():
            patch_bandwidth_px = float(max(patch_bandwidth_px or 0.0, 2.0 * float(distance_transform.max())))

        return {
            "major_axis_px": float(major_axis) if major_axis > 0 else None,
            "major_axis_mm": self._to_mm(major_axis),
            "minor_axis_px": float(minor_axis) if minor_axis > 0 else None,
            "minor_axis_mm": self._to_mm(minor_axis),
            "rectangularity": float(binary.sum() / rect_area) if rect_area > 0 else None,
            "elongation": float(major_axis / minor_axis) if minor_axis > 0 else None,
            "boundary_roughness": (
                float(perimeter_px / hull_perimeter) if hull_perimeter > 0 else None
            ),
            "patch_bandwidth_px": patch_bandwidth_px,
            "patch_bandwidth_mm": self._to_mm(patch_bandwidth_px),
        }

    def _compute_network_geometry(
        self,
        mask: np.ndarray,
        skeleton_info: dict[str, Any] | None,
        distance_transform: np.ndarray | None,
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        skeleton_mask = self._extract_skeleton_mask(skeleton_info)
        skeleton_length_px = self._estimate_skeleton_length(skeleton_mask)
        image_area = max(1, int(image_shape[0] * image_shape[1]))
        width_stats = self._compute_width_statistics(skeleton_mask, distance_transform)

        return {
            "skeleton_length_px": skeleton_length_px,
            "skeleton_length_mm": self._to_mm(skeleton_length_px),
            "crack_density": (
                float(skeleton_length_px / image_area) if skeleton_length_px is not None else None
            ),
            "junction_density": (
                float((skeleton_info or {}).get("total_junctions", 0) / image_area)
                if skeleton_info is not None
                else None
            ),
            "width_max_px": width_stats["max_px"],
            "width_max_mm": self._to_mm(width_stats["max_px"]),
            "width_mean_px": width_stats["mean_px"],
            "width_mean_mm": self._to_mm(width_stats["mean_px"]),
        }

    def _extract_skeleton_mask(self, skeleton_info: dict[str, Any] | None) -> np.ndarray | None:
        if not skeleton_info:
            return None
        skeleton_mask = skeleton_info.get("skeleton_mask")
        if skeleton_mask is None:
            return None
        return (np.asarray(skeleton_mask) > 0).astype(np.uint8)

    def _estimate_skeleton_length(self, skeleton_mask: np.ndarray | None) -> float | None:
        if skeleton_mask is None:
            return None

        skel = (np.asarray(skeleton_mask) > 0)
        if not skel.any():
            return 0.0

        horizontal = np.logical_and(skel[:, :-1], skel[:, 1:]).sum()
        vertical = np.logical_and(skel[:-1, :], skel[1:, :]).sum()
        diag_down = np.logical_and(skel[:-1, :-1], skel[1:, 1:]).sum()
        diag_up = np.logical_and(skel[:-1, 1:], skel[1:, :-1]).sum()

        length = float(horizontal + vertical + np.sqrt(2.0) * (diag_down + diag_up))
        return length

    def _compute_width_statistics(
        self,
        skeleton_mask: np.ndarray | None,
        distance_transform: np.ndarray | None,
    ) -> dict[str, float | None]:
        if skeleton_mask is None or distance_transform is None:
            return {"max_px": None, "mean_px": None, "median_px": None, "std_px": None}

        widths = 2.0 * distance_transform[skeleton_mask > 0]
        if widths.size == 0:
            return {"max_px": None, "mean_px": None, "median_px": None, "std_px": None}

        return {
            "max_px": float(np.max(widths)),
            "mean_px": float(np.mean(widths)),
            "median_px": float(np.median(widths)),
            "std_px": float(np.std(widths)),
        }

    def _compute_pca_angle(self, points: np.ndarray) -> float | None:
        if points.shape[0] < 2:
            return None

        centered = points - points.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
        angle = float(np.degrees(np.arctan2(principal_vector[1], principal_vector[0])))
        return angle

    def _compute_solidity(self, points: np.ndarray, pixel_count: int) -> float | None:
        if points.shape[0] < 3:
            return None

        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            return None
        return float(pixel_count / hull_area)

    def _to_mm(self, value_px: float | None) -> float | None:
        if value_px is None:
            return None
        return float(value_px * self.d)

    def _populate_complexity_and_engineering(
        self,
        class_result: dict[str, Any],
        class_mask: np.ndarray,
        components: list[dict[str, Any]],
        label_map: np.ndarray,
        image_shape: tuple[int, int],
    ) -> None:
        class_result["complexity"].update(
            self._compute_complexity_metrics(
                family=class_result["family"],
                mask=class_mask,
                skeleton_info=class_result.get("skeleton"),
                graph=class_result.get("graph"),
                image_shape=image_shape,
            )
        )
        class_result["engineering"].update(
            self._compute_engineering_metrics(
                class_name=class_result["class_name"],
                family=class_result["family"],
                common_geometry=class_result["common_geometry"],
                family_geometry=class_result["family_geometry"],
            )
        )

        for component in components:
            component_id = int(component["component_id"])
            component_mask = (label_map == component_id).astype(np.uint8)
            component["complexity"].update(
                self._compute_complexity_metrics(
                    family=component["family"],
                    mask=component_mask,
                    skeleton_info=component.get("skeleton"),
                    graph=component.get("graph"),
                    image_shape=image_shape,
                )
            )
            component["engineering"].update(
                self._compute_engineering_metrics(
                    class_name=component["class_name"],
                    family=component["family"],
                    common_geometry=component["common_geometry"],
                    family_geometry=component["family_geometry"],
                )
            )

    def _compute_complexity_metrics(
        self,
        family: str,
        mask: np.ndarray,
        skeleton_info: dict[str, Any] | None,
        graph: dict[str, Any] | None,
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        area_px = int(binary.sum())
        if area_px == 0:
            return {}

        bbox = self._compute_bbox(binary)
        bbox_area = max(1, int(bbox["w"] * bbox["h"]))
        fill_ratio = float(area_px / bbox_area)

        fractal_target = binary
        if family in {LINEAR_FAMILY, NETWORK_FAMILY}:
            skeleton_mask = self._extract_skeleton_mask(skeleton_info)
            if skeleton_mask is not None and np.any(skeleton_mask):
                fractal_target = skeleton_mask

        graph = graph or {}
        n_branches = int(graph.get("n_branches", 0) or 0)
        n_nodes = int(
            graph.get(
                "n_nodes",
                (graph.get("n_endpoints", 0) or 0) + (graph.get("n_junctions", 0) or 0),
            )
            or 0
        )
        branch_density = float(n_branches / area_px) if n_branches > 0 else 0.0
        node_density = float(n_nodes / area_px) if n_nodes > 0 else 0.0

        image_area = max(1, int(image_shape[0] * image_shape[1]))
        global_occupancy = float(area_px / image_area)
        complexity_index = float(fill_ratio + branch_density + node_density + global_occupancy)

        return {
            "fractal_dimension": self._box_count_fractal_dimension(fractal_target),
            "fill_ratio": fill_ratio,
            "branch_density": branch_density,
            "node_density": node_density,
            "complexity_index": complexity_index,
        }

    def _compute_engineering_metrics(
        self,
        class_name: str,
        family: str,
        common_geometry: dict[str, Any],
        family_geometry: dict[str, Any],
    ) -> dict[str, Any]:
        if class_name == "fixpatch":
            return {
                "hazard_level": "fixpatch_region",
                "reference_standard": "custom_fixpatch_region",
                "hazard_basis": "Region recorded as maintenance patch, not crack-width graded.",
                "recommended_action": "Track geometry and maintenance coverage.",
            }

        width_mm = None
        if family == LINEAR_FAMILY:
            width_mm = family_geometry.get("width_max_mm")
        elif family == NETWORK_FAMILY:
            width_mm = family_geometry.get("width_max_mm")

        if width_mm is None:
            return {
                "hazard_level": None,
                "reference_standard": "GB50010 crack width grading",
                "hazard_basis": "Width unavailable.",
                "recommended_action": None,
            }

        hazard_level, action = self._classify_hazard_from_width(width_mm)
        area_mm2 = common_geometry.get("area_mm2")
        return {
            "hazard_level": hazard_level,
            "reference_standard": "GB50010 crack width grading",
            "hazard_basis": f"width_max_mm={width_mm:.4f}, area_mm2={area_mm2:.4f}" if area_mm2 is not None else f"width_max_mm={width_mm:.4f}",
            "recommended_action": action,
        }

    def _classify_hazard_from_width(self, width_mm: float) -> tuple[str, str]:
        if width_mm < 0.2:
            return "slight", "Routine observation."
        if width_mm < 0.3:
            return "moderate", "Increase monitoring frequency."
        if width_mm < 0.5:
            return "severe", "Plan reinforcement or repair."
        return "critical", "Immediate treatment is recommended."

    def _compute_bbox(self, mask: np.ndarray) -> dict[str, int]:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return {"x": 0, "y": 0, "w": 0, "h": 0, "x2": 0, "y2": 0}
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
        return {
            "x": x_min,
            "y": y_min,
            "w": x_max - x_min + 1,
            "h": y_max - y_min + 1,
            "x2": x_max,
            "y2": y_max,
        }

    def _box_count_fractal_dimension(self, mask: np.ndarray) -> float | None:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if not np.any(binary):
            return None

        h, w = binary.shape
        min_dim = min(h, w)
        sizes: list[int] = []
        size = 1
        while size <= max(1, min_dim // 2):
            sizes.append(size)
            size *= 2

        if len(sizes) < 2:
            return None

        counts: list[int] = []
        valid_sizes: list[int] = []
        for box_size in sizes:
            count = self._count_nonempty_boxes(binary, box_size)
            if count > 0:
                counts.append(count)
                valid_sizes.append(box_size)

        if len(valid_sizes) < 2:
            return None

        x = np.log(1.0 / np.asarray(valid_sizes, dtype=np.float64))
        y = np.log(np.asarray(counts, dtype=np.float64))
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    def _count_nonempty_boxes(self, mask: np.ndarray, box_size: int) -> int:
        h, w = mask.shape
        count = 0
        for y in range(0, h, box_size):
            for x in range(0, w, box_size):
                if np.any(mask[y : y + box_size, x : x + box_size] > 0):
                    count += 1
        return count

    def to_serializable_result(
        self,
        result: dict[str, Any],
        include_intermediate: bool = False,
        include_exports: bool = False,
    ) -> dict[str, Any]:
        skip_keys = set()
        if not include_intermediate:
            skip_keys.add("intermediate")
        if not include_exports:
            skip_keys.add("exports")
        return self._convert_to_serializable(result, skip_keys=skip_keys)

    def _convert_to_serializable(self, value: Any, skip_keys: set[str] | None = None) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "__ndarray__": True,
                "shape": [int(v) for v in value.shape],
                "dtype": str(value.dtype),
                "nonzero": int(np.count_nonzero(value)),
            }
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, dict):
            return {
                key: self._convert_to_serializable(val, skip_keys=skip_keys)
                for key, val in value.items()
                if skip_keys is None or key not in skip_keys
            }
        if isinstance(value, (list, tuple)):
            return [self._convert_to_serializable(item, skip_keys=skip_keys) for item in value]
        return value

    def to_class_rows(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in result.get("classes", []):
            common = item.get("common_geometry", {})
            family = item.get("family_geometry", {})
            complexity = item.get("complexity", {})
            engineering = item.get("engineering", {})
            graph = item.get("graph", {}) or {}
            rows.append(
                {
                    "class_id": item.get("class_id"),
                    "class_name": item.get("class_name"),
                    "family": item.get("family"),
                    "pixel_count": item.get("pixel_count"),
                    "area_mm2": item.get("area_mm2"),
                    "component_count": item.get("component_count"),
                    "perimeter_mm": common.get("perimeter_mm"),
                    "aspect_ratio": common.get("aspect_ratio"),
                    "dominant_angle_deg": common.get("dominant_angle_deg"),
                    "length_mm": family.get("length_mm", family.get("skeleton_length_mm")),
                    "width_max_mm": family.get("width_max_mm", family.get("patch_bandwidth_mm")),
                    "width_mean_mm": family.get("width_mean_mm"),
                    "n_branches": family.get("n_branches"),
                    "n_endpoints": family.get("n_endpoints"),
                    "n_junctions": family.get("n_junctions"),
                    "n_loops": family.get("n_loops"),
                    "tortuosity": family.get("tortuosity"),
                    "direction_entropy": family.get("direction_entropy"),
                    "anisotropy_index": family.get("anisotropy_index", graph.get("anisotropy_index")),
                    "fractal_dimension": complexity.get("fractal_dimension"),
                    "complexity_index": complexity.get("complexity_index"),
                    "hazard_level": engineering.get("hazard_level"),
                    "reference_standard": engineering.get("reference_standard"),
                }
            )
        return rows

    def to_component_rows(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in result.get("classes", []):
            for component in item.get("components", []):
                common = component.get("common_geometry", {})
                family = component.get("family_geometry", {})
                complexity = component.get("complexity", {})
                engineering = component.get("engineering", {})
                bbox = component.get("bbox", {})
                rows.append(
                    {
                        "class_id": component.get("class_id"),
                        "class_name": component.get("class_name"),
                        "component_id": component.get("component_id"),
                        "pixel_count": component.get("pixel_count"),
                        "area_mm2": component.get("area_mm2"),
                        "bbox_x": bbox.get("x"),
                        "bbox_y": bbox.get("y"),
                        "bbox_w": bbox.get("w"),
                        "bbox_h": bbox.get("h"),
                        "perimeter_mm": common.get("perimeter_mm"),
                        "aspect_ratio": common.get("aspect_ratio"),
                        "dominant_angle_deg": common.get("dominant_angle_deg"),
                        "length_mm": family.get("length_mm", family.get("skeleton_length_mm")),
                        "width_max_mm": family.get("width_max_mm", family.get("patch_bandwidth_mm")),
                        "width_mean_mm": family.get("width_mean_mm"),
                        "n_branches": family.get("n_branches"),
                        "n_endpoints": family.get("n_endpoints"),
                        "n_junctions": family.get("n_junctions"),
                        "n_loops": family.get("n_loops"),
                        "tortuosity": family.get("tortuosity"),
                        "direction_entropy": family.get("direction_entropy"),
                        "anisotropy_index": family.get("anisotropy_index"),
                        "fractal_dimension": complexity.get("fractal_dimension"),
                        "complexity_index": complexity.get("complexity_index"),
                        "hazard_level": engineering.get("hazard_level"),
                    }
                )
        return rows

    def render_visualizations(
        self,
        result: dict[str, Any],
        base_image: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        normalized_mask = result["intermediate"]["normalized_mask"]
        height, width = normalized_mask.shape
        if base_image is None:
            canvas = np.full((height, width, 3), 32, dtype=np.uint8)
        else:
            canvas = self._prepare_canvas(base_image, (height, width))

        class_overlay = self._render_class_overlay(canvas.copy(), normalized_mask)
        skeleton_overlay = self._render_skeleton_overlay(canvas.copy(), result)
        node_overlay = self._render_node_overlay(canvas.copy(), result)

        return {
            "class_overlay": class_overlay,
            "skeleton_overlay": skeleton_overlay,
            "node_overlay": node_overlay,
        }

    def _prepare_canvas(self, base_image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        canvas = np.asarray(base_image).copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if canvas.shape[:2] != target_shape:
            canvas = cv2.resize(canvas, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        return canvas.astype(np.uint8)

    def _render_class_overlay(self, canvas: np.ndarray, normalized_mask: np.ndarray) -> np.ndarray:
        colors = {
            1: (60, 76, 231),
            2: (219, 152, 52),
            3: (39, 174, 96),
            4: (179, 68, 142),
        }
        overlay = canvas.copy()
        for class_id, color in colors.items():
            overlay[normalized_mask == class_id] = color
        return cv2.addWeighted(canvas, 0.45, overlay, 0.55, 0.0)

    def _render_skeleton_overlay(self, canvas: np.ndarray, result: dict[str, Any]) -> np.ndarray:
        overlay = canvas.copy()
        for class_item in result.get("classes", []):
            class_name = class_item.get("class_name")
            skeleton_info = class_item.get("skeleton")
            if not skeleton_info:
                continue
            skeleton_mask = skeleton_info.get("skeleton_mask")
            if skeleton_mask is None:
                continue
            color = self._color_for_class_name(class_name)
            overlay[np.asarray(skeleton_mask) > 0] = color
        return cv2.addWeighted(canvas, 0.5, overlay, 0.5, 0.0)

    def _render_node_overlay(self, canvas: np.ndarray, result: dict[str, Any]) -> np.ndarray:
        overlay = canvas.copy()
        for class_item in result.get("classes", []):
            skeleton_info = class_item.get("skeleton")
            if not skeleton_info:
                continue
            endpoint_mask = skeleton_info.get("endpoint_mask")
            junction_mask = skeleton_info.get("junction_mask")
            if endpoint_mask is not None:
                overlay[np.asarray(endpoint_mask) > 0] = (0, 255, 255)
            if junction_mask is not None:
                overlay[np.asarray(junction_mask) > 0] = (0, 0, 255)
        return cv2.addWeighted(canvas, 0.6, overlay, 0.4, 0.0)

    def _color_for_class_name(self, class_name: str | None) -> tuple[int, int, int]:
        mapping = {
            "Transverse": (60, 76, 231),
            "Longitudinal": (219, 152, 52),
            "fixpatch": (39, 174, 96),
            "Alligator": (179, 68, 142),
        }
        return mapping.get(class_name, (200, 200, 200))

    def _build_topology_graphs(
        self,
        class_result: dict[str, Any],
        label_map: np.ndarray,
        components: list[dict[str, Any]],
        spec: DiseaseClassSpec,
    ) -> dict[str, Any] | None:
        if not self._supports_skeleton(spec.family):
            return None

        component_graphs: list[dict[str, Any]] = []
        total_nodes = 0
        total_branches = 0
        total_endpoints = 0
        total_junctions = 0
        total_loops = 0
        branch_angles: list[float] = []
        branch_lengths: list[float] = []

        for component in components:
            component_id = int(component["component_id"])
            skeleton_info = component.get("skeleton")
            skeleton_mask = self._extract_skeleton_mask(skeleton_info)
            component_mask = (label_map == component_id).astype(np.uint8)
            graph = self._build_component_graph(
                component_mask=component_mask,
                skeleton_mask=skeleton_mask,
                component=component,
            )
            component["graph"] = graph
            component_graphs.append(graph)

            total_branches += int(graph["n_branches"])
            total_nodes += int(graph["n_nodes"])
            total_endpoints += int(graph["n_endpoints"])
            total_junctions += int(graph["n_junctions"])
            total_loops += int(graph["n_loops"])
            branch_angles.extend(graph["branch_angles_deg"])
            branch_lengths.extend(graph["branch_lengths_px"])

            if spec.family == LINEAR_FAMILY:
                component["family_geometry"]["n_endpoints"] = graph["n_endpoints"]
                component["family_geometry"]["n_junctions"] = graph["n_junctions"]
                component["family_geometry"]["n_branches"] = graph["n_branches"]
                component["family_geometry"]["tortuosity"] = graph["tortuosity"]
                component["family_geometry"]["direction_entropy"] = graph["direction_entropy"]
            if spec.family == NETWORK_FAMILY:
                component["family_geometry"]["n_endpoints"] = graph["n_endpoints"]
                component["family_geometry"]["n_junctions"] = graph["n_junctions"]
                component["family_geometry"]["n_branches"] = graph["n_branches"]
                component["family_geometry"]["n_loops"] = graph["n_loops"]
                component["family_geometry"]["direction_entropy"] = graph["direction_entropy"]
                component["family_geometry"]["anisotropy_index"] = graph["anisotropy_index"]

        summary = {
            "component_count": len(component_graphs),
            "n_nodes": int(total_nodes),
            "n_branches": int(total_branches),
            "n_endpoints": int(total_endpoints),
            "n_junctions": int(total_junctions),
            "n_loops": int(total_loops),
            "direction_entropy": self._compute_direction_entropy(branch_angles, branch_lengths),
            "anisotropy_index": self._compute_anisotropy_index_from_components(components),
            "components": component_graphs,
        }

        if spec.family == LINEAR_FAMILY:
            class_result["family_geometry"]["n_endpoints"] = summary["n_endpoints"]
            class_result["family_geometry"]["n_junctions"] = summary["n_junctions"]
            class_result["family_geometry"]["n_branches"] = summary["n_branches"]
            class_result["family_geometry"]["direction_entropy"] = summary["direction_entropy"]
            class_result["family_geometry"]["tortuosity"] = self._aggregate_linear_tortuosity(component_graphs)
        if spec.family == NETWORK_FAMILY:
            class_result["family_geometry"]["n_endpoints"] = summary["n_endpoints"]
            class_result["family_geometry"]["n_junctions"] = summary["n_junctions"]
            class_result["family_geometry"]["n_branches"] = summary["n_branches"]
            class_result["family_geometry"]["n_loops"] = summary["n_loops"]
            class_result["family_geometry"]["direction_entropy"] = summary["direction_entropy"]
            class_result["family_geometry"]["anisotropy_index"] = summary["anisotropy_index"]

        return summary

    def _build_component_graph(
        self,
        component_mask: np.ndarray,
        skeleton_mask: np.ndarray | None,
        component: dict[str, Any],
    ) -> dict[str, Any]:
        empty_graph = {
            "n_nodes": 0,
            "n_endpoints": 0,
            "n_junctions": 0,
            "n_branches": 0,
            "n_loops": 0,
            "tortuosity": None,
            "direction_entropy": None,
            "anisotropy_index": None,
            "nodes": [],
            "edges": [],
            "branch_angles_deg": [],
            "branch_lengths_px": [],
        }

        if skeleton_mask is None or not np.any(skeleton_mask):
            return empty_graph

        node_clusters, node_label_map = self._build_node_clusters(skeleton_mask)
        branch_components = self._extract_branch_components(skeleton_mask, node_label_map)
        direct_edges = self._extract_direct_node_edges(skeleton_mask, node_label_map)

        nodes_preview = [
            {
                "node_id": node["node_id"],
                "kind": node["kind"],
                "pixel_count": node["pixel_count"],
                "centroid": node["centroid"],
            }
            for node in node_clusters
        ]

        edges: list[dict[str, Any]] = []
        branch_angles: list[float] = []
        branch_lengths: list[float] = []
        n_loops = 0

        for branch_idx, branch in enumerate(branch_components, start=1):
            node_ids = branch["node_ids"]
            length_px = branch["length_px"]
            angle_deg = branch["angle_deg"]
            edge = {
                "edge_id": branch_idx,
                "kind": "branch",
                "node_ids": node_ids,
                "length_px": length_px,
                "length_mm": self._to_mm(length_px),
                "angle_deg": angle_deg,
                "pixel_count": branch["pixel_count"],
            }
            edges.append(edge)
            if angle_deg is not None and length_px is not None:
                branch_angles.append(angle_deg)
                branch_lengths.append(length_px)
            if len(node_ids) == 0:
                n_loops += 1

        edge_offset = len(edges)
        for local_idx, direct_edge in enumerate(direct_edges, start=1):
            edge = {
                "edge_id": edge_offset + local_idx,
                "kind": "direct_node_link",
                "node_ids": direct_edge["node_ids"],
                "length_px": direct_edge["length_px"],
                "length_mm": self._to_mm(direct_edge["length_px"]),
                "angle_deg": direct_edge["angle_deg"],
                "pixel_count": 0,
            }
            edges.append(edge)
            if direct_edge["angle_deg"] is not None:
                branch_angles.append(direct_edge["angle_deg"])
                branch_lengths.append(direct_edge["length_px"])

        endpoint_nodes = sum(1 for node in node_clusters if node["kind"] == "endpoint")
        junction_nodes = sum(1 for node in node_clusters if node["kind"] == "junction")
        n_branches = len(edges)
        anisotropy_index = self._compute_anisotropy_index_from_mask(skeleton_mask)
        tortuosity = self._compute_tortuosity_from_graph(node_clusters, edges)

        graph = {
            "n_nodes": len(node_clusters),
            "n_endpoints": endpoint_nodes,
            "n_junctions": junction_nodes,
            "n_branches": n_branches,
            "n_loops": int(n_loops),
            "tortuosity": tortuosity,
            "direction_entropy": self._compute_direction_entropy(branch_angles, branch_lengths),
            "anisotropy_index": anisotropy_index,
            "nodes": nodes_preview,
            "edges": edges,
            "branch_angles_deg": branch_angles,
            "branch_lengths_px": branch_lengths,
        }

        component["skeleton"]["graph_ready"] = True
        component["skeleton"]["node_count"] = len(node_clusters)
        component["skeleton"]["branch_count"] = n_branches
        return graph

    def _build_node_clusters(
        self,
        skeleton_mask: np.ndarray,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        node_info = self._classify_skeleton_nodes(skeleton_mask)
        node_pixels = (
            (node_info["endpoint_mask"] > 0) | (node_info["junction_mask"] > 0)
        ).astype(np.uint8)
        if not np.any(node_pixels):
            return [], np.zeros_like(skeleton_mask, dtype=np.int32)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            node_pixels,
            connectivity=8,
        )

        nodes: list[dict[str, Any]] = []
        relabeled = np.zeros_like(labels, dtype=np.int32)

        for label_idx in range(1, num_labels):
            cluster_mask = labels == label_idx
            pixel_count = int(stats[label_idx, cv2.CC_STAT_AREA])
            kind = "junction" if np.any(node_info["junction_mask"][cluster_mask] > 0) else "endpoint"
            node_id = len(nodes) + 1
            relabeled[cluster_mask] = node_id
            nodes.append(
                {
                    "node_id": node_id,
                    "kind": kind,
                    "pixel_count": pixel_count,
                    "centroid": {
                        "x": float(centroids[label_idx][0]),
                        "y": float(centroids[label_idx][1]),
                    },
                }
            )

        return nodes, relabeled

    def _extract_branch_components(
        self,
        skeleton_mask: np.ndarray,
        node_label_map: np.ndarray,
    ) -> list[dict[str, Any]]:
        branch_core = ((skeleton_mask > 0) & (node_label_map == 0)).astype(np.uint8)
        if not np.any(branch_core):
            return []

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(branch_core, connectivity=8)
        branches: list[dict[str, Any]] = []

        for label_idx in range(1, num_labels):
            branch_mask = (labels == label_idx).astype(np.uint8)
            pixel_count = int(stats[label_idx, cv2.CC_STAT_AREA])
            dilated = cv2.dilate(branch_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            touching_nodes = sorted(
                int(node_id)
                for node_id in np.unique(node_label_map[dilated > 0]).tolist()
                if int(node_id) > 0
            )
            branches.append(
                {
                    "pixel_count": pixel_count,
                    "node_ids": touching_nodes,
                    "length_px": self._estimate_skeleton_length(branch_mask),
                    "angle_deg": self._compute_branch_angle(branch_mask),
                }
            )

        return branches

    def _extract_direct_node_edges(
        self,
        skeleton_mask: np.ndarray,
        node_label_map: np.ndarray,
    ) -> list[dict[str, Any]]:
        if not np.any(skeleton_mask) or not np.any(node_label_map):
            return []

        seen_pairs: set[tuple[int, int]] = set()
        direct_edges: list[dict[str, Any]] = []
        height, width = skeleton_mask.shape

        for y in range(height):
            for x in range(width):
                node_id = int(node_label_map[y, x])
                if node_id <= 0:
                    continue
                for ny, nx in self._iter_neighbors(y, x, height, width):
                    other_id = int(node_label_map[ny, nx])
                    if other_id <= 0 or other_id == node_id:
                        continue
                    pair = tuple(sorted((node_id, other_id)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    angle = float(np.degrees(np.arctan2(ny - y, nx - x))) % 180.0
                    length_px = float(np.hypot(nx - x, ny - y))
                    direct_edges.append(
                        {
                            "node_ids": [pair[0], pair[1]],
                            "length_px": length_px,
                            "angle_deg": angle,
                        }
                    )

        return direct_edges

    def _compute_branch_angle(self, branch_mask: np.ndarray) -> float | None:
        ys, xs = np.where(branch_mask > 0)
        if xs.size < 2:
            return None
        points = np.column_stack((xs, ys)).astype(np.float32)
        angle = self._compute_pca_angle(points)
        if angle is None:
            return None
        return float(angle % 180.0)

    def _compute_tortuosity_from_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> float | None:
        endpoints = [node for node in nodes if node["kind"] == "endpoint"]
        if len(endpoints) < 2:
            return None

        total_length = sum(
            float(edge["length_px"])
            for edge in edges
            if edge.get("length_px") is not None
        )
        if total_length <= 0:
            return None

        max_distance = 0.0
        for idx, node_a in enumerate(endpoints):
            ax = node_a["centroid"]["x"]
            ay = node_a["centroid"]["y"]
            for node_b in endpoints[idx + 1 :]:
                bx = node_b["centroid"]["x"]
                by = node_b["centroid"]["y"]
                max_distance = max(max_distance, float(np.hypot(bx - ax, by - ay)))

        if max_distance <= 0:
            return None
        return float(total_length / max_distance)

    def _compute_direction_entropy(
        self,
        angles_deg: list[float],
        weights: list[float] | None = None,
        bins: int = 12,
    ) -> float | None:
        if not angles_deg:
            return None

        angles = np.asarray(angles_deg, dtype=np.float64) % 180.0
        if weights is None or len(weights) != len(angles_deg):
            weights_array = np.ones_like(angles, dtype=np.float64)
        else:
            weights_array = np.asarray(weights, dtype=np.float64)

        hist, _ = np.histogram(angles, bins=bins, range=(0.0, 180.0), weights=weights_array)
        total = hist.sum()
        if total <= 0:
            return None

        probs = hist[hist > 0] / total
        return float(-(probs * np.log2(probs)).sum())

    def _compute_anisotropy_index_from_mask(self, mask: np.ndarray) -> float | None:
        ys, xs = np.where(mask > 0)
        if xs.size < 2:
            return None

        points = np.column_stack((xs, ys)).astype(np.float64)
        centered = points - points.mean(axis=0, keepdims=True)
        cov = np.cov(centered, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(cov)
        eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]
        if eigenvalues.shape[0] < 2 or eigenvalues[0] <= 0:
            return None
        return float(1.0 - (eigenvalues[1] / eigenvalues[0]))

    def _compute_anisotropy_index_from_components(
        self,
        components: list[dict[str, Any]],
    ) -> float | None:
        values = [
            graph_value
            for component in components
            for graph_value in [component.get("graph", {}).get("anisotropy_index")]
            if graph_value is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    def _aggregate_linear_tortuosity(self, component_graphs: list[dict[str, Any]]) -> float | None:
        values = [
            graph["tortuosity"]
            for graph in component_graphs
            if graph.get("tortuosity") is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    def _iter_neighbors(
        self,
        y: int,
        x: int,
        height: int,
        width: int,
    ) -> list[tuple[int, int]]:
        neighbors: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbors.append((ny, nx))
        return neighbors

    def _extract_skeletons(
        self,
        label_map: np.ndarray,
        components: list[dict[str, Any]],
        spec: DiseaseClassSpec,
    ) -> dict[str, Any] | None:
        if not self._supports_skeleton(spec.family):
            return None

        empty = np.zeros_like(label_map, dtype=np.uint8)
        if not components:
            return {
                "skeleton_mask": empty,
                "endpoint_mask": empty,
                "junction_mask": empty,
                "edge_point_mask": empty,
                "component_count": 0,
            }

        skeleton_mask = np.zeros_like(label_map, dtype=np.uint8)
        endpoint_mask = np.zeros_like(label_map, dtype=np.uint8)
        junction_mask = np.zeros_like(label_map, dtype=np.uint8)
        edge_point_mask = np.zeros_like(label_map, dtype=np.uint8)

        total_endpoints = 0
        total_junctions = 0
        total_skeleton_pixels = 0

        for component in components:
            component_id = int(component["component_id"])
            component_mask = (label_map == component_id).astype(np.uint8)
            component_skeleton = self._skeletonize(component_mask)
            node_info = self._classify_skeleton_nodes(component_skeleton)

            skeleton_mask[component_skeleton > 0] = 1
            endpoint_mask[node_info["endpoint_mask"] > 0] = 1
            junction_mask[node_info["junction_mask"] > 0] = 1
            edge_point_mask[node_info["edge_point_mask"] > 0] = 1

            total_skeleton_pixels += int(component_skeleton.sum())
            total_endpoints += node_info["n_endpoints"]
            total_junctions += node_info["n_junctions"]

            component["skeleton"] = {
                "applied": True,
                "skeleton_mask": component_skeleton,
                "skeleton_pixel_count": int(component_skeleton.sum()),
                "n_endpoints": node_info["n_endpoints"],
                "n_junctions": node_info["n_junctions"],
                "n_edge_points": node_info["n_edge_points"],
                "endpoints_preview": node_info["endpoints_preview"],
                "junctions_preview": node_info["junctions_preview"],
            }

            if spec.family == LINEAR_FAMILY:
                component["family_geometry"]["n_endpoints"] = node_info["n_endpoints"]
                component["family_geometry"]["n_junctions"] = node_info["n_junctions"]
            if spec.family == NETWORK_FAMILY:
                component["family_geometry"]["n_endpoints"] = node_info["n_endpoints"]
                component["family_geometry"]["n_junctions"] = node_info["n_junctions"]

        return {
            "skeleton_mask": skeleton_mask,
            "endpoint_mask": endpoint_mask,
            "junction_mask": junction_mask,
            "edge_point_mask": edge_point_mask,
            "component_count": len(components),
            "total_skeleton_pixels": int(total_skeleton_pixels),
            "total_endpoints": int(total_endpoints),
            "total_junctions": int(total_junctions),
        }

    def _supports_skeleton(self, family: str) -> bool:
        return family in {LINEAR_FAMILY, NETWORK_FAMILY}

    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        binary = (np.asarray(mask) > 0).astype(np.uint8)
        if not binary.any():
            return binary

        skeleton = np.zeros_like(binary, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        working = (binary * 255).astype(np.uint8)

        while True:
            eroded = cv2.erode(working, element)
            opened = cv2.dilate(eroded, element)
            residue = cv2.subtract(working, opened)
            skeleton = cv2.bitwise_or(skeleton, residue)
            working = eroded
            if cv2.countNonZero(working) == 0:
                break

        return (skeleton > 0).astype(np.uint8)

    def _classify_skeleton_nodes(self, skeleton: np.ndarray) -> dict[str, Any]:
        binary = (np.asarray(skeleton) > 0).astype(np.uint8)
        empty = np.zeros_like(binary, dtype=np.uint8)
        if not binary.any():
            return {
                "endpoint_mask": empty,
                "junction_mask": empty,
                "edge_point_mask": empty,
                "n_endpoints": 0,
                "n_junctions": 0,
                "n_edge_points": 0,
                "endpoints_preview": [],
                "junctions_preview": [],
            }

        padded = np.pad(binary, pad_width=1, mode="constant", constant_values=0)
        neighbor_count = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )

        endpoint_mask = ((binary == 1) & (neighbor_count == 1)).astype(np.uint8)
        junction_mask = ((binary == 1) & (neighbor_count >= 3)).astype(np.uint8)
        edge_point_mask = ((binary == 1) & (neighbor_count == 2)).astype(np.uint8)

        endpoint_coords = np.argwhere(endpoint_mask > 0)
        junction_coords = np.argwhere(junction_mask > 0)

        return {
            "endpoint_mask": endpoint_mask,
            "junction_mask": junction_mask,
            "edge_point_mask": edge_point_mask,
            "n_endpoints": int(endpoint_mask.sum()),
            "n_junctions": int(junction_mask.sum()),
            "n_edge_points": int(edge_point_mask.sum()),
            "endpoints_preview": self._coords_preview(endpoint_coords),
            "junctions_preview": self._coords_preview(junction_coords),
        }

    def _coords_preview(self, coords: np.ndarray, limit: int = 20) -> list[dict[str, int]]:
        preview = []
        for y, x in coords[:limit].tolist():
            preview.append({"x": int(x), "y": int(y)})
        return preview

    def _analyze_linear_components(self, mask: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError("Linear crack analysis will be added later.")

    def _analyze_patch_components(self, mask: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError("Patch analysis will be added later.")

    def _analyze_network_components(self, mask: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError("Alligator analysis will be added later.")


if __name__ == "__main__":
    demo_mask = np.array(
        [
            [0, 0, 1, 1, 0, 3],
            [0, 0, 1, 1, 0, 3],
            [0, 2, 2, 0, 4, 4],
            [0, 2, 0, 0, 4, 4],
        ],
        dtype=np.uint8,
    )

    extractor = CrackGeometryExtractor(calibration=0.2, min_area=1)
    result = extractor.extract(demo_mask)

    print("Stage:", result["stage"])
    print("Mask mode:", result["image"]["mask_mode"])
    print("Classes present:", result["image"]["class_ids_present"])
    print("Active classes:", result["summary"]["active_class_names"])
    print("Export class rows:", len(result["exports"]["class_rows"]))
    print("Export component rows:", len(result["exports"]["component_rows"]))
