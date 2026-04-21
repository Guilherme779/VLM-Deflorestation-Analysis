"""Add grounding bounding boxes for converted land (pasture/agriculture).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils_io import (
    atomic_write_json,
    iter_meta_json_paths,
)

import numpy as np

try:  # optional scipy for connected components
    from scipy import ndimage as _ndimage  # type: ignore
except Exception:
    _ndimage = None

import rasterio


IGNORE_LABEL = 255
CLASS_FOREST = 0
CLASS_PASTURE = 1
CLASS_AGRICULTURE = 2


@dataclass
class ComponentStats:
    label: int
    area: int
    bbox_px: Tuple[int, int, int, int]


@dataclass(frozen=True)
class GroundingConfig:
    """Configuration for grounding bbox generation."""

    top_k: int
    min_area_px: int
    min_area_frac: float
    conversion_fraction_threshold: float
    overwrite: bool
    dry_run: bool


def load_label(path: str | Path) -> np.ndarray:
    """Load label raster as a 2D numpy array using rasterio.

    Returns an array of shape (H, W) with integer labels.
    """

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Label raster not found: {p}")

    with rasterio.open(p) as src:
        arr = src.read(1)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D label raster, got shape {arr.shape} for {p}")
    return arr


def _connected_components_8_numpy(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Pure-numpy 8-connected components.

    Returns (labels, num_labels) where labels has 0 for background and
    1..num_labels for components.
    """

    # Simple BFS-based labeling for 8-neighborhood.
    h, w = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    current_label = 0

    # Offsets for 8-neighborhood
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]

    mask_bool = mask.astype(bool)

    for y in range(h):
        for x in range(w):
            if not mask_bool[y, x] or labels[y, x] != 0:
                continue
            current_label += 1
            # BFS/stack
            stack = [(y, x)]
            labels[y, x] = current_label
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask_bool[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label
                            stack.append((ny, nx))

    return labels, current_label


def connected_components_8(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Connected components with 8-connectivity.

    If scipy.ndimage is available, use ndimage.label with a full 3x3
    structure. Otherwise, fall back to a pure-numpy implementation.

    Returns (labels, num_labels).
    """

    if mask.size == 0:
        return np.zeros_like(mask, dtype=np.int32), 0

    if _ndimage is not None:
        structure = np.ones((3, 3), dtype=np.int8)
        labeled, n = _ndimage.label(mask.astype(bool), structure=structure)
        return labeled.astype(np.int32), int(n)

    return _connected_components_8_numpy(mask)


def _bbox_from_coords(coords: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute inclusive pixel bbox [x1, y1, x2, y2] from (y, x) coords."""

    if coords.size == 0:
        raise ValueError("Cannot compute bbox from empty coords")
    ys = coords[:, 0]
    xs = coords[:, 1]
    y1 = int(ys.min())
    y2 = int(ys.max())
    x1 = int(xs.min())
    x2 = int(xs.max())
    return x1, y1, x2, y2


def normalize_bbox(bbox_px: Sequence[int], w: int, h: int) -> List[float]:
    """Normalize inclusive pixel bbox [x1, y1, x2_incl, y2_incl] by (w, h).

    The input bbox is expressed in inclusive pixel indices on a raster of
    width ``w`` and height ``h``. Normalized coordinates follow an
    exclusive-right/bottom edge convention:

        - pixel bbox: [x1, y1, x2_incl, y2_incl]
        - normalized bbox: [nx1, ny1, nx2, ny2] with x2, y2 **exclusive**

    computed as::

        nx1 = x1 / w
        ny1 = y1 / h
        nx2 = (x2_incl + 1) / w
        ny2 = (y2_incl + 1) / h

    and then clipped to [0, 1]. This guarantees
    0 <= nx1 < nx2 <= 1 and 0 <= ny1 < ny2 <= 1 for non-empty boxes.
    """

    if len(bbox_px) != 4:
        raise ValueError(f"bbox_px must have length 4, got {bbox_px}")

    x1, y1, x2, y2 = map(float, bbox_px)
    if w <= 0 or h <= 0:
        raise ValueError("Width and height must be positive")

    nx1 = max(0.0, min(1.0, x1 / w))
    ny1 = max(0.0, min(1.0, y1 / h))
    nx2 = max(0.0, min(1.0, (x2 + 1) / w))
    ny2 = max(0.0, min(1.0, (y2 + 1) / h))

    # Ensure ordering
    if nx2 < nx1:
        nx1, nx2 = nx2, nx1
    if ny2 < ny1:
        ny1, ny2 = ny2, ny1

    # Avoid degenerate boxes
    eps = 1e-6
    if nx2 - nx1 < eps:
        nx2 = min(1.0, nx1 + eps)
    if ny2 - ny1 < eps:
        ny2 = min(1.0, ny1 + eps)

    return [float(nx1), float(ny1), float(nx2), float(ny2)]


def component_bboxes_from_mask(
        mask: np.ndarray,
        top_k: int,
        min_area_px: int,
        min_area_frac: float,
        valid_pixel_count: int,
) -> Tuple[
        Optional[Tuple[int, int, int, int]],
        List[Tuple[int, int, int, int]],
        Dict[str, int],
]:
    """Compute union and per-component bboxes from a boolean mask.

    mask: boolean array where True indicates converted pixels.
    top_k: maximum number of components to keep.
    min_area_px: minimum absolute area in pixels.
    min_area_frac: minimum fraction of valid pixels; whichever is larger
            between min_area_px and (min_area_frac * valid_pixel_count)
            becomes the effective threshold.
    valid_pixel_count: number of pixels with lbl != IGNORE.

    Returns (union_bbox_all_px, component_bboxes_px, stats_dict).

    - union_bbox_all_px is an inclusive bbox over **all** converted pixels
        (mask union) or None if there are no converted pixels at all.
    - component_bboxes_px is a list of inclusive bboxes for filtered
        components, sorted by area descending.
    - stats_dict contains pixel_count, n_components, kept_components.
    """

    if mask.size == 0:
        stats = {"pixel_count": 0, "n_components": 0, "kept_components": 0}
        return None, [], stats

    mask_bool = mask.astype(bool)
    pixel_count = int(mask_bool.sum())

    if pixel_count == 0:
        stats = {"pixel_count": 0, "n_components": 0, "kept_components": 0}
        return None, [], stats

    # Union bbox over *all* converted pixels (mask union), independent of
    # component filtering.
    coords_all = np.argwhere(mask_bool)
    union_bbox_all = _bbox_from_coords(coords_all)

    labels, n_components = connected_components_8(mask_bool)

    if n_components == 0:
        stats = {
            "pixel_count": pixel_count,
            "n_components": 0,
            "kept_components": 0,
        }
        return union_bbox_all, [], stats

    eff_min_area = max(int(min_area_px), int(np.ceil(min_area_frac * max(valid_pixel_count, 1))))

    components: List[ComponentStats] = []
    for label in range(1, n_components + 1):
        coords = np.argwhere(labels == label)
        area = int(coords.shape[0])
        if area < eff_min_area:
            continue
        bbox_px = _bbox_from_coords(coords)
        components.append(ComponentStats(label=label, area=area, bbox_px=bbox_px))

    # Sort by area descending
    components.sort(key=lambda c: c.area, reverse=True)

    kept = components[: max(0, top_k)] if top_k > 0 else components
    bboxes_px = [c.bbox_px for c in kept]

    stats = {
        "pixel_count": pixel_count,
        "n_components": int(n_components),
        "kept_components": int(len(kept)),
    }
    return union_bbox_all, bboxes_px, stats


def _compute_converted_mask(lbl: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return conversion mask and valid pixel count from label raster.

    Converted = (pasture OR agriculture) AND lbl != IGNORE_LABEL.
    """

    valid = lbl != IGNORE_LABEL
    valid_pixel_count = int(valid.sum())
    converted = ((lbl == CLASS_PASTURE) | (lbl == CLASS_AGRICULTURE)) & valid
    return converted, valid_pixel_count


def _update_meta_json(
    meta_path: Path,
    qa_entry: Dict[str, object],
    overwrite: bool,
    dry_run: bool,
) -> None:
    """Inject grounding_converted into a meta.json file.
    """

    if not meta_path.is_file():
        print(f"[WARN] meta.json not found: {meta_path}", file=sys.stderr)
        return

    with meta_path.open("r", encoding="utf-8") as f:
        try:
            meta = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON {meta_path}: {e}", file=sys.stderr)
            return

    # Legacy support: if grounding_converted lives under qa and overwrite
    # is False, respect that and do nothing.
    qa = meta.get("qa") or {}
    if not overwrite and isinstance(qa, dict) and "grounding_converted" in qa:
        if dry_run:
            print(f"[DRY-RUN] Would keep existing qa.grounding_converted in {meta_path}")
        return

    # New schema: store grounding_converted at the top level.
    meta["grounding_converted"] = qa_entry

    # Optional: remove legacy nested field if present and we're overwriting.
    if isinstance(qa, dict) and "grounding_converted" in qa:
        if overwrite:
            qa.pop("grounding_converted", None)
            if qa:
                meta["qa"] = qa
            elif "qa" in meta:
                # Drop empty qa dict to keep meta.json tidy.
                del meta["qa"]

    if dry_run:
        print(f"[DRY-RUN] Would update {meta_path}")
        return

    atomic_write_json(meta_path, meta)


class GroundingStage:
    """Stage-style grounding generator used from RegionPipeline.

    Reads label rasters, computes bounding boxes, and:
    - Writes ``grounding_converted`` into per-patch meta.json.
    - Writes back into the index via :meth:`PatchIndexBuilder.update_records`.
    """

    def __init__(
        self,
        *,
        patch_root: Path,
        index_builder,          # PatchIndexBuilder — import avoided to prevent cycles
        repo_root: Path,
        config: GroundingConfig,
    ) -> None:
        self.patch_root = patch_root
        self.index_builder = index_builder
        self.repo_root = repo_root
        self.config = config
        self.repo_root = repo_root
        self.config = config

    # ---- core grounding logic ----

    def build_answer(self, lbl: np.ndarray) -> Dict[str, object]:
        """Compute grounding_converted answer dict for a single patch."""

        h, w = lbl.shape
        converted_mask, valid_pixel_count = _compute_converted_mask(lbl)
        union_bbox_all_px, comp_bboxes_px, stats = component_bboxes_from_mask(
            converted_mask,
            top_k=self.config.top_k,
            min_area_px=self.config.min_area_px,
            min_area_frac=self.config.min_area_frac,
            valid_pixel_count=valid_pixel_count,
        )

        pixel_count = stats.get("pixel_count", 0)
        conversion_fraction = (pixel_count / valid_pixel_count) if valid_pixel_count > 0 else 0.0

        present = bool(
            valid_pixel_count > 0
            and pixel_count > 0
            and conversion_fraction >= self.config.conversion_fraction_threshold
            and union_bbox_all_px is not None
        )

        if present and union_bbox_all_px is not None:
            union_bbox_norm = normalize_bbox(union_bbox_all_px, w=w, h=h)
            comp_bboxes_norm = [normalize_bbox(b, w=w, h=h) for b in comp_bboxes_px]
        else:
            union_bbox_norm = None
            comp_bboxes_norm = []

        answer = {
            "present": present,
            "union_bbox": union_bbox_norm,
            "component_bboxes": comp_bboxes_norm,
        }

        qa_entry = {
            "question": "return the coordinates of the deforestated area.",
            "answer": answer,
        }

        return qa_entry

    # ---- public API ----

    def run(self) -> None:
        """Compute grounding for all patches and write to meta.json + index.

        For each patch:
        1. Read the label raster to compute bounding boxes.
        2. Write ``grounding_converted`` into per-patch meta.json.
        3. Write ``grounding_converted`` back into the index via
           :meth:`PatchIndexBuilder.update_records`.
        """
        # Phase 1: build patch_id → qa_entry mapping from label rasters.
        grounding_map: Dict[str, object] = {}
        processed = 0
        errors = 0

        for meta_path in iter_meta_json_paths(self.patch_root):
            with meta_path.open("r", encoding="utf-8") as f:
                try:
                    meta = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Failed to parse {meta_path}: {e}", file=sys.stderr)
                    errors += 1
                    continue

            patch_id = str(meta.get("patch_id", ""))

            # Resolve label path.
            label_path_str = meta.get("label_path")
            if label_path_str:
                label_path_full = resolve_path(label_path_str, repo_root=self.repo_root, base=meta_path.parent)
            else:
                candidate1 = meta_path.parent / "mapbiomas_5c.tif"
                candidate2 = meta_path.parent / "label.tif"
                if candidate1.is_file():
                    label_path_full = candidate1
                elif candidate2.is_file():
                    label_path_full = candidate2
                else:
                    print(f"[ERROR] No label file found in {meta_path.parent}", file=sys.stderr)
                    errors += 1
                    continue

            try:
                lbl = load_label(label_path_full)
            except Exception as e:
                print(f"[ERROR] Failed to load label {label_path_full}: {e}", file=sys.stderr)
                errors += 1
                continue

            qa_entry = self.build_answer(lbl)

            # Write into meta.json (unless dry_run).
            _update_meta_json(
                meta_path=meta_path,
                qa_entry=qa_entry,
                overwrite=self.config.overwrite,
                dry_run=self.config.dry_run,
            )

            if patch_id:
                grounding_map[patch_id] = qa_entry

            processed += 1
            if processed % 500 == 0:
                print(f"[grounding] processed={processed}", file=sys.stderr)

        print(f"[grounding] meta.json pass done. processed={processed}, errors={errors}", file=sys.stderr)

        # Phase 2: write grounding_converted back into the index.
        def _transform(rec: dict) -> dict:
            pid = str(rec.get("patch_id", ""))
            qa_entry = grounding_map.get(pid)
            if qa_entry is None:
                return rec
            if "grounding_converted" in rec and not self.config.overwrite:
                return rec
            rec["grounding_converted"] = qa_entry
            return rec

        if not self.config.dry_run:
            written = self.index_builder.update_records(_transform)
            print(f"[grounding] Index updated. records={written}", file=sys.stderr)


def _self_check() -> None:
    """Generate synthetic masks and print bboxes for quick verification."""

    print("Running self-check for component_bboxes_from_mask()", file=sys.stderr)

    h, w = 16, 16

    # One blob in the center
    lbl1 = np.full((h, w), CLASS_FOREST, dtype=np.uint8)
    lbl1[4:8, 4:8] = CLASS_PASTURE
    converted1, valid1 = _compute_converted_mask(lbl1)
    ub1, cb1, stats1 = component_bboxes_from_mask(converted1, 3, 1, 0.0, valid1)
    print("Case 1 - one blob:")
    print(" union_bbox_px=", ub1)
    print(" comp_bboxes_px=", cb1)
    print(" stats=", stats1)

    # Two blobs
    lbl2 = np.full((h, w), CLASS_FOREST, dtype=np.uint8)
    lbl2[1:4, 1:4] = CLASS_PASTURE
    lbl2[10:14, 10:15] = CLASS_AGRICULTURE
    converted2, valid2 = _compute_converted_mask(lbl2)
    ub2, cb2, stats2 = component_bboxes_from_mask(converted2, 3, 1, 0.0, valid2)
    print("Case 2 - two blobs:")
    print(" union_bbox_px=", ub2)
    print(" comp_bboxes_px=", cb2)
    print(" stats=", stats2)

    # Many tiny scattered pixels (should be filtered by min_area thresholds)
    lbl3 = np.full((h, w), CLASS_FOREST, dtype=np.uint8)
    lbl3[::3, ::3] = CLASS_PASTURE
    converted3, valid3 = _compute_converted_mask(lbl3)
    ub3, cb3, stats3 = component_bboxes_from_mask(converted3, 3, 10, 0.2, valid3)
    print("Case 3 - scattered tiny:")
    print(" union_bbox_px=", ub3)
    print(" comp_bboxes_px=", cb3)
    print(" stats=", stats3)


def resolve_path(p: str | Path, *, repo_root: Path, base: Optional[Path] = None) -> Path:
    """Resolve ``p`` to an absolute ``Path``.

    Resolution rules:

    - If ``p`` is absolute, return it as-is.
    - Else if ``base`` is provided, return ``(base / p).resolve()``.
    - Otherwise, resolve relative to ``repo_root``.
    """

    p_path = Path(p)
    if p_path.is_absolute():
        return p_path
    if base is not None:
        return (base / p_path).resolve()
    return (repo_root / p_path).resolve()


    # CLI entrypoints removed; this module is now library-only and used
    # via a single stage-style class from RegionPipeline.
