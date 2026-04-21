"""Upsample a patch dataset by creating rotated copies of each patch.

This is a *file-based* augmentation:
- For each patch folder under PATCH_ROOT, create a sibling folder with a suffix
  (default: "_rot90")
- Rotate GeoTIFF rasters by 90 degrees clockwise (image + labels)
- Update dataset_index.jsonl by appending new records for the rotated patches
- Update split files (train/val/test) so each augmented patch stays in the same
  split as its source patch (avoids leakage).

Notes on georeferencing:
- We rotate pixel grids and update the GeoTIFF affine transform so that the
  rotated raster still maps to the same world footprint.

Typical usage (standalone):
  python scripts/pipeline/augment_rotate_patches.py \
    --patch-root data/tiles/santarem_s2_2021_patches

Pipeline runner can call this via scripts/pipeline/run_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine


class PatchAugmenter:
    """File-based patch augmentation via rotations.

    This mirrors the previous augment_rotate90 function but supports multiple
    rotation angles and encapsulates the logic in a class.
    """

    def __init__(
        self,
        *,
        patch_root: Path,
        rot_suffix: str = "_rot90",
        angles: list[int] | None = None,
        overwrite: bool = False,
        update_splits: bool = True,
    ) -> None:
        self.patch_root = patch_root.resolve()
        self.rot_suffix = rot_suffix
        # default: 90, 180, 270
        self.angles = angles or [90, 180, 270]
        self.overwrite = overwrite
        self.update_splits = update_splits
        self.explicit_tifs = [
            "s2_rgbnir.tif",
            "mapbiomas.tif",
            "mapbiomas_5c.tif",
            "prodes_deforestation.tif",
            "prodes_no_forest.tif",
            "prodes_residual.tif",
            "prodes_hydrography.tif",
        ]

    def run(self) -> None:
        split_map = self._load_split_map()
        patch_dirs = sorted([p for p in self.patch_root.iterdir() if p.is_dir()])

        created = 0
        new_ids_by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}

        for pdir in patch_dirs:
            src_id = pdir.name

            for angle in self.angles:
                dst_id = self._dst_id(src_id, angle)
                dst_dir = self.patch_root / dst_id

                if dst_dir.exists() and not self.overwrite:
                    continue

                dst_dir.mkdir(parents=True, exist_ok=True)
                self._copy_meta(pdir, dst_dir)
                self._rotate_all_tifs(pdir, dst_dir, angle)

                created += 1

                if self.update_splits and src_id in split_map:
                    split = split_map[src_id]
                    if split in new_ids_by_split:
                        new_ids_by_split[split].append(dst_id)

        if self.update_splits:
            self._append_to_split_files(new_ids_by_split=new_ids_by_split)

        print(f"Patch root: {self.patch_root}")
        print(f"Created rotated patches: {created}")
        

    def _dst_id(self, src_id: str, angle: int) -> str:
        """Construct destination patch id.

        Always uses the clean _rot{angle} suffix:
          90  -> {src_id}_rot90
          180 -> {src_id}_rot180
          270 -> {src_id}_rot270
        """

        return f"{src_id}_rot{angle % 360}"

    @staticmethod
    def _copy_meta(src_dir: Path, dst_dir: Path) -> None:
        meta_src = src_dir / "meta.json"
        if meta_src.exists():
            shutil.copy2(meta_src, dst_dir / "meta.json")

    def _rotate_all_tifs(self, src_dir: Path, dst_dir: Path, angle: int) -> None:
        # Rotate explicit known rasters first.
        for name in self.explicit_tifs:
            src_tif = src_dir / name
            if not src_tif.exists():
                continue
            self._rotate_geotiff(src_tif, dst_dir / name, angle=angle)

        # Rotate any other *.tif (except skipped suffixes), preserving filenames.
        for src_tif in sorted(src_dir.glob("*.tif")):
            if src_tif.name in self.explicit_tifs:
                continue
            self._rotate_geotiff(src_tif, dst_dir / src_tif.name, angle=angle)

    def _rot_k(self, arr: np.ndarray, k: int) -> np.ndarray:
        """Rotate an array by k * 90 degrees counter-clockwise.

        Accepts (H,W) or (C,H,W).
        """

        if arr.ndim == 2:
            return np.rot90(arr, k=k)
        if arr.ndim == 3:
            return np.rot90(arr, k=k, axes=(1, 2))
        raise ValueError(f"Expected 2D or 3D array, got shape={arr.shape}")

    def _rotate_transform(self, transform: Affine, height: int, width: int, angle: int) -> Affine:
        """Return new transform after rotating the pixel grid by angle degrees.

        Angle must be one of {90, 180, 270}. We keep the same world footprint.
        """

        angle = angle % 360
        if angle not in {90, 180, 270}:
            raise ValueError(f"Unsupported rotation angle: {angle}")

        if angle == 90:
            # new[r, c] = old[H-1-c, r]
            m = Affine(0, 1, 0, -1, 0, height - 1)
            return transform * m
        if angle == 180:
            # new[r, c] = old[H-1-r, W-1-c]
            m = Affine(-1, 0, width - 1, 0, -1, height - 1)
            return transform * m
        # angle == 270 (or -90): new[r, c] = old[c, W-1-r]
        m = Affine(0, -1, width - 1, 1, 0, 0)
        return transform * m

    def _rotate_geotiff(self, src_path: Path, dst_path: Path, *, angle: int) -> None:
        if dst_path.exists() and not self.overwrite:
            return

        with rasterio.open(src_path) as src:
            arr = src.read()  # (C,H,W)
            profile = src.profile
            height = int(src.height)
            width = int(src.width)

        # numpy.rot90 uses counter-clockwise; angles are clockwise, so k = (360-angle)/90
        k = ((360 - (angle % 360)) // 90) % 4
        rot = self._rot_k(arr, k)
        new_transform = self._rotate_transform(profile["transform"], height, width, angle)

        profile = dict(profile)
        profile.update(
            height=int(rot.shape[1]),
            width=int(rot.shape[2]),
            transform=new_transform,
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(rot)

    def _load_split_map(self) -> dict[str, str]:
        split_map: dict[str, str] = {}

        def _load(txt: Path, name: str):
            if not txt.exists():
                return
            for ln in txt.read_text(encoding="utf-8").splitlines():
                pid = ln.strip()
                if not pid:
                    continue
                split_map[pid] = name

        _load(self.patch_root / "train.txt", "train")
        _load(self.patch_root / "val.txt", "val")
        _load(self.patch_root / "test.txt", "test")

        return split_map

    def _append_to_split_files(self, *, new_ids_by_split: dict[str, list[str]]) -> None:
        def _append(path: Path, ids: list[str]):
            if not ids:
                return
            existing: set[str] = set()
            if path.exists():
                existing = {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}
            to_add = [pid for pid in ids if pid not in existing]
            if not to_add:
                return
            with path.open("a", encoding="utf-8") as f:
                for pid in to_add:
                    f.write(pid + "\n")

        _append(self.patch_root / "train.txt", new_ids_by_split.get("train", []))
        _append(self.patch_root / "val.txt", new_ids_by_split.get("val", []))
        _append(self.patch_root / "test.txt", new_ids_by_split.get("test", []))


