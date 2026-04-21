
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

@dataclass
class SceneTiler:
    """Cut a big Sentinel-2 scene into fixed-size patches.

    This wraps the original ``tile_scene`` script into a reusable object
    that can also be driven from an object-oriented pipeline.
    """

    scene_path: Path
    out_dir: Path
    region: str
    year: int
    patch_size: int = 256
    stride: int = 256
    max_empty_fraction: float = 0.95

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

        with rasterio.open(self.scene_path) as src:
            height, width = src.height, src.width
            profile = src.profile
            nodata = src.nodata
            print(f"- Tiling {self.region} -")
            print(f"Scene shape: {height} x {width}, bands: {profile['count']}")
            print(f"Nodata value: {nodata}")

            patch_count = 0
            skipped_empty = 0

            for row, col, window in self._iter_windows(height, width):
                patch = src.read(window=window)
                valid_fraction = self._compute_valid_fraction(patch, nodata)

                if valid_fraction < (1 - self.max_empty_fraction):
                    skipped_empty += 1
                    continue

                patch_dir, patch_id = self._prepare_patch_dir(row, col)
                self._write_patch(src, patch, window, profile, patch_dir)
                self._write_meta(patch_dir, patch_id, row, col, valid_fraction)

                patch_count += 1

            print(
                f"Done. Kept {patch_count} patches, "
                f"skipped {skipped_empty} mostly-empty patches."
            )

    def _iter_windows(self, height: int, width: int):
        """Yield (row, col, Window) over the scene grid."""
        for row in range(0, height - self.patch_size + 1, self.stride):
            for col in range(0, width - self.patch_size + 1, self.stride):
                yield row, col, Window(col, row, self.patch_size, self.patch_size)

    def _compute_valid_fraction(self, patch: np.ndarray, nodata) -> float:
        """Compute fraction of non-empty pixels in a patch."""
        if nodata is not None:
            mask_valid = ~(np.isclose(patch, nodata))
            return float(mask_valid.any(axis=0).mean())

        nonzero = np.any(patch != 0, axis=0)
        return float(nonzero.mean())

    def _prepare_patch_dir(self, row: int, col: int):
        scene_id = self.scene_path.stem
        patch_id = f"{scene_id}_r{row}_c{col}"
        patch_dir = self.out_dir / patch_id
        patch_dir.mkdir(parents=True, exist_ok=True)
        return patch_dir, patch_id

    def _write_patch(
        self,
        src: rasterio.io.DatasetReader,
        patch: np.ndarray,
        window: Window,
        profile: dict,
        patch_dir: Path,
    ) -> None:
        patch_profile = profile.copy()
        patch_profile.update(
            {
                "height": self.patch_size,
                "width": self.patch_size,
                "transform": rasterio.windows.transform(window, src.transform),
            }
        )

        out_img_path = patch_dir / "s2_rgbnir.tif"
        with rasterio.open(out_img_path, "w", **patch_profile) as dst:
            dst.write(patch)

    def _write_meta(
        self,
        patch_dir: Path,
        patch_id: str,
        row: int,
        col: int,
        valid_fraction: float,
    ) -> None:
        meta = {
            "scene_file": str(self.scene_path),
            "patch_id": patch_id,
            "row": int(row),
            "col": int(col),
            "patch_size": self.patch_size,
            "valid_fraction": float(valid_fraction),
            "bands": ["B2", "B3", "B4", "B8"],
            "region": self.region,
            "year": self.year,
        }
        with open(patch_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

