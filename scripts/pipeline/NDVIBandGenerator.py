"""Add NDVI (Normalized Difference Vegetation Index) bands to all patches.

This step computes NDVI from the existing NIR (B8) and RED (B4) bands and appends
it as a 5th band to each patch's s2_rgbnir.tif file.

NDVI = (NIR - RED) / (NIR + RED)

Typical usage (standalone):
  python scripts/pipeline/ndvi_band_generator.py \
    --patch-root data/tiles/santarem_s2_2021_patches

Pipeline runner calls this via scripts/pipeline/run_region_pipeline.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.io import MemoryFile


class NDVIBandGenerator:
    """Compute and append NDVI bands to all patches in a dataset."""

    def __init__(self, *, patch_root: Path) -> None:
        self.patch_root = patch_root.resolve()
        self.img_name = "s2_rgbnir.tif"

    def run(self) -> None:
        """Process all patch directories and add NDVI bands."""
        patch_dirs = sorted([p for p in self.patch_root.iterdir() if p.is_dir()])
        
        processed = 0
        skipped = 0
        
        for patch_dir in patch_dirs:
            img_path = patch_dir / self.img_name
            
            if not img_path.exists():
                skipped += 1
                continue
            
            try:
                self._add_ndvi_band(img_path)
                processed += 1
            except Exception as e:
                print(f"  [WARN] Failed to process {patch_dir.name}: {e}")
                skipped += 1
        
        print(f"Patch root: {self.patch_root}")
        print(f"Processed patches: {processed}")
        print(f"Skipped patches: {skipped}")

    def _add_ndvi_band(self, img_path: Path) -> None:
        """Read s2_rgbnir.tif, compute NDVI, and append as 5th band.
        
        NDVI is stored as int16 with scaling [-10000, 10000] to match the 
        reflectance value convention (reflectance × 10000).
        """
        with rasterio.open(img_path) as src:
            if src.count != 4:
                raise ValueError(f"Expected 4 bands, got {src.count}")
            
            # Band ordering in s2_rgbnir.tif: B2 (blue), B3 (green), B4 (red), B8 (nir)
            # Indices: 1 (B2), 2 (B3), 3 (B4), 4 (B8)
            red_band = src.read(3)      # B4 at index 2 (read uses 1-based indexing, so band 3)
            nir_band = src.read(4)      # B8 at index 3 (read uses 1-based indexing, so band 4)
            
            # Calculate NDVI with float32 precision to avoid overflow
            red_f = red_band.astype(np.float32)
            nir_f = nir_band.astype(np.float32)
            
            denominator = nir_f + red_f
            # Avoid division by zero
            ndvi = np.zeros_like(denominator, dtype=np.float32)
            mask = denominator != 0
            ndvi[mask] = (nir_f[mask] - red_f[mask]) / denominator[mask]
            
            # Scale NDVI to int16 range [-10000, 10000] to match reflectance scaling
            # NDVI ranges from -1 to 1, so we map [-1, 1] -> [-10000, 10000]
            ndvi_int16 = (ndvi * 10000).astype(np.int16)
            
            # Read all existing bands
            profile = src.profile.copy()
            all_bands = src.read()
            
        # Create new dataset with 5 bands, updating dtype to int16
        profile.update(count=5, dtype=rasterio.int16)
        
        with rasterio.open(img_path, 'w', **profile) as dst:
            # Write original 4 bands
            for i in range(1, 5):
                dst.write(all_bands[i - 1], i)
            # Write NDVI as 5th band
            dst.write(ndvi_int16, 5)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Add NDVI bands to all patches in a dataset."
    )
    parser.add_argument(
        "--patch-root",
        type=Path,
        required=True,
        help="Root directory containing patch folders (e.g., data/tiles/santarem_s2_2021_patches)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for standalone execution."""
    args = parse_args(argv)
    
    if not args.patch_root.exists():
        print(f"Error: Patch root directory not found: {args.patch_root}")
        return 1
    
    generator = NDVIBandGenerator(patch_root=args.patch_root)
    generator.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
