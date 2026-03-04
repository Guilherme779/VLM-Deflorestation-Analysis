from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


@dataclass
class PatchLabeler:
    """Attach MapBiomas/PRODES labels to each patch directory."""

    patch_root: Path
    mapbiomas_path: Path
    prodes_defor_path: Path
    prodes_noforest_path: Path
    prodes_resid_path: Path
    prodes_hydro_path: Path

    def run(self) -> None:
        patch_dirs = self._iter_patch_dirs()
        print(f"Found {len(patch_dirs)} patch folders. Adding labels...")

        mb_ds, def_ds, nf_ds, res_ds, hydro_ds = self._open_label_datasets()

        try:
            for patch_dir in patch_dirs:
                ref_patch_path = patch_dir / "s2_rgbnir.tif"
                if not self._is_valid_patch_dir(patch_dir, ref_patch_path):
                    continue

                with rasterio.open(ref_patch_path) as ref_ds:
                    ref_profile = ref_ds.profile.copy()
                    ref_profile.update(count=1)

                    self._write_all_labels(
                        patch_dir,
                        ref_ds,
                        ref_profile,
                        mb_ds,
                        def_ds,
                        nf_ds,
                        res_ds,
                        hydro_ds,
                    )

            print("Done. Labels added to all patches.")
        finally:
            mb_ds.close()
            def_ds.close()
            nf_ds.close()
            res_ds.close()
            hydro_ds.close()

    def _iter_patch_dirs(self) -> list[Path]:
        return [p for p in self.patch_root.iterdir() if p.is_dir()]

    def _open_label_datasets(self):
        mb_ds = self._load_label_dataset(self.mapbiomas_path, "MapBiomas")
        def_ds = self._load_label_dataset(self.prodes_defor_path, "PRODES_deforestation")
        nf_ds = self._load_label_dataset(self.prodes_noforest_path, "PRODES_no_forest")
        res_ds = self._load_label_dataset(self.prodes_resid_path, "PRODES_residual")
        hydro_ds = self._load_label_dataset(self.prodes_hydro_path, "PRODES_hydrography")
        return mb_ds, def_ds, nf_ds, res_ds, hydro_ds

    def _is_valid_patch_dir(self, patch_dir: Path, ref_patch_path: Path) -> bool:
        meta_path = patch_dir / "meta.json"
        if not meta_path.exists():
            print(f"Skipping {patch_dir.name}: no meta.json")
            return False

        if not ref_patch_path.exists():
            print(f"Skipping {patch_dir.name}: missing s2_rgbnir.tif")
            return False

        return True

    def _write_all_labels(
        self,
        patch_dir: Path,
        ref_ds,
        ref_profile: dict,
        mb_ds,
        def_ds,
        nf_ds,
        res_ds,
        hydro_ds,
    ) -> None:
        self._write_single_label("mapbiomas", mb_ds, ref_ds, ref_profile, patch_dir)
        self._write_single_label("prodes_deforestation", def_ds, ref_ds, ref_profile, patch_dir)
        self._write_single_label("prodes_no_forest", nf_ds, ref_ds, ref_profile, patch_dir)
        self._write_single_label("prodes_residual", res_ds, ref_ds, ref_profile, patch_dir)
        self._write_single_label("prodes_hydrography", hydro_ds, ref_ds, ref_profile, patch_dir)

    def _write_single_label(
        self,
        name: str,
        src_ds,
        ref_ds,
        ref_profile: dict,
        patch_dir: Path,
    ) -> None:
        arr, nodata_val = self._reproject_band_to_match(
            src_ds,
            ref_ds,
            band_index=1,
            resampling=Resampling.nearest,
        )
        out_path = patch_dir / f"{name}.tif"
        prof = ref_profile.copy()
        prof["dtype"] = src_ds.dtypes[0]
        if nodata_val is not None:
            prof["nodata"] = nodata_val
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(arr.astype(prof["dtype"], copy=False), 1)

    def _reproject_band_to_match(
        self,
        src_ds,
        ref_ds,
        *,
        band_index: int = 1,
        resampling=Resampling.nearest,
    ):
        """Reproject a single band from src_ds into ref_ds grid (CRS/transform/shape)."""

        dst = np.empty((ref_ds.height, ref_ds.width), dtype=src_ds.dtypes[band_index - 1])
        reproject(
            source=rasterio.band(src_ds, band_index),
            destination=dst,
            src_transform=src_ds.transform,
            src_crs=src_ds.crs,
            src_nodata=src_ds.nodata,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            dst_nodata=src_ds.nodata,
            resampling=resampling,
        )
        return dst, src_ds.nodata

    @staticmethod
    def _load_label_dataset(label_path: Path, name: str):
        """Open a label raster for later per-patch reprojection."""

        ds = rasterio.open(label_path)
        if ds.count != 1:
            raise ValueError(f"Expected single-band raster for {name}, got {ds.count}: {label_path}")
        return ds
