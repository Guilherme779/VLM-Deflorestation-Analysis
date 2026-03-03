import argparse
from typing import Callable, List, Tuple

from ExportConfig import ExportConfig
from SceneContext import SceneContext
import ee


class Exporter:
    """Small helper that wires SceneContext into a set of export "jobs".

    Usage pattern:
      1) Create an Exporter with a ready‑built SceneContext
      2) Register one or more named jobs (scene RGB, MapBiomas labels, PRODES masks)
      3) Call run() to enqueue all exports to Google Drive
    """

    def __init__(self, SContext: SceneContext) -> None:
        # Keep a reference to the scene + configuration for all jobs
        self.SContext = SContext
        # List of (export_name, builder_fn) pairs
        self.jobs: List[Tuple[str, Callable[[SceneContext], ee.Image]]] = []

    def add_job(self, name: str, builder: Callable[[SceneContext], ee.Image]) -> None:
        """Register a new export job.

        The builder receives the SceneContext and must return an ee.Image that
        will later be reprojected, clipped and sent to Drive.
        """
        self.jobs.append((name, builder))

    def run(self) -> None:
        """Build and enqueue all configured export jobs."""
        for name, builder in self.jobs:
            img = builder(self.SContext)
            # Ensure a consistent projection/extent for every export
            img = img.reproject(self.SContext.proj).clip(self.SContext.region)
            self._export_to_drive(img, name)

    def _export_to_drive(self, image: ee.Image, name: str) -> None:
        """Send an ee.Image to Google Drive using the parameters from ExportConfig."""
        kwargs = {
            "image": image,
            "description": name,
            "folder": self.SContext.EConfig.drive_folder,
            "fileNamePrefix": name,
            "region": self.SContext.region,
            "scale": self.SContext.EConfig.export_scale,
            "maxPixels": 1e13,
        }
        if self.SContext.EConfig.crs:
            kwargs["crs"] = self.SContext.EConfig.crs
        task = ee.batch.Export.image.toDrive(**kwargs)
        task.start()
        print(f"[EXPORT] Started {name}")

    # --- Job builders ---

    def build_scene_image(self, SContext: SceneContext) -> ee.Image:
        """Return the 4‑band RGB+NIR Sentinel‑2 image for export."""
        return SContext.s2_image.select(["B2", "B3", "B4", "B8"])

    def _rasterize_fc(self, fc: ee.FeatureCollection, value: int, band_name: str) -> ee.Image:
        """Burn a FeatureCollection into a single‑band uint8 raster with a fixed value."""
        img = ee.Image(0).byte().paint(fc, value).rename(band_name)
        return img

    def build_mapbiomas_image(self, SContext: SceneContext) -> ee.Image:
        """Load the MapBiomas land‑cover band for the configured year."""
        img = ee.Image(self.SContext.EConfig.mapbiomas_asset).select(f"classification_{self.SContext.EConfig.year}")
        return img

    def build_prodes_deforestation_image(self, SContext: SceneContext) -> ee.Image:
        """Rasterize yearly PRODES deforestation polygons inside the scene region."""
        fc = (
            ee.FeatureCollection(self.SContext.EConfig.prodes_yearly_fc)
            .filter(ee.Filter.eq("year", self.SContext.EConfig.year))
            .filterBounds(SContext.region)
        )
        return self._rasterize_fc(fc, 1, "deforestation")

    def build_prodes_no_forest_image(self, SContext: SceneContext) -> ee.Image:
        """Rasterize PRODES "no forest" mask within the scene region."""
        fc = ee.FeatureCollection(self.SContext.EConfig.prodes_no_forest_fc).filterBounds(SContext.region)
        return self._rasterize_fc(fc, 1, "no_forest")

    def build_prodes_residual_image(self, SContext: SceneContext) -> ee.Image:
        """Rasterize PRODES residual (uncertain/remaining) polygons."""
        fc = ee.FeatureCollection(self.SContext.EConfig.prodes_residual_fc).filterBounds(SContext.region)
        return self._rasterize_fc(fc, 1, "residual")

    def build_prodes_hydro_image(self, SContext: SceneContext) -> ee.Image:
        """Rasterize PRODES hydrography layer (rivers, water bodies)."""
        fc = ee.FeatureCollection(self.SContext.EConfig.prodes_hydro_fc).filterBounds(SContext.region)
        return self._rasterize_fc(fc, 1, "hydrography")


def parse_args() -> argparse.Namespace:
    """CLI wrapper: expose ExportConfig fields as command‑line options."""
    parser = argparse.ArgumentParser(description="Unified exporter for S2 scene and labels.")
    parser.add_argument("--mode", choices=["scene", "labels", "all"], default="all")
    parser.add_argument("--year", type=int, default=ExportConfig.year)
    parser.add_argument("--start-date", type=str, default=ExportConfig.start_date)
    parser.add_argument("--end-date", type=str, default=ExportConfig.end_date)
    parser.add_argument("--use-full-tile", action="store_true", default=ExportConfig.use_full_tile)
    parser.add_argument("--box-half-side-km", type=float, default=ExportConfig.box_half_side_km)
    parser.add_argument("--max-cloud-percent", type=int, default=ExportConfig.max_cloud_percent)
    parser.add_argument("--min-geom-area-km2", type=float, default=ExportConfig.min_geom_area_km2)
    parser.add_argument("--export-scale", type=int, default=ExportConfig.export_scale)
    parser.add_argument("--drive-folder", type=str, default=ExportConfig.drive_folder)
    parser.add_argument("--crs", type=str, default=ExportConfig.crs)
    parser.add_argument("--prefix", type=str, default=ExportConfig.prefix)
    return parser.parse_args()


def main() -> None:
    """Entry point when running the exporter as a script."""
    args = parse_args()

    # Build and validate the high‑level configuration from CLI args
    EConfig = ExportConfig(
        year=args.year,
        start_date=args.start_date,
        end_date=args.end_date,
        use_full_tile=args.use_full_tile,
        box_half_side_km=args.box_half_side_km,
        max_cloud_percent=args.max_cloud_percent,
        min_geom_area_km2=args.min_geom_area_km2,
        export_scale=args.export_scale,
        drive_folder=args.drive_folder,
        crs=args.crs,
        prefix=args.prefix,
    )
    EConfig.validate()

    # Build the EE scene context (select image, region, metadata)
    SContext = SceneContext(EConfig)
    SContext.build()

    print("Chosen image PRODUCT_ID:", SContext.product_id)
    print("Cloud percentage:", SContext.cloud_pct)
    print("Scene ID:", SContext.scene_id)
    print("Date:", SContext.date_str)

    exporter = Exporter(SContext)

    # Register the scene RGB export, if requested
    if args.mode in ("scene", "all"):
        scene_name = f"S2_RGBNIR_{SContext.EConfig.year}_{SContext.scene_id}_{SContext.date_str}"
        exporter.add_job(scene_name, exporter.build_scene_image)

    # Register label exports, if requested
    if args.mode in ("labels", "all"):
        prefix = SContext.EConfig.prefix
        exporter.add_job(f"MapBiomas_{SContext.EConfig.year}_{prefix}", exporter.build_mapbiomas_image)
        exporter.add_job(f"PRODES_deforestation_{SContext.EConfig.year}_{prefix}", exporter.build_prodes_deforestation_image)
        exporter.add_job(f"PRODES_no_forest_{prefix}", exporter.build_prodes_no_forest_image)
        exporter.add_job(f"PRODES_residual_{prefix}", exporter.build_prodes_residual_image)
        exporter.add_job(f"PRODES_hydrography_{SContext.EConfig.year}_{prefix}", exporter.build_prodes_hydro_image)

    exporter.run()


if __name__ == "__main__":
    main()
