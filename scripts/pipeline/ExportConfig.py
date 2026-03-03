import argparse
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class ExportConfig:
    """High‑level configuration for picking a scene and exporting rasters.

    These values control:
    - When / where to look for Sentinel‑2 scenes (time window + lon/lat)
    - How strict the filters are (cloud percentage, minimum area)
    - How big the export region is (full tile vs buffered box)
    - How images are written to Drive (scale, CRS, folder and name prefix)
    - Which MapBiomas/PRODES assets and feature collections to use
    """

    # Time window and year tag used in names
    year: int = 2021
    start_date: str = "2021-06-01"
    end_date: str = "2021-09-30"

    # Reference point for selecting the scene / region
    scene_lon: float = -55.58055074791175
    scene_lat: float = -11.34892164907087

    # Geometry options
    use_full_tile: bool = True           # if True use full S2 footprint, otherwise a box
    box_half_side_km: float = 20.0       # half side of box, only used if use_full_tile is False

    # Image quality and footprint filters
    max_cloud_percent: int = 20          # maximum CLOUDY_PIXEL_PERCENTAGE
    min_geom_area_km2: float = 6000.0    # minimum footprint area for the image

    # Export options
    export_scale: int = 10               # output resolution in meters
    drive_folder: str = "sinop"          # Google Drive folder name
    crs: str = "EPSG:31980"             # target CRS for exports (optional)

    # Earth Engine assets / project
    project_id: str = "prodes-dataset"
    mapbiomas_asset: str = (
        "projects/mapbiomas-public/assets/brazil/lulc/collection9/"
        "mapbiomas_collection90_integration_v1"
    )
    prodes_yearly_fc: str = "users/guilhermeteix2016/prodes_yearly_deforestation"
    prodes_no_forest_fc: str = "users/guilhermeteix2016/prodes_no_forest"
    prodes_residual_fc: str = "users/guilhermeteix2016/prodes_residual"
    prodes_hydro_fc: str = "users/guilhermeteix2016/prodes_hydrography"

    # Naming helper for label exports
    prefix: str = "sinop"

    def validate(self) -> None:
        """Basic sanity checks on the configuration values."""
        assert self.start_date <= self.end_date, "start_date must be <= end_date"
        assert self.max_cloud_percent >= 0, "max_cloud_percent must be non-negative"
        assert self.min_geom_area_km2 > 0, "min_geom_area_km2 must be positive"
        assert self.export_scale > 0, "export_scale must be positive"