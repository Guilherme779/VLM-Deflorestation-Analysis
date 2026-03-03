from ExportConfig import ExportConfig
import ee


class SceneContext:
    """Holds everything about the chosen Sentinel‑2 scene.

    Responsible for:
    - Initializing the Earth Engine client using the values from ExportConfig
    - Picking a single S2 image (low clouds, enough area, in date range)
    - Deriving the geometry to export (full tile or a box around a point)
    - Exposing handy metadata such as product_id, scene_id, date_str, cloud_pct
    """

    def __init__(self, EConfig: ExportConfig) -> None:
        # Store the configuration object so the exporter can reuse it later.
        self.EConfig = EConfig
        # Will be filled in by build()
        self.point = None            # ee.Geometry.Point for the reference lon/lat
        self.s2_image = None         # ee.Image for the selected Sentinel‑2 scene
        self.region = None           # ee.Geometry used as export region
        self.proj = None             # ee.Projection taken from an S2 band
        self.product_id = None       # PRODUCT_ID metadata string
        self.scene_id = None         # SYSTEM_INDEX / system:index
        self.date_str = None         # Acquisition date as YYYYMMdd
        self.cloud_pct = None        # CLOUDY_PIXEL_PERCENTAGE

    def build(self) -> None:
        """Connect to EE, select the scene and derive region/projection/metadata."""
        self._init_ee()
        self.point = ee.Geometry.Point([self.EConfig.scene_lon, self.EConfig.scene_lat])
        self.s2_image = self._select_s2_image(self.point)
        self._fill_metadata()
        self.region = self._build_region(self.point, self.s2_image)
        self.proj = self._get_proj(self.s2_image)

    def _init_ee(self) -> None:
        """Initialize Earth Engine, prompting auth on first use if needed."""
        try:
            ee.Initialize(project=self.EConfig.project_id)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=self.EConfig.project_id)

    def _mask_s2_clouds(self, image: ee.Image) -> ee.Image:
        """Mask out cloudy pixels using the QA60 cloud bits."""
        qa = image.select("QA60")
        cloud_bit_mask = (1 << 10) | (1 << 11)
        mask = qa.bitwise_and(cloud_bit_mask).eq(0)
        return image.updateMask(mask)

    def _add_geom_area_km2(self, image: ee.Image) -> ee.Image:
        """Attach the geometry area (km²) as a property used for filtering."""
        area_km2 = image.geometry().area(ee.ErrorMargin(1)).divide(1e6)
        return image.set("geom_area_km2", area_km2)

    def _select_s2_image(self, point: ee.Geometry) -> ee.Image:
        """Pick the least‑cloudy S2 image that matches the config filters."""
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(self.EConfig.start_date, self.EConfig.end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.EConfig.max_cloud_percent))
        )
        # Require a minimum footprint area so we do not accidentally pick tiny geometries
        col = col.map(self._add_geom_area_km2).filter(
            ee.Filter.gte("geom_area_km2", self.EConfig.min_geom_area_km2)
        )
        # Cloud‑sorted collection; we keep the first one
        col = col.sort("CLOUDY_PIXEL_PERCENTAGE")
        img = col.first()
        if img is None:
            raise RuntimeError("No Sentinel-2 images found for given filters.")
        return self._mask_s2_clouds(ee.Image(img))

    def _build_region(self, point: ee.Geometry, s2_image: ee.Image):
        """Return the export region: full tile or a square buffer around the point."""
        if self.EConfig.use_full_tile:
            return s2_image.geometry()
        return point.buffer(float(self.EConfig.box_half_side_km) * 1000.0).bounds(
            ee.ErrorMargin(1)
        )

    def _get_proj(self, s2_image: ee.Image):
        """Use the B2 band projection as the reference for all exports."""
        return s2_image.select("B2").projection()

    def _fill_metadata(self) -> None:
        """Pull a few useful metadata fields from the chosen image for logging."""
        img = self.s2_image
        self.product_id = img.get("PRODUCT_ID").getInfo()
        self.cloud_pct = img.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
        self.scene_id = img.get("SYSTEM_INDEX").getInfo() or img.get(
            "system:index"
        ).getInfo()
        self.date_str = ee.Date(img.get("system:time_start")).format("YYYYMMdd").getInfo()