from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from SceneTiler import SceneTiler
from PatchLabeler import PatchLabeler
from MapBiomasRemapper import MapbiomasRemapper
from PatchAugmenter import PatchAugmenter
from PatchIndexBuilder import PatchIndexBuilder
from PatchCaptionGenerator import PatchCaptionGenerator
from PatchQAGenerator import PatchQAGenerator
from GroundingAnswerBuilder import GroundingStage, GroundingConfig

@dataclass(frozen=True)
class SceneConfig:
    """Configuration for a single remote-sensing scene.

    This holds semantic information about the scene itself and how we want
    to tile it, but is intentionally path-agnostic. Concrete filesystem
    locations are handled by ``RegionPaths``.
    """

    region: str
    year: int
    patch_size: int
    stride: int
    max_empty_fraction: float


@dataclass(frozen=True)
class PatchDatasetConfig:
    """Configuration for the derived patch dataset.

    ``RegionPaths`` knows *where* things live on disk; this specifies
    logical properties of the patch dataset and labels.
    """

    ignore_label: int = 255
    min_label_valid_frac: float = 0.80
    classes: Dict[int, str] | None = None


@dataclass(frozen=True)
class RegionPaths:
    """Resolved paths for a (region, year) pair."""

    region: str
    year: int

    scene_dir: Path
    tiles_dir: Path

    s2_tif: Path
    mapbiomas_tif: Path
    prodes_defor_tif: Path
    prodes_no_forest_tif: Path
    prodes_residual_tif: Path
    prodes_hydro_tif: Path


def find_repo_root(start: Path) -> Path:
    """Best-effort repo root finder used by pipeline scripts.

    Looks for a directory containing either ``pyproject.toml`` or ``.git``.
    Falls back to two levels up from the provided path.
    """

    cur = start.resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve().parents[2]


def infer_regions_available(scenes_dir: Path) -> list[str]:
    """Infer available regions from data/scenes/*_s2_<year>_original folders."""

    if not scenes_dir.exists():
        return []

    regions: set[str] = set()
    for p in scenes_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.endswith("_original") and "_s2_" in name:
            regions.add(name.split("_s2_")[0])
    return sorted(regions)


def region_paths(repo_root: Path, region: str, year: int) -> RegionPaths:
    """Resolve standard scene/tile and label paths for a region/year."""

    scenes_dir = repo_root / "data" / "scenes"
    tiles_dir = repo_root / "data" / "tiles"

    scene_dir = scenes_dir / f"{region}_s2_{year}_original"
    tiles_root = tiles_dir / f"{region}_s2_{year}_patches"

    s2_tif_prefix = scene_dir / f"S2_RGBNIR_{year}_None_"  # prefix only
    candidates = sorted(scene_dir.glob(f"S2_RGBNIR_{year}_*.tif"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No Sentinel-2 tif found for region={region} year={year} in: {scene_dir}\n"
            f"Expected something like: S2_RGBNIR_{year}_*.tif"
        )
    if len(candidates) > 1:
        raise FileExistsError(
            f"Found multiple Sentinel-2 tifs in {scene_dir}. Extend RegionPipeline if you need to disambiguate."
        )
    s2_tif = candidates[0]

    mapbiomas_tif = scene_dir / f"MapBiomas_{year}_{region}_scene.tif"
    prodes_defor_tif = scene_dir / f"PRODES_deforestation_{year}_{region}_scene.tif"
    prodes_hydro_tif = scene_dir / f"PRODES_hydrography_{year}_{region}_scene.tif"
    prodes_no_forest_tif = scene_dir / f"PRODES_no_forest_{region}_scene.tif"
    prodes_residual_tif = scene_dir / f"PRODES_residual_{region}_scene.tif"

    return RegionPaths(
        region=region,
        year=year,
        scene_dir=scene_dir,
        tiles_dir=tiles_root,
        s2_tif=s2_tif,
        mapbiomas_tif=mapbiomas_tif,
        prodes_defor_tif=prodes_defor_tif,
        prodes_no_forest_tif=prodes_no_forest_tif,
        prodes_residual_tif=prodes_residual_tif,
        prodes_hydro_tif=prodes_hydro_tif,
    )


@dataclass
class RegionPipeline:
    """High-level pipeline for one (region, year) dataset.

    Steps (in order):
      1. Tile scene
      2. Add raw labels
      3. Remap labels
      4. Augment patches (rotations)
      5. Build index (with geo fields)
      6. Generate captions
      7. Generate QA
      8. Generate grounding

    This class is intentionally thin: it just wires together existing
    step-specific classes/commands in scripts/pipeline.
    """

    repo_root: Path
    scene_config: SceneConfig
    patch_config: PatchDatasetConfig
    augment_rotations: Iterable[int] = (90, 180, 270)
    augment_suffix: str = "_rot90"
    augment_overwrite: bool = False

    def _paths(self) -> RegionPaths:
        return region_paths(self.repo_root, self.scene_config.region, self.scene_config.year)

    # ---- individual stages ----

    def tile_scene(self) -> None:
        sc = self.scene_config
        tiler = SceneTiler(
            scene_path=self.region_path.s2_tif,
            out_dir=self.region_path.tiles_dir,
            region=sc.region,
            year=sc.year,
            patch_size=sc.patch_size,
            stride=sc.stride,
            max_empty_fraction=sc.max_empty_fraction,
        )
        tiler.run()

    def add_raw_labels(self) -> None:
        labeler = PatchLabeler(
            patch_root=self.region_path.tiles_dir,
            mapbiomas_path=self.region_path.mapbiomas_tif,
            prodes_defor_path=self.region_path.prodes_defor_tif,
            prodes_noforest_path=self.region_path.prodes_no_forest_tif,
            prodes_resid_path=self.region_path.prodes_residual_tif,
            prodes_hydro_path=self.region_path.prodes_hydro_tif,
        )
        labeler.run()

    def remap_labels(self) -> None:
        remapper = MapbiomasRemapper(patch_root=self.region_path.tiles_dir)
        remapper.run()

    def augment_patches(self) -> None:

        augmenter = PatchAugmenter(
            patch_root=self.region_path.tiles_dir,
            rot_suffix=self.augment_suffix,
            angles=list(self.augment_rotations),
            overwrite=self.augment_overwrite,
            update_splits=True,
        )
        augmenter.run()

    def build_index(self) -> PatchIndexBuilder:
        index_builder = PatchIndexBuilder(
            patch_root=self.region_path.tiles_dir,
            img_name="s2_rgbnir.tif",
            lbl_name="mapbiomas_5c.tif",
            region=self.scene_config.region,
            year=self.scene_config.year,
            out_jsonl=self.region_path.tiles_dir / "dataset_index.jsonl",
        )
        index_builder.run()
        return index_builder

    def generate_captions(self, *, index_builder: PatchIndexBuilder) -> None:
        generator = PatchCaptionGenerator(
            patch_root=self.region_path.tiles_dir,
            index_builder=index_builder,
        )
        generator.run()

    def generate_qa(self, *, index_builder: PatchIndexBuilder) -> None:
        gen = PatchQAGenerator(
            patch_root=self.region_path.tiles_dir,
            index_builder=index_builder,
            dry_run=False,
            limit=None,
        )
        gen.run()

    def generate_grounding(self, *, index_builder: PatchIndexBuilder) -> None:
        stage = GroundingStage(
            patch_root=self.region_path.tiles_dir,
            index_builder=index_builder,
            repo_root=self.repo_root,
            config=GroundingConfig(
                top_k=3,
                min_area_px=16,
                min_area_frac=0.001,
                conversion_fraction_threshold=0.01,
                overwrite=True,
                dry_run=False,
            ),
        )
        stage.run()

    # ---- composite runners ----

    def run_all(self, *, with_augmentation: bool = True) -> None:
        """Run the full pipeline chain for this region/year."""

        self.region_path = self._paths()
        # self.tile_scene()
        # self.add_raw_labels()
        # self.remap_labels()
        # if with_augmentation:
        #     self.augment_patches()
        index_builder = self.build_index()
        self.generate_captions(index_builder=index_builder)
        self.generate_qa(index_builder=index_builder)
        self.generate_grounding(index_builder=index_builder)
