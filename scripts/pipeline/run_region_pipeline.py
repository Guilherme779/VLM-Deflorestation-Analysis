from __future__ import annotations

"""CLI entrypoint to run the RegionPipeline for one or more regions.

Example:
  python scripts/pipeline/run_region_pipeline.py --region santarem --year 2021
  python scripts/pipeline/run_region_pipeline.py --regions santarem altamira --year 2021
"""

import argparse
from pathlib import Path
from typing import List

from RegionPipeline import RegionPipeline, find_repo_root, infer_regions_available, SceneConfig, PatchDatasetConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the high-level RegionPipeline.")
    parser.add_argument("--year", type=int, default=2021)

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--region", type=str, help="Single region (e.g. santarem)")
    group.add_argument("--regions", nargs="+", help="One or more regions")

    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-empty-fraction", type=float, default=0.95)

    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable patch augmentation step (rotate 90/180/270).",
    )

    parser.add_argument(
        "--augment-suffix",
        type=str,
        default="_rot90",
        help="Base suffix for rotated patches (default: _rot90).",
    )
    parser.add_argument(
        "--augment-overwrite",
        action="store_true",
        help="Overwrite existing rotated patch folders/files if present.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    repo_root = find_repo_root(Path(__file__))
    scenes_dir = repo_root / "data" / "scenes"
    available_regions = infer_regions_available(scenes_dir)

    if args.region:
        regions: List[str] = [args.region]
    elif args.regions:
        regions = list(args.regions)
    else:
        if not available_regions:
            raise SystemExit(
                "No regions found under data/scenes. Provide --region or create "
                "data/scenes/<region>_s2_<year>_original."
            )
        regions = available_regions

    for region in regions:
        print(f"\n=== RegionPipeline: {region} ({args.year}) ===")
        scene_cfg = SceneConfig(
            region=region,
            year=args.year,
            patch_size=args.patch_size,
            stride=args.stride,
            max_empty_fraction=args.max_empty_fraction,
        )

        patch_cfg = PatchDatasetConfig(
            ignore_label=255,
            min_label_valid_frac=0.80,
            classes=None,
        )

        pipeline = RegionPipeline(
            repo_root=repo_root,
            scene_config=scene_cfg,
            patch_config=patch_cfg,
            augment_rotations=(90, 180, 270),
            augment_suffix=str(args.augment_suffix),
            augment_overwrite=bool(args.augment_overwrite),
        )
        pipeline.run_all(with_augmentation=not bool(args.no_augment))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
