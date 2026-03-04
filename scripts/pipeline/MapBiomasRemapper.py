import json
from pathlib import Path

import numpy as np
import rasterio
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio


REPO_ROOT = Path(__file__).resolve().parents[2]


IGNORE = 255  # uint8 ignore label
FOREST = 0
PASTURE = 1
AGRICULTURE = 2
WATER = 3
URBAN = 4


# MapBiomas Collection 9 codes
MB_TO_5C = {
    3: FOREST,   # Forest Formation
    5: FOREST,   # Mangrove
    6: FOREST,   # Floodable Forest
    49: FOREST,  # Wooded Sandbank Vegetation
    15: PASTURE,  # Pasture
    18: AGRICULTURE,
    19: AGRICULTURE,
    39: AGRICULTURE,
    40: AGRICULTURE,
    41: AGRICULTURE,
    33: WATER,   # Water bodies
    24: URBAN,   # Urban area
}





@dataclass
class MapbiomasRemapper:
    """Remap MapBiomas rasters in each patch folder to 5 reduced classes."""

    patch_root: Path
    in_name: str = "mapbiomas.tif"
    out_name: str = "mapbiomas_5c.tif"
    write_summary_json: bool = True
    summary_json_name: str = "mapbiomas_5c_summary.json"

    def run(self) -> None:
        if not self.patch_root.exists():
            raise FileNotFoundError(f"PATCH_ROOT does not exist: {self.patch_root}")

        patch_dirs = self._iter_patch_dirs()
        print(f"Scanning: {self.patch_root}")
        print(f"Found {len(patch_dirs)} patch folders")

        global_counts, per_patch_counts = self._init_counts()
        written = 0
        skipped_missing = 0
        skipped_exists = 0

        for i, pdir in enumerate(patch_dirs, 1):
            status = self._process_single_patch(pdir, global_counts, per_patch_counts)
            if status == "missing":
                skipped_missing += 1
            elif status == "exists":
                skipped_exists += 1
            elif status == "written":
                written += 1

            if i % 200 == 0:
                print(f"Processed {i}/{len(patch_dirs)} folders...")

        self._report(global_counts, written, skipped_missing, skipped_exists)

        if self.write_summary_json:
            self._write_summary_json(global_counts, per_patch_counts)

    def _iter_patch_dirs(self) -> list[Path]:
        return sorted([p for p in self.patch_root.iterdir() if p.is_dir()])

    def _remap(self, arr: np.ndarray) -> np.ndarray:
        out = np.full(arr.shape, IGNORE, dtype=np.uint8)
        for mb_code, cls in MB_TO_5C.items():
            out[arr == mb_code] = np.uint8(cls)
        return out

    @staticmethod
    def _init_counts() -> tuple[dict[int, int], dict[str, dict[int, int]]]:
        global_counts = {FOREST: 0, PASTURE: 0, AGRICULTURE: 0, WATER: 0, URBAN: 0, IGNORE: 0}
        per_patch_counts: dict[str, dict[int, int]] = {}
        return global_counts, per_patch_counts

    def _process_single_patch(
        self,
        pdir: Path,
        global_counts: dict[int, int],
        per_patch_counts: dict[str, dict[int, int]],
    ) -> str:
        """Process one patch directory.

        Returns a status string: "missing", "exists", or "written".
        """

        in_path = pdir / self.in_name
        out_path = pdir / self.out_name

        if not in_path.exists():
            return "missing"

        if out_path.exists():
            return "exists"

        with rasterio.open(in_path) as src:
            mb = src.read(1)
            profile = src.profile.copy()

        mapped = self._remap(mb)

        vals, cnts = np.unique(mapped, return_counts=True)
        stats = {int(v): int(c) for v, c in zip(vals, cnts)}
        per_patch_counts[pdir.name] = stats

        for v, c in stats.items():
            global_counts[v] = global_counts.get(v, 0) + c

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=IGNORE,
            compress="lzw",
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mapped, 1)

        return "written"

    def _report(
        self,
        global_counts: dict[int, int],
        written: int,
        skipped_missing: int,
        skipped_exists: int,
    ) -> None:
        total_px = sum(global_counts.values())

        def pct(x: int) -> float:
            return 100.0 * x / total_px if total_px else 0.0

        print("\nDone.")
        print(f"Written: {written}")
        print(f"Skipped (missing {self.in_name}): {skipped_missing}")
        print(f"Skipped (already had {self.out_name}): {skipped_exists}")

        print("\nGlobal distribution:")
        print(f"  Forest      (0): {pct(global_counts[FOREST]):6.2f}%  ({global_counts[FOREST]})")
        print(f"  Pasture     (1): {pct(global_counts[PASTURE]):6.2f}%  ({global_counts[PASTURE]})")
        print(f"  Agriculture (2): {pct(global_counts[AGRICULTURE]):6.2f}%  ({global_counts[AGRICULTURE]})")
        print(f"  Water       (3): {pct(global_counts[WATER]):6.2f}%  ({global_counts[WATER]})")
        print(f"  Urban       (4): {pct(global_counts[URBAN]):6.2f}%  ({global_counts[URBAN]})")
        print(f"  Ignore    (255): {pct(global_counts[IGNORE]):6.2f}%  ({global_counts[IGNORE]})")

    def _write_summary_json(
        self,
        global_counts: dict[int, int],
        per_patch_counts: dict[str, dict[int, int]],
    ) -> None:
        out_json = self.patch_root / self.summary_json_name
        payload = {
            "in_name": self.in_name,
            "out_name": self.out_name,
            "ignore_value": IGNORE,
            "class_ids": {
                "forest": FOREST,
                "pasture": PASTURE,
                "agriculture": AGRICULTURE,
                "water": WATER,
                "urban": URBAN,
                "ignore": IGNORE,
            },
            "mapbiomas_codes_to_class": {str(k): int(v) for k, v in MB_TO_5C.items()},
            "global_counts": {str(k): int(v) for k, v in global_counts.items()},
            "per_patch_counts": per_patch_counts,
        }
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote summary JSON: {out_json}")
