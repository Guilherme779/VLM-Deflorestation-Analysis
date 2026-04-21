import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import rasterio


IGNORE = 255
CLASSES = {
    0: "forest",
    1: "pasture",
    2: "agriculture",
    3: "water",
    4: "urban",
}


@dataclass
class PatchIndexBuilder:
    """Build and maintain dataset_index.jsonl for a directory of patches.

    The index is the single source of truth for per-patch metadata.  All
    downstream stages (captions, QA, grounding) write back into it via
    :meth:`update_records` instead of managing their own JSONL I/O.
    """

    patch_root: Path
    img_name: str
    lbl_name: str
    region: str
    year: int
    out_jsonl: Path
    min_lbl_valid_frac: float = 0.80

    def run(self) -> None:
        patch_dirs = self._iter_patch_dirs()
        print(f"Found {len(patch_dirs)} patch folders")

        kept = 0
        dropped = 0
        global_counts_all = {k: 0 for k in CLASSES.keys()}
        global_counts_kept = {k: 0 for k in CLASSES.keys()}
        global_counts_dropped = {k: 0 for k in CLASSES.keys()}

        with self.out_jsonl.open("w", encoding="utf-8") as f:
            for i, pdir in enumerate(patch_dirs, 1):
                img_path, lbl_path = self._patch_paths(pdir)
                if not img_path.exists() or not lbl_path.exists():
                    dropped += 1
                    continue

                lbl = self._read_label(lbl_path)
                valid_frac, counts, fracs, dominant = self._summarize_label(lbl)

                self._accumulate_counts(global_counts_all, counts)

                if valid_frac < self.min_lbl_valid_frac:
                    dropped += 1
                    self._accumulate_counts(global_counts_dropped, counts)
                    continue

                self._accumulate_counts(global_counts_kept, counts)
                self._write_record(f, pdir, img_path, lbl_path, valid_frac, counts, fracs, dominant)
                kept += 1

                if i % 200 == 0:
                    print(f"Processed {i}/{len(patch_dirs)} | kept={kept} dropped={dropped}")

        self._print_distributions(global_counts_all, global_counts_kept, global_counts_dropped)

        print(f"\nWrote index: {self.out_jsonl}")
        print(f"Kept: {kept}")
        print(
            f"Dropped: {dropped} (missing files or low valid label fraction < {self.min_lbl_valid_frac})"
        )

    def _iter_patch_dirs(self) -> list[Path]:
        return [p for p in self.patch_root.iterdir() if p.is_dir()]

    def _patch_paths(self, pdir: Path) -> tuple[Path, Path]:
        return pdir / self.img_name, pdir / self.lbl_name

    def _read_label(self, lbl_path: Path) -> np.ndarray:
        with rasterio.open(lbl_path) as src:
            return src.read(1)

    def _accumulate_counts(self, acc: dict[int, int], counts: dict[int, int]) -> None:
        for k in counts:
            acc[k] += counts[k]

    def _summarize_label(self, lbl: np.ndarray):
        valid = lbl != IGNORE
        valid_count = int(valid.sum())
        total = int(lbl.size)
        valid_frac = valid_count / total if total else 0.0

        counts = {k: 0 for k in CLASSES.keys()}
        if valid_count > 0:
            vals, cnts = np.unique(lbl[valid], return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                if int(v) in counts:
                    counts[int(v)] = int(c)

        fracs = {CLASSES[k]: (counts[k] / valid_count if valid_count else 0.0) for k in counts}
        dominant = max(fracs.items(), key=lambda kv: kv[1])[0] if valid_count else "none"

        return valid_frac, counts, fracs, dominant

    def _write_record(
        self,
        f,
        pdir: Path,
        img_path: Path,
        lbl_path: Path,
        valid_frac: float,
        counts: dict[int, int],
        fracs: dict[str, float],
        dominant: str,
    ) -> None:
        rec = {
            "patch_id": pdir.name,
            "image_path": str(img_path.as_posix()),
            "label_path": str(lbl_path.as_posix()),
            "label_valid_fraction": valid_frac,
            "class_counts": {CLASSES[k]: counts[k] for k in counts},
            "class_fractions": fracs,
            "dominant_class": dominant,
            "region": self.region,
            "year": self.year,
        }
        f.write(json.dumps(rec) + "\n")

    def _print_distributions(
        self,
        global_counts_all: dict[int, int],
        global_counts_kept: dict[int, int],
        global_counts_dropped: dict[int, int],
    ) -> None:
        def print_dist(title: str, counts: dict[int, int]) -> None:
            total = sum(counts.values())
            print(f"\n{title}")
            for k in sorted(counts):
                name = CLASSES.get(k, str(k))
                frac = counts[k] / total if total > 0 else 0
                print(f"  {name:12s}: {counts[k]:10d}  ({frac*100:6.2f}%)")

        print_dist("All patches (kept + dropped)", global_counts_all)
        print_dist("Kept patches", global_counts_kept)
        print_dist("Dropped patches", global_counts_dropped)

    # ------------------------------------------------------------------ #
    #  Shared index I/O used by downstream stages                          #
    # ------------------------------------------------------------------ #

    def load_index(self) -> Dict[str, Dict]:
        """Load the JSONL index into a dict keyed by patch_id.

        Downstream stages call this to look up fractions without re-reading
        the label rasters.  Returns an empty dict if the index does not yet
        exist.
        """
        if not self.out_jsonl.exists():
            return {}
        records: Dict[str, Dict] = {}
        with self.out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                pid = str(rec.get("patch_id", ""))
                if pid:
                    records[pid] = rec
        return records

    def update_records(self, transform: Callable[[Dict], Dict]) -> None:
        """Apply *transform* to every record in the JSONL index and rewrite it.

        This is the single write-back hook used by captions, QA, and
        grounding stages so that they do not need to manage JSONL I/O
        themselves.  *transform* receives a record dict and must return the
        (possibly modified) dict.
        """
        if not self.out_jsonl.exists():
            raise FileNotFoundError(
                f"Index not found: {self.out_jsonl}. Run build_index() first."
            )

        tmp_path = self.out_jsonl.with_suffix(self.out_jsonl.suffix + ".tmp")
        written = 0
        with (
            self.out_jsonl.open("r", encoding="utf-8") as r,
            tmp_path.open("w", encoding="utf-8") as w,
        ):
            for line in r:
                raw = line.rstrip("\n")
                if not raw.strip():
                    w.write("\n")
                    continue
                rec = json.loads(raw)
                rec = transform(rec)
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

        tmp_path.replace(self.out_jsonl)
        return written
