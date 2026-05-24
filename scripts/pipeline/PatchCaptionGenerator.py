"""Generate structured captions for patches based on MapBiomas labels.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import rasterio
from scipy.ndimage import label as _ndlabel
from MapBiomasClasses import MapBiomasClasses
from PatchIndexBuilder import PatchIndexBuilder

@dataclass
class SpatialStats:
    pixel_count: int
    centroid_y: float
    centroid_x: float
    locations: List[str]
    fragment_count: int = 0
    largest_fragment_fraction: float = 0.0
    scattered_locations: List[str] = field(default_factory=list)

class PatchCaptionGenerator:
    """Generate and attach captions to patches.
    """

    def __init__(
        self,
        *,
        patch_root: Path,
        index_builder: PatchIndexBuilder,
    ) -> None:
        self.patch_root = patch_root
        self.index_builder = index_builder

    @staticmethod
    def _load_label(label_path: Path) -> np.ndarray:
        with rasterio.open(label_path) as src:
            return src.read(1)

    @staticmethod
    def _percent(class_fraction: float) -> int:
        return int(round(class_fraction * 100.0))

    @staticmethod
    def _detect_roads(
        img_path: Path,
        min_fraction: float = 0.003,
        max_fraction: float = 0.08,
        min_aspect: float = 4.0,
        max_blob_frac: float = 0.02,
    ) -> bool:
        """Detect road presence via per-band brightness and linearity analysis.

        Roads in Amazonian Sentinel-2 imagery appear as very bright, thin,
        elongated features in all visible bands. Clouds are also very bright
        but form large, compact blobs.

        After finding bright pixels (top 1% in all RGB bands simultaneously),
        we inspect each connected component:
          - Skip tiny noise (< 15 px)
          - Skip large blobs (> 2% of image pixels) — these are clouds
          - If any remaining component has aspect ratio >= 4 → road detected
        """
        with rasterio.open(img_path) as src:
            rgb = src.read([1, 2, 3]).astype(np.float32)
        thresholds = np.array([np.percentile(rgb[i], 99) for i in range(3)])
        bright = np.all(rgb > thresholds[:, None, None], axis=0)
        frac = float(bright.mean())
        if not (min_fraction <= frac <= max_fraction):
            return False

        # Shape filter: require at least one thin, elongated bright component.
        labeled, n_comp = _ndlabel(bright)
        total_pixels = bright.size
        for comp_id in range(1, n_comp + 1):
            comp = labeled == comp_id
            size = int(comp.sum())
            if size < 15:
                continue
            if size > total_pixels * max_blob_frac:
                continue
            ys, xs = np.where(comp)
            h = int(ys.max() - ys.min() + 1)
            w = int(xs.max() - xs.min() + 1)
            aspect = max(h, w) / max(min(h, w), 1)
            if aspect >= min_aspect:
                return True
        return False

    @staticmethod
    def _count_components(mask: np.ndarray, min_size: int = 9) -> Tuple[int, float]:
        """Connected component analysis; returns (num_components, largest_component_fraction).

        Fragments smaller than min_size pixels are treated as noise and ignored.
        """
        labeled, num = _ndlabel(mask)
        if num == 0:
            return 0, 0.0
        total_pixels = int(mask.sum())
        sizes = sorted(
            [int((labeled == i).sum()) for i in range(1, num + 1) if int((labeled == i).sum()) >= min_size],
            reverse=True,
        )
        if not sizes:
            return 1, 1.0
        largest_frac = sizes[0] / total_pixels if total_pixels > 0 else 1.0
        return len(sizes), largest_frac

    def analyze_class_spatial(
        self,
        mask: np.ndarray,
    ) -> Optional[SpatialStats]:
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            return None

        H, W = mask.shape

        ys, xs = np.nonzero(mask)
        centroid_y = float(ys.mean()) / max(H - 1, 1)
        centroid_x = float(xs.mean()) / max(W - 1, 1)

        # 3x3 grid: cell index = row*3 + col  (row 0=top, col 0=left)
        ys_2d = np.arange(H)[:, None]
        xs_2d = np.arange(W)[None, :]
        y_cells = np.clip(ys_2d * 3 // max(H, 1), 0, 2)
        x_cells = np.clip(xs_2d * 3 // max(W, 1), 0, 2)
        cell_map = (y_cells * 3 + x_cells).astype(np.int32)

        region_totals = np.bincount(cell_map.ravel(), minlength=9)
        class_in_region = np.bincount(cell_map[mask].ravel(), minlength=9)

        # Only include cells where the class covers >= 2.5% of that cell's pixels
        MIN_REGION_FRACTION = 0.025
        present_cells = [
            i for i in range(9)
            if region_totals[i] > 0 and (class_in_region[i] / region_totals[i]) >= MIN_REGION_FRACTION
        ]

        # Greedily pick dominant cells until 65% of class pixels are covered
        COVERAGE_TARGET = 0.65
        present_by_count = sorted(present_cells, key=lambda i: class_in_region[i], reverse=True)
        cumulative = 0.0
        dominant_cells: List[int] = []
        for i in present_by_count:
            dominant_cells.append(i)
            cumulative += class_in_region[i] / pixel_count
            if cumulative >= COVERAGE_TARGET:
                break

        dominant_set = set(dominant_cells)

        # Drop scattered cells adjacent to a dominant one — likely spillover.
        # Adjacency = edge-sharing only (no diagonals).
        def _neighbors(idx: int) -> set:
            r, c = divmod(idx, 3)
            return {
                (r + dr) * 3 + (c + dc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= r + dr < 3 and 0 <= c + dc < 3
            }

        scattered_cells = [
            i for i in present_by_count
            if i not in dominant_set and not (_neighbors(i) & dominant_set)
        ]

        locations = self._compress_locations([(i // 3, i % 3) for i in dominant_cells])
        scattered_locations = self._compress_locations([(i // 3, i % 3) for i in scattered_cells])

        fragment_count, largest_fragment_fraction = self._count_components(mask)

        return SpatialStats(
            pixel_count=pixel_count,
            centroid_y=float(centroid_y),
            centroid_x=float(centroid_x),
            locations=locations,
            fragment_count=fragment_count,
            largest_fragment_fraction=largest_fragment_fraction,
            scattered_locations=scattered_locations,
        )

    def compute_stats_and_phrase(
        self,
        mask: np.ndarray,
    ) -> Tuple[Optional[Dict[str, float | int | List[str]]], Optional[str]]:
        stats = self.analyze_class_spatial(mask)
        if stats is None:
            return None, None
        phrase = None
        stats_dict: Dict[str, float | int | List[str]] = {
            "pixel_count": int(stats.pixel_count),
            "centroid_y": float(stats.centroid_y),
            "centroid_x": float(stats.centroid_x),
            "locations": list(stats.locations),
            "scattered_locations": list(stats.scattered_locations),
            "fragment_count": int(stats.fragment_count),
            "largest_fragment_fraction": float(stats.largest_fragment_fraction),
        }
        return stats_dict, phrase

    def compute_spatial_stats(
        self,
        lbl: np.ndarray,
    ) -> Dict[str, Optional[Dict[str, float | int | List[str]]]]:
        spatial_stats: Dict[str, Optional[Dict[str, float | int | List[str]]]] = {}
        valid_mask = (lbl != MapBiomasClasses.IGNORE)

        for idx, class_name in enumerate(MapBiomasClasses.CLASS_NAMES):
            mask = (lbl == idx) & valid_mask
            stats_dict, _ = self.compute_stats_and_phrase(mask)
            if stats_dict is not None:
                spatial_stats[class_name] = stats_dict

        return spatial_stats

    def build_caption(
        self,
        class_fractions: Dict[str, float],
        spatial_stats: Dict[str, Optional[Dict[str, float | int | List[str]]]],
    ) -> Tuple[str, Dict[str, float]]:
        ordered = sorted(class_fractions.items(), key=lambda kv: kv[1], reverse=True)
        classes_to_mention = self._filter_low_presence_classes(ordered)

        class_info = [
            f"{MapBiomasClasses.CLASS_LABELS.get(class_name, class_name)} (about {self._percent(class_fraction)}%)"
            for class_name, class_fraction in classes_to_mention
        ]
        composition = f"Image composed by {self._format_list(class_info)}."

        sentences: List[str] = []
        for class_name, class_fraction in classes_to_mention:
            stats = spatial_stats.get(class_name) or {}
            label = MapBiomasClasses.CLASS_LABELS.get(class_name, class_name)
            dom_locs = list(stats.get("locations") or [])
            scat_locs = list(stats.get("scattered_locations") or [])
            sentence = self._class_sentence(label, class_fraction, dom_locs, scat_locs)
            if sentence:
                sentences.append(sentence)

        caption = " ".join([composition] + sentences)
        return caption, {k: float(v) for k, v in class_fractions.items()}

    @staticmethod
    def _filter_low_presence_classes(
        ordered: List[Tuple[str, float]],
    ) ->  List[Tuple[str, float]]:
        #classes below 5% are usually not worth mentioning, unless they're water or urban (which are more salient even at low presence)
        return [
            (name, frac)
            for name, frac in ordered
            if frac >= 0.05 or (name in ("water", "urban") and frac > 0.0)
        ]

    @staticmethod
    def _class_sentence(
        class_label: str,
        class_fraction: float,
        dominant_locs: List[str],
        scattered_locs: List[str],
    ) -> str:
        label = class_label.capitalize()

        if class_fraction > 0.80:
            return f"{label} covers the whole image."

        all_locs = dominant_locs + [l for l in scattered_locs if l not in dominant_locs]

        if class_fraction >= 0.20:
            if not dominant_locs:
                return ""
            dom = PatchCaptionGenerator._format_list(dominant_locs)
            if not scattered_locs:
                return f"{label} is mostly in {dom}."
            scat = PatchCaptionGenerator._format_list(scattered_locs)
            return f"{label} is mostly in {dom}, scattered in {scat}."

        if class_fraction >= 0.05:
            if not all_locs:
                return ""
            regions = PatchCaptionGenerator._format_list(all_locs)
            return f"{label} is scattered in {regions}."

        if not all_locs:
            return ""
        regions = PatchCaptionGenerator._format_list(all_locs)
        return f"{label} appears as small patches in {regions}."

    @staticmethod
    def _format_list(items: List[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    @staticmethod
    def _compress_locations(cells: List[Tuple[int, int]]) -> List[str]:
        """Compress 3x3 grid (row, col) cell list into readable location strings.

        Row 0=top, 1=middle, 2=bottom. Col 0=left, 1=center, 2=right.
        Multi-pass: full rows → full columns → partial rows → partial columns → singles.
        """
        if not cells:
            return []

        row_names = {0: "top", 1: "middle", 2: "bottom"}
        col_names = {0: "left", 1: "center", 2: "right"}

        rows_map: Dict[int, List[int]] = {}
        cols_map: Dict[int, List[int]] = {}
        cell_set = set(cells)
        for r, c in sorted(cell_set):
            rows_map.setdefault(r, []).append(c)
            cols_map.setdefault(c, []).append(r)

        labels: List[str] = []
        used: set = set()

        # Pass 1: full rows (all 3 columns)
        for r in (0, 1, 2):
            if len(set(rows_map.get(r, []))) == 3:
                labels.append(f"the {row_names[r]} part")
                for c in range(3):
                    used.add((r, c))

        # Pass 2: full columns (all 3 rows) for remaining cells
        for c in (0, 1, 2):
            remaining = [r for r in cols_map.get(c, []) if (r, c) not in used]
            if len(set(remaining)) == 3:
                label = "the center" if c == 1 else f"the {col_names[c]} side"
                labels.append(label)
                for r in range(3):
                    used.add((r, c))

        # Pass 3: partial rows (2 contiguous columns)
        for r in (0, 1, 2):
            cols = sorted(c for c in rows_map.get(r, []) if (r, c) not in used)
            if len(cols) == 2:
                if cols == [0, 1]:
                    labels.append(f"the {row_names[r]} left")
                    used.update({(r, 0), (r, 1)})
                elif cols == [1, 2]:
                    labels.append(f"the {row_names[r]} right")
                    used.update({(r, 1), (r, 2)})
                else:  # [0, 2] non-contiguous
                    labels.append(f"the {row_names[r]} left")
                    labels.append(f"the {row_names[r]} right")
                    used.update({(r, 0), (r, 2)})

        # Pass 4: partial columns (2 contiguous rows) for remaining cells
        for c in (0, 1, 2):
            rows = sorted(r for r in cols_map.get(c, []) if (r, c) not in used)
            if len(rows) == 2:
                col_label = "center" if c == 1 else col_names[c]
                if rows == [0, 1]:
                    labels.append(f"the top {col_label}")
                    used.update({(0, c), (1, c)})
                elif rows == [1, 2]:
                    labels.append(f"the bottom {col_label}")
                    used.update({(1, c), (2, c)})
                else:  # [0, 2] non-contiguous
                    labels.append(f"the top {col_label}")
                    labels.append(f"the bottom {col_label}")
                    used.update({(0, c), (2, c)})

        # Pass 5: remaining individual cells
        def _cell_label(r: int, c: int) -> str:
            vr, hc = row_names[r], col_names[c]
            if vr == "middle" and hc == "center":
                return "the middle"
            if vr == "middle":
                return f"the {hc}"
            if hc == "center":
                return f"the {vr} center"
            return f"the {vr} {hc}"

        for r, c in sorted(cell_set):
            if (r, c) not in used:
                labels.append(_cell_label(r, c))

        # Deduplicate preserving order
        seen: set = set()
        out: List[str] = []
        for L in labels:
            if L not in seen:
                out.append(L)
                seen.add(L)
        return out

    @staticmethod
    def _update_patch_meta(label_path: Path, caption: dict) -> None:
        """Write caption into the patch's meta.json."""
        patch_dir = label_path.parent
        meta_path = patch_dir / "meta.json"
        if not meta_path.exists():
            return
        try:
            import json as _json
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return
        meta["caption"] = caption
        from utils_io import atomic_write_json
        atomic_write_json(meta_path, meta)

    # ---- high-level runner ----

    def _process_record(self, record: dict) -> dict:
        """Caption a single record. Defined as an instance method so it is
        picklable for multiprocessing."""
        import json as _json

        fracs: Dict[str, float] = record.get("class_fractions") or {}
        if not fracs:
            return record

        label_path = Path(record["label_path"])
        lbl = self._load_label(label_path)
        spatial_stats = self.compute_spatial_stats(lbl)
        caption_text, fracs = self.build_caption(fracs, spatial_stats)

        try:
            roads_visible = self._detect_roads(Path(record["image_path"]))
        except Exception:
            roads_visible = False
        road_phrase = "Roads are visible." if roads_visible else "No roads visible."
        caption_text = f"{caption_text} {road_phrase}"

        caption = {"question": "Describe the image", "answer": caption_text}
        record["caption"] = caption
        record["caption_meta"] = {"spatial_stats": spatial_stats}
        self._update_patch_meta(label_path, caption)
        return record

    def run(self, workers: int = 1) -> None:
        """Generate captions for every record in the index.

        workers > 1 processes patches in parallel using a multiprocessing pool,
        making better use of multi-core nodes.
        """
        import json as _json
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import functools

        jsonl_path = self.index_builder.out_jsonl
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Index not found: {jsonl_path}. Run build_index() first.")

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        records = [_json.loads(ln) for ln in lines if ln.strip()]
        total = len(records)

        print(f"[captions] Processing {total} records with {workers} worker(s)...")

        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(self._process_record, records, chunksize=32))
        else:
            results = [self._process_record(r) for r in records]

        processed = sum(1 for r in results if r.get("caption"))
        skipped = total - processed

        tmp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for rec in results:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_path.replace(jsonl_path)

        print(f"[captions] Done. processed={processed}, skipped={skipped}, index_records={total}")


