"""Generate structured captions for patches based on MapBiomas labels.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import rasterio
import math
from MapBiomasClasses import MapBiomasClasses
from PatchIndexBuilder import PatchIndexBuilder

@dataclass
class SpatialStats:
    pixel_count: int
    centroid_y: float
    centroid_x: float
    locations: List[str]

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

    def analyze_class_spatial(
        self,
        mask: np.ndarray,
    ) -> Optional[SpatialStats]:
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            return None

        height, width = mask.shape

        ys, xs = np.nonzero(mask)
        centroid_y = float(ys.mean()) / max(height - 1, 1)
        centroid_x = float(xs.mean()) / max(width - 1, 1)

        # Determine which 3x3 grid cells contain a meaningful share of the class
        # Map pixels to cell indices 0..2 for y (rows) and x (cols)
        # Use integer math on pixel coordinates rather than normalized centroids
        y_cells = (ys * 3) // max(height, 1)
        x_cells = (xs * 3) // max(width, 1)
        y_cells = np.clip(y_cells, 0, 2)
        x_cells = np.clip(x_cells, 0, 2)

        counts: Dict[Tuple[int, int], int] = {}
        for ry, rx in zip(y_cells, x_cells):
            key = (int(ry), int(rx))
            counts[key] = counts.get(key, 0) + 1

        # Keep cells that contain at least 2% of the class pixels to avoid noise
        significant = [(k, v) for k, v in counts.items() if (v / pixel_count) >= 0.02]
        # If too many small cells pass the threshold, limit to the top N
        significant.sort(key=lambda kv: kv[1], reverse=True)
        MAX_CELLS_TO_REPORT = 3
        # prefer spatially diverse cells: pick top cells but avoid immediate neighbors
        selected: List[Tuple[int, int]] = []
        for k, _ in significant:
            if len(selected) >= MAX_CELLS_TO_REPORT:
                break
            r, c = k
            too_close = False
            for sr, sc in selected:
                if abs(sr - r) + abs(sc - c) <= 1:
                    too_close = True
                    break
            if not too_close:
                selected.append(k)
        # if we didn't pick enough diverse cells, fill with top ones
        if len(selected) < MAX_CELLS_TO_REPORT:
            for k, _ in significant:
                if k not in selected:
                    selected.append(k)
                if len(selected) >= MAX_CELLS_TO_REPORT:
                    break
        loc_cells = selected

        locations: List[str] = self._compress_locations(loc_cells)

        return SpatialStats(
            pixel_count=pixel_count,
            centroid_y=float(centroid_y),
            centroid_x=float(centroid_x),
            locations=locations,
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

        #format class name and percentage to a readable format
        class_info = [
            f"{MapBiomasClasses.CLASS_LABELS.get(class_name, class_name)} (about {self._percent(class_fraction)}%)"
            for class_name, class_fraction in classes_to_mention
        ]
        first = f"Image composed by {self._format_list(class_info)},"

        # 2) Location phrases: allow reporting multiple occupied cells when present
        # First, collect per-class pixel counts and candidate locations.
        class_pixel_counts: Dict[str, int] = {}
        class_locs: Dict[str, List[str]] = {}
        for class_name, _ in classes_to_mention:
            stats = spatial_stats.get(class_name) or {}
            if not isinstance(stats, dict):
                raise ValueError(f"Expected spatial_stats for class '{class_name}' to be a dict, got {type(stats)}")

            stats_pixel_count = stats.get("pixel_count")
            if not isinstance(stats_pixel_count, (int, float)):
                raise ValueError(f"Expected 'pixel_count' in spatial_stats for class '{class_name}' to be a number, got {type(stats_pixel_count)}")

            class_pixel_counts[class_name] = int(stats_pixel_count)
            locs = stats.get("locations")
            if not isinstance(locs, list):
                raise ValueError(f"Expected 'locations' in spatial_stats for class '{class_name}' to be a list, got {type(locs)}")
            class_locs[class_name] = list(locs)

        # Resolve ownership: each compressed location text is assigned to the class
        # with the largest pixel count among claimants.
        loc_claims: Dict[str, Tuple[str, int]] = {}
        for cname, locs in class_locs.items():
            for loc in locs:
                cur = loc_claims.get(loc)
                if cur is None or class_pixel_counts.get(cname, 0) > cur[1]:
                    loc_claims[loc] = (cname, class_pixel_counts.get(cname, 0))

        loc_info: List[str] = []
        for class_name, _ in classes_to_mention:
            owned_locs = [loc for loc in class_locs.get(class_name, []) if loc_claims.get(loc, (None, 0))[0] == class_name]
            if owned_locs:
                if len(owned_locs) == 1:
                    phrase = f"{MapBiomasClasses.CLASS_LABELS.get(class_name, class_name)} is mostly in {owned_locs[0]}"
                else:
                    phrase = f"{MapBiomasClasses.CLASS_LABELS.get(class_name, class_name)} is mostly in {self._format_list(owned_locs)}"
            else:
                # fall back to centroid-based phrasing when no owned compressed locations
                phrase = self._location_phrase_for_class(class_name, spatial_stats)
            if phrase:
                loc_info.append(phrase)
        second = "" if not loc_info else f"{', '.join(loc_info)}."

        # 3) Level of mixedness using normalized entropy
        third = self._mixedness_level(class_fractions)

        caption = " ".join([s for s in (first, second, third) if s])

        return caption, {k: float(v) for k, v in class_fractions.items()}

    @staticmethod
    def _filter_low_presence_classes(
        ordered: List[Tuple[str, float]],
    ) ->  List[Tuple[str, float]]:
        #classes below 5% are usually not worth mentioning, unless they're water or urban (which are more salient even at low presence)
        classes_to_mention = [
            (name, frac)
            for name, frac in ordered
            if frac >= 0.05 or (name in ("water", "urban") and frac > 0.0)
        ]
        return classes_to_mention
    
    def _mixedness_level(self, fracs: Dict[str, float]) -> str:
        mixedness_score = self._entropy_normalized(fracs)
        if mixedness_score <= 0.15 and fracs and max(fracs.values()) >= 0.90:
            return "Land cover is highly homogeneous."
        elif mixedness_score >= 0.65:
            return "Land cover is highly mixed."
        elif mixedness_score >= 0.45:
            return "Land cover is moderately mixed."
        else:
            return "Land cover is slightly mixed."

    @staticmethod
    def _location_phrase_for_class(
        name: str,
        spatial_stats: Dict[str, Optional[Dict[str, float | int | List[str]]]],
    ) -> str:
        stats = spatial_stats.get(name)
        if not stats:
            return ""
        
        stats_centroid_x = stats.get("centroid_x")
        stats_centroid_y = stats.get("centroid_y")
        if not isinstance(stats_centroid_x, (int, float)) or not isinstance(stats_centroid_y, (int, float)):
            raise ValueError(f"Expected 'centroid_x' and 'centroid_y' in spatial_stats for class '{name}' to be numbers, got {type(stats_centroid_x)} and {type(stats_centroid_y)}")

        centroid_x = float(stats_centroid_x)
        centroid_y = float(stats_centroid_y)
        horiz_location = PatchCaptionGenerator._horizontal_location(centroid_x)
        vert_location = PatchCaptionGenerator._vertical_location(centroid_y)

        if horiz_location and vert_location:
            loc = f"mostly in the {vert_location} {horiz_location}"
        elif horiz_location:
            loc = f"mostly on the {horiz_location} side"
        elif vert_location:
            loc = f"mostly in the {vert_location} part"
        else:
            loc = "mostly in the middle"

        label = MapBiomasClasses.CLASS_LABELS.get(name, name)
        return f"{label} is {loc}"

    @staticmethod
    def _horizontal_location(centroid_x: float) -> Optional[str]:
        if centroid_x <= 0.33:
            return "left"
        if centroid_x >= 0.67:
            return "right"
        return None

    @staticmethod
    def _vertical_location(centroid_y: float) -> Optional[str]:
        if centroid_y <= 0.33:
            return "upper"
        if centroid_y >= 0.67:
            return "lower"
        return None


    @staticmethod
    def _entropy_normalized(fracs: Dict[str, float]) -> float:
        vals = [v for v in fracs.values() if v > 0.0]
        if not vals:
            return 0.0
        h = 0.0
        for v in vals:
            h -= v * math.log(v)
        return h / math.log(len(fracs))

    @staticmethod
    def _format_list(items: List[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    def _local_entropy(self, lbl: np.ndarray, window: int = 5) -> float:
        """Compute average normalized Shannon entropy over sliding windows.

        Uses integral images per class for efficient windowed counts.
        Returns a normalized entropy in [0,1].
        """
        if window <= 1:
            return 0.0

        h, w = lbl.shape
        K = len(MapBiomasClasses.CLASS_NAMES)

        # prepare padded integral images of shape (h+1, w+1)
        ws = window
        pad_h = h + 1
        pad_w = w + 1

        valid_mask = (lbl != MapBiomasClasses.IGNORE)
        total_valid = int(valid_mask.sum())
        if total_valid == 0:
            return 0.0

        # build integral image per class
        int_imgs = []
        for idx in range(K):
            arr = (lbl == idx).astype('uint32')
            # integral image
            ii = arr.cumsum(axis=0).cumsum(axis=1)
            # pad to (h+1,w+1)
            ii_p = np.zeros((pad_h, pad_w), dtype='uint32')
            ii_p[1:, 1:] = ii
            int_imgs.append(ii_p)

        # compute sums for windows whose top-left corner ranges
        # from (0,0) to (h-ws, w-ws)
        if h < ws or w < ws:
            # fallback to global entropy of fractions
            fracs = []
            for idx in range(K):
                cnt = int(((lbl == idx) & valid_mask).sum())
                if cnt > 0:
                    fracs.append(cnt / total_valid)
            if not fracs:
                return 0.0
            hval = 0.0
            for p in fracs:
                hval -= p * math.log(p)
            return hval / math.log(K) if K > 1 else 0.0

        out_h = h - ws + 1
        out_w = w - ws + 1

        entropies = np.zeros((out_h, out_w), dtype=float)
        for idx in range(K):
            ii = int_imgs[idx]
            # sum over window: use vectorized ops
            S = ii[ws:, ws:] - ii[:-ws, ws:] - ii[ws:, :-ws] + ii[:-ws, :-ws]
            entropies += 0.0  # ensure loop exists; we'll collect counts per class below
            if idx == 0:
                counts = S.astype(float)[None, ...]
            else:
                counts = np.concatenate((counts, S.astype(float)[None, ...]), axis=0)

        # counts shape: (K, out_h, out_w)
        counts_sum = counts.sum(axis=0)
        # avoid division by zero windows
        mask_nonzero = counts_sum > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            fracs = np.divide(counts, counts_sum[None, :, :])
            # compute entropy per window
            # ignore zero fractions
            fpos = np.where(fracs > 0, fracs * np.log(fracs), 0.0)
            H = -np.sum(fpos, axis=0)
            H_norm = H / math.log(K) if K > 1 else H
            # average over non-zero windows
            if np.any(mask_nonzero):
                return float(H_norm[mask_nonzero].mean())
            return 0.0
    @staticmethod
    def _compress_locations(cells: List[Tuple[int, int]]) -> List[str]:
        if not cells:
            return []

        # Build mapping row -> cols and col -> rows
        rows_map: Dict[int, List[int]] = {}
        cols_map: Dict[int, List[int]] = {}
        cell_set = set(cells)
        for r, c in sorted(cell_set):
            rows_map.setdefault(r, []).append(c)
            cols_map.setdefault(c, []).append(r)

        labels: List[str] = []
        used: set = set()

        row_names = {0: "upper", 1: "middle", 2: "lower"}
        col_names = {0: "left", 1: "center", 2: "right"}

        # Prefer row-based compression when a row has at least 2 occupied columns.
        for r in (0, 1, 2):
            cols = sorted(set(rows_map.get(r, [])))
            if not cols:
                continue
            if len(cols) == 3:
                labels.append(f"the {row_names[r]} part")
                for c in cols:
                    used.add((r, c))
            elif len(cols) == 2:
                # contiguous pairs compress to the nearer corner; non-contiguous -> both corners
                if cols == [0, 1]:
                    labels.append(f"the {row_names[r]} left")
                    used.update({(r, 0), (r, 1)})
                elif cols == [1, 2]:
                    labels.append(f"the {row_names[r]} right")
                    used.update({(r, 1), (r, 2)})
                else:
                    # [0,2]
                    labels.append(f"the {row_names[r]} left")
                    labels.append(f"the {row_names[r]} right")
                    used.update({(r, 0), (r, 2)})

        # Then prefer column-based compression for remaining cells
        for c in (0, 1, 2):
            rows = sorted(set(cols_map.get(c, [])))
            remaining_rows = [r for r in rows if (r, c) not in used]
            if not remaining_rows:
                continue
            if len(remaining_rows) == 3:
                labels.append(f"the {col_names[c]} side")
                for r in remaining_rows:
                    used.add((r, c))
            elif len(remaining_rows) == 2:
                if remaining_rows == [0, 1]:
                    labels.append(f"the upper {col_names[c]}")
                    used.update({(0, c), (1, c)})
                elif remaining_rows == [1, 2]:
                    labels.append(f"the lower {col_names[c]}")
                    used.update({(1, c), (2, c)})
                else:
                    labels.append(f"the upper {col_names[c]}")
                    labels.append(f"the lower {col_names[c]}")
                    used.update({(0, c), (2, c)})

        # Remaining individual cells -> corner/center labels
        def _cell_label(row: int, col: int) -> str:
            vr = row_names.get(row, "middle")
            hc = col_names.get(col, "center")
            if vr == "middle" and hc == "center":
                return "the middle"
            if vr == "middle":
                return f"the {hc}"
            if hc == "center":
                return f"the {vr} part"
            return f"the {vr} {hc}"

        for r, c in sorted(cell_set):
            if (r, c) in used:
                continue
            labels.append(_cell_label(r, c))

        # Deduplicate while preserving order
        seen = set()
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

    def run(self) -> None:
        """Generate captions for every record in the index.
        """
        processed = 0
        skipped = 0

        def _transform(record: dict) -> dict:
            nonlocal processed, skipped

            fracs: Dict[str, float] = record.get("class_fractions") or {}
            if not fracs:
                skipped += 1
                return record  # no fractions → nothing to do

            label_path = Path(record["label_path"])

            spatial_stats: Dict[str, Optional[Dict[str, float | int | List]]] = {}

            
            lbl = self._load_label(label_path)
            spatial_stats = self.compute_spatial_stats(
                lbl,
            )

            caption_text, fracs = self.build_caption(fracs, spatial_stats)

            # spatial local entropy: higher means more spatial scrambling
            try:
                local_entropy = float(self._local_entropy(lbl, window=5))
            except Exception:
                local_entropy = 0.0

            # append a short spatial-mixedness sentence based on local entropy
            spatial_phrase = ""
            if local_entropy >= 0.65:
                spatial_phrase = "Pixels are highly spatially mixed."
            elif local_entropy >= 0.45:
                spatial_phrase = "Pixels are moderately spatially mixed."
            elif local_entropy >= 0.25:
                spatial_phrase = "Pixels are slightly spatially mixed."

            if spatial_phrase:
                caption_text = f"{caption_text} {spatial_phrase}"

            caption = {"question": "Describe the image", "answer": caption_text}

            record["caption"] = caption
            # caption_meta no longer contains dominant/secondary fields — keep spatial_stats only
            record["caption_meta"] = {
                "spatial_stats": spatial_stats,
                "local_entropy": local_entropy,
            }

            # Mirror caption into per-patch meta.json
            self._update_patch_meta(label_path, caption)

            processed += 1
            if processed % 2000 == 0:
                print(f"[captions] processed={processed}")
            return record

        written = self.index_builder.update_records(_transform)
        print(f"[captions] Done. processed={processed}, skipped={skipped}, index_records={written}")


