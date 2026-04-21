"""Generate structured captions for patches based on MapBiomas labels.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import rasterio
import math
from MapBiomasClasses import MapBiomasClasses
from PatchIndexBuilder import PatchIndexBuilder

try:
    from scipy.ndimage import label as nd_label, binary_erosion as nd_binary_erosion
except Exception:
    nd_label = None
    nd_binary_erosion = None


@dataclass
class CaptionConfig:
    min_dominant_frac: float = 0.5
    min_secondary_frac: float = 0.01
    max_secondary: int = 3
    no_spatial: bool = False


@dataclass
class SpatialStats:
    pixel_count: int
    n_cc: int
    largest_cc_frac: float
    median_cc_patch_frac: float
    p90_cc_patch_frac: float
    small_cc_area_frac: float
    small_cc_share: float
    n_small_cc_patch: int
    n_large_cc_patch: int
    centroid_y: float
    centroid_x: float
    var_x: float
    var_y: float
    elongation: float
    contact_ratio: float

class PatchCaptionGenerator:
    """Generate and attach captions to patches.
    """

    def __init__(
        self,
        *,
        patch_root: Path,
        index_builder: PatchIndexBuilder,
        config: Optional[CaptionConfig] = None,
    ) -> None:
        self.patch_root = patch_root
        self.index_builder = index_builder
        self.config = config or CaptionConfig()

    # ---- low-level helpers ----

    @staticmethod
    def _load_label(label_path: Path) -> np.ndarray:
        with rasterio.open(label_path) as src:
            return src.read(1)

    @staticmethod
    def _percent(frac: float) -> int:
        return int(round(frac * 100.0))

    @staticmethod
    def _verb_for_label(label: str) -> str:
        return "are" if label.endswith("areas") else "is"

    @staticmethod
    def _connected_components_8(mask: np.ndarray) -> List[int]:
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        sizes: List[int] = []
        for y in range(h):
            for x in range(w):
                if not mask[y, x] or visited[y, x]:
                    continue
                stack = [(y, x)]
                visited[y, x] = True
                size = 0
                while stack:
                    cy, cx = stack.pop()
                    size += 1
                    for ny in (cy - 1, cy, cy + 1):
                        if ny < 0 or ny >= h:
                            continue
                        for nx in (cx - 1, cx + 1):
                            if nx < 0 or nx >= w:
                                continue
                            if not mask[ny, nx] or visited[ny, nx]:
                                continue
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                sizes.append(size)
        return sizes

    def analyze_class_spatial(
        self,
        mask: np.ndarray,
        *,
        valid_pixel_count: int,
        lbl: Optional[np.ndarray] = None,
        class_idx: Optional[int] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Optional[SpatialStats]:
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            return None

        h, w = mask.shape
        if nd_label is not None:
            structure = np.ones((3, 3), dtype=np.int8)
            labels, n_cc = nd_label(mask, structure=structure)
            if n_cc > 0:
                sizes = np.bincount(labels.ravel())[1:]
                sizes = sizes[sizes > 0]
                sizes_list = sizes.tolist()
            else:
                sizes_list = []
        else:
            sizes_list = self._connected_components_8(mask)
            n_cc = len(sizes_list)

        largest_cc = max(sizes_list) if sizes_list else 0
        largest_cc_frac = float(largest_cc) / float(pixel_count)

        valid_denom = max(int(valid_pixel_count), 1)
        if sizes_list:
            cc_fracs = np.array(sizes_list, dtype=np.float64) / float(pixel_count)
            cc_patch_fracs = np.array(sizes_list, dtype=np.float64) / float(valid_denom)
            median_cc_patch_frac = float(np.median(cc_patch_fracs))
            p90_cc_patch_frac = float(np.percentile(cc_patch_fracs, 90))
            sizes_arr = np.array(sizes_list, dtype=np.float64)
            small_mask = (sizes_arr <= 32) | (cc_patch_fracs < 0.002)
            large_mask = (cc_patch_fracs >= 0.02) | (sizes_arr >= (0.10 * float(pixel_count)))
            n_small_cc_patch = int(small_mask.sum())
            n_large_cc_patch = int(large_mask.sum())
            small_cc_area_frac = float(cc_fracs[small_mask].sum()) if n_small_cc_patch > 0 else 0.0
            small_cc_share = float(n_small_cc_patch) / float(max(n_cc, 1))
        else:
            median_cc_patch_frac = 0.0
            p90_cc_patch_frac = 0.0
            small_cc_area_frac = 0.0
            small_cc_share = 0.0
            n_small_cc_patch = 0
            n_large_cc_patch = 0

        ys, xs = np.nonzero(mask)
        cy = float(ys.mean()) / max(h - 1, 1)
        cx = float(xs.mean()) / max(w - 1, 1)

        x = xs.astype(np.float64) / max(w - 1, 1)
        y = ys.astype(np.float64) / max(h - 1, 1)
        mx = x.mean()
        my = y.mean()
        vx = float(np.mean((x - mx) ** 2))
        vy = float(np.mean((y - my) ** 2))
        cov = float(np.mean((x - mx) * (y - my)))
        trace = vx + vy
        det = max(vx * vy - cov * cov, 0.0)
        disc = max(trace * trace - 4.0 * det, 0.0)
        lambda_max = (trace + math.sqrt(disc)) / 2.0
        lambda_min = max((trace - math.sqrt(disc)) / 2.0, 1e-12)
        elongation = math.sqrt(lambda_max / lambda_min)

        contact_ratio = 0.0
        if lbl is not None and class_idx is not None and valid_mask is not None:
            contact_ratio = self._compute_contact_ratio(mask, lbl, valid_mask, class_idx)

        return SpatialStats(
            pixel_count=pixel_count,
            n_cc=int(n_cc),
            largest_cc_frac=float(largest_cc_frac),
            median_cc_patch_frac=median_cc_patch_frac,
            p90_cc_patch_frac=p90_cc_patch_frac,
            small_cc_area_frac=small_cc_area_frac,
            small_cc_share=small_cc_share,
            n_small_cc_patch=n_small_cc_patch,
            n_large_cc_patch=n_large_cc_patch,
            centroid_y=float(cy),
            centroid_x=float(cx),
            var_x=float(vx),
            var_y=float(vy),
            elongation=float(elongation),
            contact_ratio=float(contact_ratio),
        )

    def spatial_phrase_from_stats(self, stats: SpatialStats, *, dom_frac: Optional[float] = None) -> Optional[str]:
        if stats.pixel_count == 0:
            return None

        n_cc = int(stats.n_cc)
        largest_cc_frac = float(stats.largest_cc_frac)
        small_cc_area_frac = float(stats.small_cc_area_frac)
        n_large_cc_patch = int(stats.n_large_cc_patch)
        small_cc_share = float(stats.small_cc_share)
        median_patch = float(stats.median_cc_patch_frac)
        p90_patch = float(stats.p90_cc_patch_frac)
        elongation = float(stats.elongation)
        var_x = float(stats.var_x)
        var_y = float(stats.var_y)
        contact_ratio = float(stats.contact_ratio)

        if largest_cc_frac >= 0.95 and contact_ratio < 0.40:
            structure = "a single contiguous region"
        elif largest_cc_frac >= 0.85 and n_cc <= 5:
            if contact_ratio >= 0.55:
                structure = "a main region interwoven with other classes"
            else:
                structure = "a single main block"
        elif largest_cc_frac >= 0.60 and n_cc <= 8:
            structure = self._counted_structure("a few clustered regions", n_cc)
        else:
            if n_cc >= 6 and median_patch <= 0.0015 and p90_patch <= 0.01:
                structure = self._counted_structure("scattered in small regions", n_cc)
            elif (n_cc >= 8 and small_cc_share >= 0.70) or (small_cc_area_frac >= 0.55 and n_large_cc_patch == 0):
                structure = self._counted_structure("scattered in small regions", n_cc)
            elif n_cc >= 10:
                structure = "scattered in many regions"
            elif n_cc >= 5:
                structure = self._counted_structure("scattered in multiple regions", n_cc)
            else:
                structure = self._counted_structure("multiple regions", n_cc)

        shape = None
        if elongation >= 3.0:
            shape = "forming a linear band"
        elif elongation <= 1.5 and (var_x + var_y) <= 0.02:
            shape = "compact"

        spread = var_x + var_y
        concentrated = self._is_concentrated(stats, spread=spread)

        location = None
        if concentrated:
            location = self._centroid_location_phrase(stats)

        bits = [structure]
        if shape:
            bits.append(shape)
        if location:
            bits.append(location)

        if dom_frac is not None and dom_frac < 0.75 and contact_ratio >= 0.55:
            bits.append("with substantial intermixing")

        phrase = ", ".join(bits)
        return phrase.replace("across the patch", "across the image")

    def compute_stats_and_phrase(
        self,
        mask: np.ndarray,
        *,
        valid_pixel_count: int,
        lbl: Optional[np.ndarray] = None,
        class_idx: Optional[int] = None,
        valid_mask: Optional[np.ndarray] = None,
        dom_frac: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, float | int]], Optional[str]]:
        stats = self.analyze_class_spatial(
            mask,
            valid_pixel_count=valid_pixel_count,
            lbl=lbl,
            class_idx=class_idx,
            valid_mask=valid_mask,
        )
        if stats is None:
            return None, None
        phrase = self.spatial_phrase_from_stats(stats, dom_frac=dom_frac)
        # Convert SpatialStats dataclass instance to a plain dict
        stats_dict: Dict[str, float | int] = {k: float(v) if isinstance(v, float) else int(v) for k, v in asdict(stats).items()}
        return stats_dict, phrase

    def compute_spatial_stats(
        self,
        lbl: np.ndarray,
        *,
        valid_mask: np.ndarray,
        valid_pixel_count: int,
        dominant_class: str,
        dominant_frac: float,
    ) -> Tuple[Dict[str, Optional[Dict[str, float | int]]], Dict[str, str]]:
        spatial_stats: Dict[str, Optional[Dict[str, float | int]]] = {}
        spatial_phrases: Dict[str, str] = {}

        for idx, name in enumerate(MapBiomasClasses.CLASS_NAMES):
            mask = (lbl == idx) & valid_mask
            stats_dict, phrase = self.compute_stats_and_phrase(
                mask,
                valid_pixel_count=valid_pixel_count,
                lbl=lbl,
                class_idx=idx,
                valid_mask=valid_mask,
                dom_frac=dominant_frac if name == dominant_class else None,
            )
            if stats_dict is not None:
                spatial_stats[name] = stats_dict
            if phrase:
                spatial_phrases[name] = phrase

        return spatial_stats, spatial_phrases

    def build_caption(
        self,
        fracs: Dict[str, float],
        spatial_phrases: Dict[str, str],
    ) -> Tuple[str, Dict[str, float], List[str], Dict[str, str]]:
        ordered = sorted(fracs.items(), key=lambda kv: kv[1], reverse=True)
        dominant, dom_frac = ordered[0]

        forced = [
            name
            for name in ("water", "urban")
            if name != dominant and fracs.get(name, 0.0) >= 0.01
        ]
        max_total = min(self.config.max_secondary + len(forced), 4)
        secondary = forced[:max_total]
        remaining = [
            name
            for name, frac in ordered[1:]
            if name not in secondary and frac >= self.config.min_secondary_frac
        ]
        secondary.extend(remaining[: max_total - len(secondary)])

        dom_label = MapBiomasClasses.CLASS_LABELS.get(dominant, dominant)
        secondary_labels = [MapBiomasClasses.CLASS_LABELS.get(n, n) for n in secondary]

        parts: List[str] = []
        if dom_frac >= self.config.min_dominant_frac:
            parts.append(f"The image is dominated by {dom_label} (about {self._percent(dom_frac)}%).")
        else:
            parts.append("The image shows mixed land cover.")

        if secondary_labels:
            sec_bits = [
                f"{MapBiomasClasses.CLASS_LABELS.get(n, n)} (about {self._percent(fracs[n])}%)"
                for n in secondary
            ]
            parts.append(f"Secondary classes include {self._format_list(sec_bits)}.")

        dom_spatial = spatial_phrases.get(dominant)
        if dom_spatial:
            verb = self._verb_for_label(dom_label)
            parts.append(f"{dom_label.capitalize()} {verb} {dom_spatial}.")

        if secondary:
            for name in secondary[:2]:
                sec_spatial = spatial_phrases.get(name)
                if sec_spatial:
                    sec_label = MapBiomasClasses.CLASS_LABELS.get(name, name)
                    verb = self._verb_for_label(sec_label)
                    parts.append(f"{sec_label.capitalize()} {verb} {sec_spatial}.")

        hn = self._entropy_normalized(fracs)
        if dom_frac >= 0.90 and hn <= 0.15:
            parts.append("Land cover is highly homogeneous.")
        elif hn >= 0.65:
            parts.append("Land cover is highly mixed across classes.")
        elif hn >= 0.45:
            parts.append("Land cover is moderately mixed.")
        else:
            parts.append("Land cover is slightly mixed.")

        caption = " ".join(parts)

        return caption, {k: float(v) for k, v in fracs.items()}, secondary, spatial_phrases

    # ---- spatial helper methods ----

    @staticmethod
    def _centroid_location_phrase(stats: SpatialStats) -> Optional[str]:
        cx = float(stats.centroid_x)
        cy = float(stats.centroid_y)

        horiz = None
        if cx <= 0.33:
            horiz = "left"
        elif cx >= 0.67:
            horiz = "right"

        vert = None
        if cy <= 0.33:
            vert = "upper"
        elif cy >= 0.67:
            vert = "lower"

        if horiz and vert:
            return f"mostly in the {vert}-{horiz}"
        if horiz:
            return f"mostly on the {horiz} side"
        if vert:
            return f"mostly in the {vert} part"
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

    @staticmethod
    def _is_concentrated(stats: SpatialStats, *, spread: Optional[float] = None) -> bool:
        if spread is None:
            spread = float(stats.var_x) + float(stats.var_y)
        return (
            float(stats.largest_cc_frac) >= 0.60
            or spread <= 0.05
            or int(stats.n_cc) <= 3
        )

    @staticmethod
    def _counted_structure(structure: str, n_cc: int) -> str:
        if n_cc <= 5:
            if structure in {"a few clustered regions", "multiple regions"}:
                return f"in {n_cc} regions"
            if structure.startswith("scattered"):
                return f"scattered in {n_cc} regions"
        return structure

    @staticmethod
    def _compute_contact_ratio(
        class_mask: np.ndarray,
        lbl: np.ndarray,
        valid_mask: np.ndarray,
        class_idx: int,
    ) -> float:
        if nd_binary_erosion is not None:
            eroded = nd_binary_erosion(class_mask, structure=np.ones((3, 3), dtype=bool))
        else:
            p = np.pad(class_mask, 1, constant_values=False)
            eroded = (
                p[1:-1, 1:-1]
                & p[:-2, :-2]
                & p[:-2, 1:-1]
                & p[:-2, 2:]
                & p[1:-1, :-2]
                & p[1:-1, 2:]
                & p[2:, :-2]
                & p[2:, 1:-1]
                & p[2:, 2:]
            )

        boundary = class_mask & (~eroded)
        boundary_count = int(boundary.sum())
        if boundary_count == 0:
            return 0.0

        h, w = class_mask.shape
        lbl_pad = np.pad(lbl, 1, constant_values=-1)
        valid_pad = np.pad(valid_mask, 1, constant_values=False)

        different = np.zeros((h, w), dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neigh_valid = valid_pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                neigh_lbl = lbl_pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                different |= neigh_valid & (neigh_lbl != class_idx)

        contact_pixels = boundary & different
        return float(contact_pixels.sum()) / float(max(boundary_count, 1))

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

        def _transform(rec: dict) -> dict:
            nonlocal processed, skipped

            fracs: Dict[str, float] = rec.get("class_fractions") or {}
            if not fracs:
                skipped += 1
                return rec  # no fractions → nothing to do

            label_path = Path(rec["label_path"])

            spatial_stats: Dict[str, Optional[Dict[str, float | int]]] = {}
            spatial_phrases: Dict[str, str] = {}

            if not self.config.no_spatial:
                lbl = self._load_label(label_path)
                valid = (lbl != MapBiomasClasses.IGNORE)
                valid_pixel_count = int(valid.sum())
                dominant_class = max(fracs.items(), key=lambda kv: kv[1])[0]
                dominant_frac = float(fracs[dominant_class])
                spatial_stats, spatial_phrases = self.compute_spatial_stats(
                    lbl,
                    valid_mask=valid,
                    valid_pixel_count=valid_pixel_count,
                    dominant_class=dominant_class,
                    dominant_frac=dominant_frac,
                )

            caption_text, fracs, secondary, spatial_phrases = self.build_caption(
                fracs, spatial_phrases
            )
            caption = {"question": "Describe the image", "answer": caption_text}

            dominant_class = max(fracs.items(), key=lambda kv: kv[1])[0]
            dominant_fraction = float(fracs[dominant_class])

            rec["caption"] = caption
            rec["caption_meta"] = {
                "dominant_class": dominant_class,
                "dominant_fraction": dominant_fraction,
                "secondary_classes": secondary,
                "spatial_phrases": spatial_phrases,
                "spatial_stats": spatial_stats,
            }

            # Mirror caption into per-patch meta.json
            self._update_patch_meta(label_path, caption)

            processed += 1
            if processed % 2000 == 0:
                print(f"[captions] processed={processed}")
            return rec

        written = self.index_builder.update_records(_transform)
        print(f"[captions] Done. processed={processed}, skipped={skipped}, index_records={written}")


