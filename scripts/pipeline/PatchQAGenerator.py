"""Add QA fields to patch meta.json files and dataset_index.jsonl.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from MapBiomasClasses import MapBiomasClasses
from PatchIndexBuilder import PatchIndexBuilder

from utils_io import (
    load_json,
    atomic_write_json,
)

PRESENCE_Q_TEMPLATE = "Is there any {CLASS} areas in the image?"


@dataclass(frozen=True)
class QAResult:
    qa: Dict[str, object]
    qa_meta: Dict[str, object]


@dataclass(frozen=True)
class QAGeneratorConfig:
    """Configuration for presence-QA generation."""

    classes: List[str]
    presence_q_template: str


class PatchQAGenerator:
    """Stage-style QA generator.
    """

    def __init__(
        self,
        *,
        patch_root: Path,
        index_builder: PatchIndexBuilder,
        dry_run: bool = False,
        limit: Optional[int] = None,
        config: Optional[QAGeneratorConfig] = None,
    ) -> None:
        self.patch_root = patch_root
        self.index_builder = index_builder
        self.dry_run = dry_run
        self.limit = limit
        self.config = config or QAGeneratorConfig(
            classes=MapBiomasClasses.CLASS_NAMES,
            presence_q_template=PRESENCE_Q_TEMPLATE,
        )


    def build_qa(self, fracs: Dict[str, float], *, source: str) -> QAResult:
        entries: List[Dict[str, object]] = []
        for name in self.config.classes:
            frac = float(fracs.get(name, 0.0))
            present = frac > 0.0
            question = self.config.presence_q_template.format(CLASS=name)
            entries.append({
                "class": name,
                "question": question,
                "answer": "yes" if present else "no",
            })

        qa = {"vqa_presence": entries}
        qa_meta = {"source": source, "fractions": fracs}
        return QAResult(qa=qa, qa_meta=qa_meta)

    # ---- meta.json helpers ----

    def _update_meta(self, patch_root: Path, patch_id: str, qa: Dict, caption) -> None:
        """Write qa (and optional caption) into the patch's meta.json."""
        meta_path = patch_root / patch_id / "meta.json"
        if not meta_path.exists():
            return
        meta = load_json(meta_path)

        # Preserve existing grounding_converted (top-level or legacy nested).
        grounding = meta.get("grounding_converted")
        if grounding is None:
            old_qa = meta.get("qa") or {}
            if isinstance(old_qa, dict):
                grounding = old_qa.get("grounding_converted")

        meta["qa"] = qa
        if grounding is not None:
            meta["grounding_converted"] = grounding

        if caption is not None:
            meta["caption"] = caption

        # Never store qa_meta in per-patch meta.json.
        meta.pop("qa_meta", None)

        if not self.dry_run:
            atomic_write_json(meta_path, meta)

    # ---- public API ----

    def run(self) -> None:
        """Generate QA for every record in the index.

        """
        processed = 0
        skipped = 0

        def _transform(rec: dict) -> dict:
            nonlocal processed, skipped

            fracs: Dict[str, float] = rec.get("class_fractions") or {}
            if not fracs:
                skipped += 1
                return rec

            qa_result = self.build_qa(fracs, source="class_fractions")

            rec["qa"] = qa_result.qa
            rec["qa_meta"] = qa_result.qa_meta

            # Mirror QA into per-patch meta.json.
            caption = rec.get("caption")
            self._update_meta(
                self.patch_root,
                str(rec.get("patch_id", "")),
                qa_result.qa,
                caption,
            )

            processed += 1
            if processed % 2000 == 0:
                print(f"[qa] processed={processed}")
            return rec

        written = self.index_builder.update_records(_transform)
        print(f"[qa] Done. processed={processed}, skipped={skipped}, index_records={written}")
