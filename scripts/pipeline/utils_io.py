from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple, Callable


def load_json(path: Path) -> Dict[str, object]:
    """Load a JSON file into a dict using UTF-8 encoding."""

    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: Dict[str, object]) -> None:
    """Atomically write a JSON object to ``path``.

    Writes to a temporary sibling file and then replaces the target.
    """

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def iter_meta_json_paths(patch_root: Path, pattern: str = "**/meta.json") -> Iterator[Path]:
    """Yield meta.json paths under ``patch_root`` matching ``pattern``.

    This uses ``Path.glob`` for simplicity and deterministic ordering.
    """

    return iter(sorted(patch_root.glob(pattern)))


def stream_jsonl(
    in_path: Path,
    out_path: Path,
    transform: Callable[[Dict[str, object]], Dict[str, object]],
    *,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Stream records from ``in_path``, apply ``transform``, and write to ``out_path``.

    Returns ``(processed, written)``.
    """

    processed = 0
    written = 0

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    writer = tmp_path.open("w", encoding="utf-8") if not dry_run else None

    try:
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n")
                if not raw.strip():
                    if writer is not None:
                        writer.write("\n")
                    continue

                rec = json.loads(raw)
                processed += 1
                rec2 = transform(rec)
                if writer is not None:
                    writer.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                written += 1
    finally:
        if writer is not None:
            writer.close()

    if not dry_run:
        tmp_path.replace(out_path)

    return processed, written


def matches_patch_id(rec_patch_id: str, target_ids: Iterable[str]) -> bool:
    """Return True if ``rec_patch_id`` matches any id in ``target_ids`` by full id or suffix."""

    if not rec_patch_id:
        return False
    target_set = set(target_ids)
    if rec_patch_id in target_set:
        return True
    suffix = rec_patch_id.split("/")[-1]
    return suffix in target_set if suffix else False
