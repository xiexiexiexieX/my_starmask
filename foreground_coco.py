"""Build read-only, single-foreground COCO views for model training.

The project source annotations may use category ids for provenance or older
experiments. Training only needs one foreground class. This module always
writes a derived copy below a work directory and never edits ``data_root``.
Image-level ``mode`` and every instance's extra metadata remain unchanged for
evaluation after inference.
"""

import json
from pathlib import Path


FOREGROUND_CATEGORIES = [{"id": 1, "name": "star"}]


def prepare_foreground_coco(data_root, output_dir):
    """Return a directory containing single-class COCO annotation copies."""
    data_root = Path(data_root).resolve()
    source_dir = data_root / "annotations"
    output_dir = Path(output_dir).resolve()
    try:
        output_dir.relative_to(data_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"Derived annotations must be outside source data_root: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    for split in ("train", "val", "test"):
        source = source_dir / f"{split}.json"
        if not source.exists():
            continue
        with source.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)

        data["categories"] = FOREGROUND_CATEGORIES
        for annotation in data.get("annotations", []):
            annotation.setdefault("source_category_id", annotation.get("category_id"))
            annotation["category_id"] = 1
            annotation["iscrowd"] = int(annotation.get("iscrowd", 0))

        destination = output_dir / f"{split}.json"
        destination.write_text(
            json.dumps(data, ensure_ascii=True, indent=2), encoding="ascii")
        converted += 1

    if converted == 0:
        raise FileNotFoundError(f"No annotations found under {source_dir}")
    return output_dir
