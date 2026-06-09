#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate per-category fields files in validate_json.py's `field_categories`
list-format, so the hardcoded validator gates each item against exactly the
fields its category requires (universal core + applicable conditional groups).
"""
from __future__ import annotations

from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
MASTER = BASE / "fields.yaml"
OUTLINE = BASE / "outline.yaml"
OUT_DIR = BASE / "fields_by_category"


def build() -> None:
    master = yaml.safe_load(MASTER.read_text(encoding="utf-8"))
    outline = yaml.safe_load(OUTLINE.read_text(encoding="utf-8"))

    categories: list[str] = master["categories"]
    core_groups: dict = master["core_fields"]
    conditional: list[dict] = master["conditional_fields"]

    # core field-categories (shared by every item), validator list-format
    core_field_categories = []
    for group_name, field_list in core_groups.items():
        core_field_categories.append(
            {
                "category": f"core_{group_name}",
                "fields": [
                    {
                        "name": f["name"],
                        "description": f["description"],
                        "detail_level": f["detail_level"],
                        "required": True,
                    }
                    for f in field_list
                ],
            }
        )

    # map category -> its conditional groups
    cond_by_cat: dict[str, list[dict]] = {c: [] for c in categories}
    for grp in conditional:
        block = {
            "category": grp["group"],
            "fields": [
                {
                    "name": f["name"],
                    "description": f["description"],
                    "detail_level": f["detail_level"],
                    "required": True,
                }
                for f in grp["fields"]
            ],
        }
        for cat in grp["applies_to"]:
            cond_by_cat[cat].append(block)

    OUT_DIR.mkdir(exist_ok=True)
    counts = {}
    for cat in categories:
        field_categories = core_field_categories + cond_by_cat[cat]
        n = sum(len(b["fields"]) for b in field_categories)
        counts[cat] = n
        doc = {
            "topic_slug": master["topic_slug"],
            "category": cat,
            "field_categories": field_categories,
        }
        (OUT_DIR / f"{cat}.yaml").write_text(
            yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=120),
            encoding="utf-8",
        )

    # report
    used = {it["category"] for it in outline["items"]}
    for cat in categories:
        flag = "" if cat in used else "  (no items!)"
        print(f"  {cat:34s} {counts[cat]:3d} fields{flag}")
    print(f"\nWrote {len(categories)} files to {OUT_DIR}")


if __name__ == "__main__":
    build()
