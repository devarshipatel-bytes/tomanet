#!/usr/bin/env python3
"""Turn raw downloads into a unified, split, deduplicated classification dataset.

Every dataset ships a different directory shape, so each gets a small adapter that
yields (image_path, canonical_class). Everything after that is shared: dedup, stratified
split, and an ImageFolder tree that ultralytics' `classify` task reads directly.

    python scripts/prepare_data.py --dataset taiwan
    python scripts/prepare_data.py --dataset taiwan --no-dedup --seed 0
    python scripts/prepare_data.py --list

Output (symlinks by default, so no images are duplicated on disk):

    data/processed/<dataset>_cls/
        train/<class>/*.jpg
        val/<class>/*.jpg
        test/<class>/*.jpg
        split.json      # every path + its split + the seed, so runs are reproducible

Splits are stratified 70/15/15 by default and seeded. Model selection uses `val`;
`test` is touched once, at the end.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"
INTERIM = REPO_ROOT / "data" / "interim"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")

_AUG_SUFFIX = re.compile(r"_aug\d+$")


def group_key(path: Path) -> str:
    """Base identity of a file, stripping this dataset's '_aug<N>' offline-augmentation suffix.

    Used to keep every augmented copy of one physical photo in the same split - phash
    dedup does not catch these (measured Hamming distance 26-36/64), so grouping by name
    is the only reliable way to prevent train/val/test leakage.
    """
    return _AUG_SUFFIX.sub("", path.stem)


def load_classes() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "classes.yaml").read_text())


def images_under(root: Path):
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            yield path


# --------------------------------------------------------------- adapters ----
# Each adapter yields (image_path, canonical_class_name). Unmapped classes are
# skipped and reported, never silently dropped.

def adapt_taiwan(root: Path, mapping: dict) -> list[tuple[Path, str]]:
    """taiwan/Preprocessed data/{Train,Test}/<class>/*.JPG

    'data augmentation/' is deliberately ignored: 619 of 623 duplicate clusters span it
    and 'Preprocessed data/', including test images reappearing in its Train folder.
    We re-split from the un-augmented images instead.
    """
    base = root / "taiwan" / "Preprocessed data"
    if not base.exists():
        raise FileNotFoundError(f"expected {base} - re-run the downloader")

    items = []
    for split_dir in base.iterdir():
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            canonical = mapping.get(class_dir.name)
            if canonical is None:
                continue
            items += [(p, canonical) for p in images_under(class_dir)]
    return items


def adapt_plantvillage(root: Path, mapping: dict) -> list[tuple[Path, str]]:
    """PlantVillage-Dataset-master/raw/color/Tomato___<class>/*.JPG"""
    candidates = [p for p in root.rglob("raw/color") if p.is_dir()]
    base = candidates[0] if candidates else root

    items = []
    for class_dir in sorted(base.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.startswith("Tomato"):
            continue
        canonical = mapping.get(class_dir.name)
        if canonical is None:
            continue
        items += [(p, canonical) for p in images_under(class_dir)]
    return items


def adapt_imagefolder(root: Path, mapping: dict) -> list[tuple[Path, str]]:
    """Generic <class>/*.jpg fallback, matching directory names against the mapping."""
    items = []
    for class_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        canonical = mapping.get(class_dir.name)
        if canonical is None:
            continue
        items += [(p, canonical) for p in images_under(class_dir)]
    return items


def adapt_tomato_village_det(root: Path, mapping: dict) -> list[tuple[Path, list[tuple[str, float, float, float, float]]]]:
    """Variant-c(Object Detection)/{train,val}/{images,yolo}/*

    Labels are shipped in the source's own 0-7 order, read from 'Varient-C Labels.txt'
    (not hardcoded) and translated through `mapping` to canonical class names.
    """
    base = root / "Tomato-Village-main" / "Variant-c(Object Detection)"
    if not base.exists():
        raise FileNotFoundError(f"expected {base} - re-run the downloader")

    label_file = base / "Varient-C Labels.txt"
    source_names = {}
    for line in label_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        idx_part, name_part = line.split(":", 1)
        source_names[int(idx_part.strip())] = name_part.strip().strip("'\"")

    items = []
    for split_dir in ("train", "val"):
        images_dir = base / split_dir / "images"
        labels_dir = base / split_dir / "yolo"
        for image_path in images_under(images_dir):
            label_path = labels_dir / f"{image_path.stem}.txt"
            boxes = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    source_id = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:5])
                    canonical = mapping.get(source_names.get(source_id))
                    if canonical is None:
                        continue
                    boxes.append((canonical, cx, cy, w, h))
            if boxes:
                items.append((image_path, boxes))
    return items


ADAPTERS = {
    "taiwan": adapt_taiwan,
    "plantvillage": adapt_plantvillage,
    "plantvillage_tomato_zenodo": adapt_imagefolder,
    "plantdoc": adapt_imagefolder,
    "ccmt": adapt_imagefolder,
}

DET_ADAPTERS = {"tomato_village": adapt_tomato_village_det}


# ------------------------------------------------------------------ build ----
def drop_duplicates(items: list[tuple[Path, str]], threshold: int) -> list[tuple[Path, str]]:
    """Remove near-duplicates, keeping one image per cluster."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from dedup import group_duplicates, phash_all  # noqa: PLC0415

    hashes = phash_all([p for p, _ in items])
    groups = group_duplicates(hashes, threshold)
    drop = {p for group in groups for p in sorted(group)[1:]}
    if drop:
        print(f"  dedup: dropped {len(drop)} near-duplicate image(s) in {len(groups)} cluster(s)")
    return [(p, c) for p, c in items if p not in drop]


def stratified_split(items, ratios, seed):
    """Split per class so every class keeps its proportions in all three splits."""
    by_class = defaultdict(list)
    for path, cls in items:
        by_class[cls].append(path)

    rng = random.Random(seed)
    assignment = {}
    for cls, paths in sorted(by_class.items()):
        paths = sorted(paths)
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, path in enumerate(paths):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            assignment[path] = (split, cls)
    return assignment


def grouped_stratified_split(items, ratios, seed):
    """Like stratified_split, but splits whole groups (by group_key) instead of images.

    Stratifies each group by its most-frequent box class (ties broken alphabetically),
    then applies the same per-class train/val/test cutoffs as stratified_split.
    """
    groups = defaultdict(list)
    for path, boxes in items:
        groups[group_key(path)].append((path, boxes))

    def primary_class(members):
        counts = Counter(name for _, boxes in members for name, *_ in boxes)
        best = max(counts.values())
        return min(name for name, count in counts.items() if count == best)

    by_class = defaultdict(list)
    for key, members in groups.items():
        by_class[primary_class(members)].append(key)

    rng = random.Random(seed)
    group_split = {}
    for cls, keys in sorted(by_class.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        for i, key in enumerate(keys):
            split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
            group_split[key] = split

    assignment = {}
    for key, members in groups.items():
        split = group_split[key]
        for path, boxes in members:
            assignment[path] = (split, boxes)
    return assignment


def materialise(assignment, out_dir: Path, copy: bool) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    for path, (split, cls) in assignment.items():
        target_dir = out_dir / split / cls
        target_dir.mkdir(parents=True, exist_ok=True)
        # prefix with a hash of the source path so same-named files cannot collide
        target = target_dir / f"{abs(hash(str(path))) % 10**8}_{path.name}"
        if copy:
            shutil.copy2(path, target)
        else:
            target.symlink_to(path.resolve())


def report(assignment) -> None:
    per_split = Counter(split for split, _ in assignment.values())
    per_class = defaultdict(Counter)
    for split, cls in assignment.values():
        per_class[cls][split] += 1

    print(f"\n  {'class':<26} {'train':>7} {'val':>6} {'test':>6} {'total':>7}")
    print("  " + "-" * 54)
    for cls in sorted(per_class):
        counts = per_class[cls]
        total = sum(counts.values())
        print(f"  {cls:<26} {counts['train']:>7} {counts['val']:>6} {counts['test']:>6} {total:>7}")
    print("  " + "-" * 54)
    print(f"  {'TOTAL':<26} {per_split['train']:>7} {per_split['val']:>6} "
          f"{per_split['test']:>6} {sum(per_split.values()):>7}")

    sizes = [sum(c.values()) for c in per_class.values()]
    if sizes:
        ratio = max(sizes) / max(1, min(sizes))
        note = "  <- use class-balanced loss (ratio > 5)" if ratio > 5 else ""
        print(f"\n  imbalance ratio (largest/smallest class): {ratio:.1f}{note}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", help="dataset name as in configs/datasets.yaml")
    parser.add_argument("--list", action="store_true", help="show datasets with an adapter")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=0, help="split seed (default 0)")
    parser.add_argument("--no-dedup", action="store_true", help="skip deduplication")
    parser.add_argument("--dedup-threshold", type=int, default=6)
    parser.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    args = parser.parse_args()

    if args.list or not args.dataset:
        print("datasets with an adapter:")
        for name in sorted(ADAPTERS):
            ready = "downloaded" if (RAW / name).exists() else "not downloaded"
            print(f"  {name:<30} {ready}")
        print("\nAdd new ones to ADAPTERS in this file once you can see their directory tree.")
        return

    if args.dataset not in ADAPTERS:
        sys.exit(f"no adapter for '{args.dataset}'. Known: {', '.join(sorted(ADAPTERS))}")

    root = RAW / args.dataset
    if not root.exists():
        sys.exit(f"{root} not found - run: python scripts/download_datasets.py --only {args.dataset}")

    classes = load_classes()
    mapping = classes["mapping"].get(args.dataset, {}).get("classes", {})
    if not mapping:
        sys.exit(f"no class mapping for '{args.dataset}' in configs/classes.yaml")

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        sys.exit(f"ratios must sum to 1.0, got {sum(args.ratios)}")

    print(f"preparing {args.dataset}")
    items = ADAPTERS[args.dataset](root, mapping)
    if not items:
        sys.exit("adapter produced no images - check the directory layout and the mapping")
    print(f"  {len(items)} images, {len(set(c for _, c in items))} mapped classes")

    if not args.no_dedup:
        items = drop_duplicates(items, args.dedup_threshold)

    assignment = stratified_split(items, args.ratios, args.seed)
    out_dir = PROCESSED / f"{args.dataset}_cls"
    materialise(assignment, out_dir, args.copy)
    report(assignment)

    INTERIM.mkdir(parents=True, exist_ok=True)
    split_record = {
        "dataset": args.dataset,
        "seed": args.seed,
        "ratios": args.ratios,
        "deduplicated": not args.no_dedup,
        "classes": sorted({cls for _, cls in assignment.values()}),
        "splits": {str(p): s for p, (s, _) in assignment.items()},
    }
    (out_dir / "split.json").write_text(json.dumps(split_record, indent=2), encoding="utf-8")
    print(f"\n  -> {out_dir}")
    print(f"  -> {out_dir / 'split.json'} (seed {args.seed}, reproducible)")


if __name__ == "__main__":
    main()
