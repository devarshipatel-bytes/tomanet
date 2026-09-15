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


def materialise_det(assignment, out_dir: Path, copy: bool, *, oversample_cap: int = 10) -> list[str]:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    class_names = sorted({name for _, boxes in assignment.values() for name, *_ in boxes})
    name_to_id = {name: i for i, name in enumerate(class_names)}

    # tomato_village_det is ~95:1 leaf_miner:potassium_deficiency by instance count, and the
    # two rarest classes are exactly the two worst-scoring ones. Duplicate each train image
    # sqrt(freq-ratio) times (capped) so rare-class images are sampled more often per epoch.
    # ponytail: naive frequency heuristic, not a real weighted sampler - revisit if per-class
    # AP is still badly skewed after this.
    train_class_counts = Counter(
        name for split, boxes in assignment.values() if split == "train" for name, *_ in boxes
    )
    max_count = max(train_class_counts.values(), default=1)

    for path, (split, boxes) in assignment.items():
        images_dir = out_dir / split / "images"
        labels_dir = out_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{abs(hash(str(path))) % 10**8}_{path.stem}"
        lines = [f"{name_to_id[name]} {cx} {cy} {w} {h}" for name, cx, cy, w, h in boxes]
        label_text = "\n".join(lines) + "\n"

        repeats = 1
        if split == "train" and boxes:
            rarest = min(train_class_counts[name] for name, *_ in boxes)
            repeats = min(oversample_cap, max(1, round((max_count / rarest) ** 0.5)))

        for r in range(repeats):
            suffix = "" if r == 0 else f"_dup{r}"
            image_target = images_dir / f"{stem}{suffix}{path.suffix}"
            if copy:
                shutil.copy2(path, image_target)
            else:
                image_target.symlink_to(path.resolve())
            (labels_dir / f"{stem}{suffix}.txt").write_text(label_text, encoding="utf-8")

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return class_names


def report_det(assignment, class_names) -> None:
    per_split_images = Counter(split for split, _ in assignment.values())
    per_split_boxes = Counter()
    per_class = defaultdict(Counter)
    for split, boxes in assignment.values():
        per_split_boxes[split] += len(boxes)
        for name, *_ in boxes:
            per_class[name][split] += 1

    print(f"\n  {'split':<8} {'images':>8} {'boxes':>8}")
    print("  " + "-" * 26)
    for split in SPLITS:
        print(f"  {split:<8} {per_split_images[split]:>8} {per_split_boxes[split]:>8}")

    print(f"\n  {'class':<26} {'train':>7} {'val':>6} {'test':>6} {'total':>7}")
    print("  " + "-" * 54)
    for cls in class_names:
        counts = per_class[cls]
        total = sum(counts.values())
        print(f"  {cls:<26} {counts['train']:>7} {counts['val']:>6} {counts['test']:>6} {total:>7}")

    sizes = [sum(per_class[c].values()) for c in class_names]
    if sizes:
        ratio = max(sizes) / max(1, min(sizes))
        note = "  <- use class-balanced loss (ratio > 5)" if ratio > 5 else ""
        print(f"\n  box imbalance ratio (largest/smallest class): {ratio:.1f}{note}")


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
    parser.add_argument("--task", default="cls", choices=["cls", "det"],
                        help="cls: ImageFolder tree (default). det: YOLO-format boxes.")
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

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        sys.exit(f"ratios must sum to 1.0, got {sum(args.ratios)}")

    if args.task == "det":
        if args.dataset not in DET_ADAPTERS:
            sys.exit(f"no detection adapter for '{args.dataset}'. Known: {', '.join(sorted(DET_ADAPTERS))}")
    else:
        if args.dataset not in ADAPTERS:
            sys.exit(f"no adapter for '{args.dataset}'. Known: {', '.join(sorted(ADAPTERS))}")

    root = RAW / args.dataset
    if not root.exists():
        sys.exit(f"{root} not found - run: python scripts/download_datasets.py --only {args.dataset}")

    classes = load_classes()
    mapping = classes["mapping"].get(args.dataset, {}).get("classes", {})
    if not mapping:
        sys.exit(f"no class mapping for '{args.dataset}' in configs/classes.yaml")

    if args.task == "det":
        print(f"preparing {args.dataset} (detection)")
        items = DET_ADAPTERS[args.dataset](root, mapping)
        if not items:
            sys.exit("adapter produced no images - check the directory layout and the mapping")
        n_boxes = sum(len(boxes) for _, boxes in items)
        n_classes = len({name for _, boxes in items for name, *_ in boxes})
        print(f"  {len(items)} images, {n_boxes} boxes, {n_classes} mapped classes")

        assignment = grouped_stratified_split(items, args.ratios, args.seed)
        out_dir = PROCESSED / f"{args.dataset}_det"
        class_names = materialise_det(assignment, out_dir, args.copy)
        report_det(assignment, class_names)

        INTERIM.mkdir(parents=True, exist_ok=True)
        split_record = {
            "dataset": args.dataset,
            "task": "det",
            "seed": args.seed,
            "ratios": args.ratios,
            "grouped_by": "base filename (strips trailing _augN) to keep augmented "
                          "copies of one photo in a single split",
            "classes": class_names,
            "splits": {str(p): s for p, (s, _) in assignment.items()},
        }
        (out_dir / "split.json").write_text(json.dumps(split_record, indent=2), encoding="utf-8")
        print(f"\n  -> {out_dir}")
        print(f"  -> {out_dir / 'split.json'} (seed {args.seed}, reproducible)")
        return

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
