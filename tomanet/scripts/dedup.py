#!/usr/bin/env python3
"""Perceptual-hash deduplication within and across the downloaded datasets.

Why this matters: PlantVillage contains near-duplicates that leak across random splits,
and most Kaggle "tomato" sets are re-splits of it. Without this step, a
"cross-dataset" result can be measuring memorisation.

    python scripts/dedup.py --report                      # counts only, no files touched
    python scripts/dedup.py --report --datasets plantvillage taiwan
    python scripts/dedup.py --write-manifest              # emit keep/drop lists

Nothing is ever deleted. The manifest is a list of paths to exclude, which the data
loaders read - so a mistake here is always reversible.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
INTERIM_DIR = REPO_ROOT / "data" / "interim"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_THRESHOLD = 6  # Hamming distance on a 64-bit pHash


def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            yield path


def phash(image, size: int = 32, keep: int = 8) -> int:
    """64-bit perceptual hash: DCT of a 32x32 grayscale, low-frequency 8x8 vs its median.

    Equivalent to imagehash.phash; inlined to avoid the dependency.
    """
    import numpy as np
    from scipy.fft import dctn

    pixels = np.asarray(image.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=float)
    low = dctn(pixels, norm="ortho")[:keep, :keep]
    bits = low > np.median(low[1:])  # exclude DC from the median, keep it as a bit
    return int("".join("1" if b else "0" for b in bits.flatten()), 2)


def phash_all(paths: list[Path]) -> dict[Path, int]:
    """Map each image to its 64-bit perceptual hash. Unreadable files are skipped."""
    from tqdm import tqdm

    hashes: dict[Path, int] = {}
    for path in tqdm(paths, desc="hashing", unit="img", disable=not sys.stderr.isatty()):
        try:
            with Image.open(path) as image:
                hashes[path] = phash(image)
        except Exception:  # noqa: BLE001 - a corrupt image should not stop the pass
            continue
    return hashes


def group_duplicates(hashes: dict[Path, int], threshold: int) -> list[list[Path]]:
    """Cluster images whose hashes are within `threshold` bits.

    Exact matches are bucketed first (the common case), then near-matches are found by
    comparing bucket representatives only.
    """
    exact: dict[int, list[Path]] = defaultdict(list)
    for path, value in hashes.items():
        exact[value].append(path)

    if threshold == 0:
        return [group for group in exact.values() if len(group) > 1]

    # ponytail: O(n^2) over unique hashes. Fine to ~50k uniques; use BK-tree if it grows.
    representatives = list(exact)
    parent = {value: value for value in representatives}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for i, a in enumerate(representatives):
        for b in representatives[i + 1 :]:
            if bin(a ^ b).count("1") <= threshold:
                parent[find(a)] = find(b)

    clusters: dict[int, list[Path]] = defaultdict(list)
    for value, paths in exact.items():
        clusters[find(value)].extend(paths)

    return [group for group in clusters.values() if len(group) > 1]


def dataset_of(path: Path) -> str:
    return path.relative_to(RAW_DIR).parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", help="limit to these (default: all downloaded)")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"max Hamming distance, 0 = exact (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--report", action="store_true", help="print counts only")
    parser.add_argument("--write-manifest", action="store_true", help="write the drop list")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        sys.exit(f"no data at {RAW_DIR} - run scripts/download_datasets.py first")

    roots = (
        [RAW_DIR / name for name in args.datasets]
        if args.datasets
        else [p for p in sorted(RAW_DIR.iterdir()) if p.is_dir()]
    )
    missing = [p.name for p in roots if not p.exists()]
    if missing:
        sys.exit(f"not downloaded: {', '.join(missing)}")

    paths = [p for root in roots for p in iter_images(root)]
    if not paths:
        sys.exit("no images found")

    print(f"{len(paths)} images across {len(roots)} dataset(s)")
    hashes = phash_all(paths)
    print(f"{len(hashes)} hashed ({len(paths) - len(hashes)} unreadable)")

    groups = group_duplicates(hashes, args.threshold)
    within = [g for g in groups if len({dataset_of(p) for p in g}) == 1]
    across = [g for g in groups if len({dataset_of(p) for p in g}) > 1]

    print(f"\nduplicate clusters (threshold {args.threshold}):")
    print(f"  within a dataset : {len(within):>6}  ({sum(len(g) - 1 for g in within)} redundant images)")
    print(f"  across datasets  : {len(across):>6}  ({sum(len(g) - 1 for g in across)} redundant images)")

    if across:
        pairs: dict[tuple[str, str], int] = defaultdict(int)
        for group in across:
            names = sorted({dataset_of(p) for p in group})
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    pairs[(a, b)] += 1
        print("\n  overlapping pairs - these are NOT independent domains:")
        for (a, b), count in sorted(pairs.items(), key=lambda kv: -kv[1]):
            print(f"    {a} <-> {b}: {count} shared cluster(s)")

    if args.report and not args.write_manifest:
        return

    # Keep the first path in each cluster, drop the rest.
    drop = sorted(str(p.relative_to(REPO_ROOT)) for g in groups for p in sorted(g)[1:])
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    manifest = INTERIM_DIR / "duplicates.json"
    manifest.write_text(
        json.dumps(
            {"threshold": args.threshold, "n_clusters": len(groups), "drop": drop}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nmanifest: {manifest} ({len(drop)} paths to exclude, nothing deleted)")


if __name__ == "__main__":
    main()
