# TomaNet Detection Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring TomaNet-D (detection) to parity with TomaNet-C (classification): a working `prepare_data.py --task det` adapter for `tomato_village`, a `train_det.py` training script, detection visualization, and an all-scale (n/s/m) verification check — mirroring the existing classification pipeline's conventions.

**Architecture:** `train_cls.py` and `train_det.py` stay as separate sibling scripts (not a merged `--task` script — ultralytics' cls vs. det argument surfaces genuinely diverge). `prepare_data.py` gains a `--task {cls,det}` flag and a second adapter registry (`DET_ADAPTERS`) alongside the existing `ADAPTERS`, reusing `load_classes()`/`images_under()` but using its own split function (`grouped_stratified_split`) because this dataset's offline augmentation makes the existing per-image `stratified_split` unsafe here.

**Tech Stack:** Python 3.12, PyTorch 2.x, Ultralytics 8.4.x (already installed and patched in `tomanet/.venv`), PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-tomanet-detection-parity-design.md`

## Global Constraints

- No new dependencies. Everything needed (`pyyaml`, `matplotlib` via ultralytics, `scikit-learn`) is already installed in `tomanet/.venv`.
- `train_cls.py`'s existing behavior, flags, and printed output must not change — only `train_det.py` is added alongside it.
- `configs/classes.yaml`'s numeric `canonical:` ids are documentation only — confirmed by grep that no script reads them. Do not build any runtime logic that depends on those numbers; the pipeline (both cls and det) keys everything by canonical **name strings**, exactly like the existing classification adapters already do.
- Detection dataset scope for this plan: `tomato_village` only. No other dataset gets a detection adapter.
- The grouped split must guarantee zero base-photo leakage across train/val/test — this is a correctness requirement, not a nice-to-have, and every task touching the split must be tested against it.
- Commands below assume the already-provisioned venv at `tomanet/.venv` (has ultralytics 8.4.137, already patched). Run all Python commands as `.venv/bin/python ...` from the `tomanet/` directory.
- Git commits end with: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`

---

### Task 1: Fix `configs/classes.yaml` canonical vocabulary and `tomato_village` mapping

**Files:**
- Modify: `tomanet/configs/classes.yaml`

**Interfaces:**
- Produces: three new canonical class names (`magnesium_deficiency`, `nitrogen_deficiency`, `potassium_deficiency`) and a corrected `mapping.tomato_village.classes` dict keyed by the dataset's real label strings (`Early_blight`, `Healthy`, `Late_blight`, `Leaf Miner`, `Magnesium Deficiency`, `Nitrogen Deficiency`, `Pottassium Deficiency`, `Spotted Wilt Virus`) — this is what `load_classes()["mapping"]["tomato_village"]["classes"]` returns from Task 4 onward.

- [ ] **Step 1: Edit the canonical vocabulary block**

In `tomanet/configs/classes.yaml`, find:
```yaml
  16: leaf_blight
  17: verticillium_wilt
```
Replace with:
```yaml
  16: leaf_blight
  17: verticillium_wilt
  18: magnesium_deficiency
  19: nitrogen_deficiency
  20: potassium_deficiency
```

- [ ] **Step 2: Replace the wrong `tomato_village` mapping**

Find:
```yaml
  tomato_village:
    verified: false
    classes:
      healthy: healthy
      early_blight: early_blight
      late_blight: late_blight
      leaf_mold: leaf_mold
      leaf_miner: leaf_miner
      spotted_wilt_virus: spotted_wilt_virus
      nutrition_deficiency: nutrient_deficiency
```
Replace with:
```yaml
  tomato_village:
    verified: true            # checked against Variant-c(Object Detection)/Varient-C Labels.txt
    variant: "Variant-c(Object Detection)"
    classes:
      Early_blight: early_blight
      Healthy: healthy
      Late_blight: late_blight
      "Leaf Miner": leaf_miner
      "Magnesium Deficiency": magnesium_deficiency
      "Nitrogen Deficiency": nitrogen_deficiency
      "Pottassium Deficiency": potassium_deficiency   # dataset's own typo, kept verbatim as the key
      "Spotted Wilt Virus": spotted_wilt_virus
```

- [ ] **Step 3: Verify the YAML loads and has the expected shape**

Run:
```bash
cd tomanet
.venv/bin/python -c "
import yaml
d = yaml.safe_load(open('configs/classes.yaml'))
assert d['canonical'][20] == 'potassium_deficiency'
m = d['mapping']['tomato_village']['classes']
assert m['Early_blight'] == 'early_blight'
assert m['Pottassium Deficiency'] == 'potassium_deficiency'
assert len(m) == 8
print('OK', len(d['canonical']), 'canonical classes,', len(m), 'tomato_village classes')
"
```
Expected: `OK 21 canonical classes, 8 tomato_village classes`

- [ ] **Step 4: Commit**

```bash
git add configs/classes.yaml
git commit -m "$(cat <<'EOF'
Add per-nutrient canonical classes and fix tomato_village mapping

The old mapping referenced leaf_mold (not a class in this dataset) and
collapsed three distinct nutrient-deficiency labels into one bucket.
Rewrite it against the dataset's real 8 label strings, verified from
Varient-C Labels.txt on the downloaded dataset.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Extend `verify_setup.py` to check all six variants (C/D x n/s/m)

**Files:**
- Modify: `tomanet/scripts/verify_setup.py:109-143`

**Interfaces:**
- Consumes: `SCALES` dict (`{"n": ..., "s": ..., "m": ...}`) already defined at module level (`verify_setup.py:26`).
- Produces: no new functions consumed elsewhere; this is a leaf verification script.

**Context:** the current `check_ultralytics(scale)` function takes a `scale` parameter but never uses it — it always loads the bare `tomanet-{task}.yaml`, which ultralytics silently resolves to scale `n` with a `WARNING no model scale passed. Assuming scale='n'.` (confirmed by running it). Fix: write scale-suffixed copies exactly like `train_cls.py`'s `build_model` already does, and loop all three scales.

- [ ] **Step 1: Replace `check_ultralytics`**

In `tomanet/scripts/verify_setup.py`, replace:
```python
def check_ultralytics(scale: str) -> None:
    print("== ultralytics integration ==")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics not installed - skipping (pip install -r requirements.txt)")
        return

    from tomanet.register import is_patched, register

    register()
    if not is_patched():
        print("  ! ultralytics source is NOT patched - run scripts/patch_ultralytics.py")
        print("    without it, channel scaling for TomaLayer will be wrong")
        return

    for task, size in (("cls", 224), ("det", 640)):
        path = REPO_ROOT / "configs" / "models" / f"tomanet-{task}.yaml"
        model = YOLO(str(path))
        params = count_params(model.model)
        gflops = count_gflops(model.model, size)
        flops_text = f", {gflops:.2f} GFLOPs" if gflops else ""
        print(f"  tomanet-{task} @ {size}px: {params / 1e6:.2f} M params{flops_text}")
```
with:
```python
def check_ultralytics() -> None:
    print("== ultralytics integration (all variants) ==")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ultralytics not installed - skipping (pip install -r requirements.txt)")
        return

    from tomanet.register import is_patched, register

    register()
    if not is_patched():
        print("  ! ultralytics source is NOT patched - run scripts/patch_ultralytics.py")
        print("    without it, channel scaling for TomaLayer will be wrong")
        return

    for task, size in (("cls", 224), ("det", 640)):
        for scale in SCALES:
            src = REPO_ROOT / "configs" / "models" / f"tomanet-{task}.yaml"
            scaled = src.parent / f"tomanet-{task}{scale}.yaml"
            scaled.write_text(src.read_text(), encoding="utf-8")
            model = YOLO(str(scaled))
            params = count_params(model.model)
            gflops = count_gflops(model.model, size)
            flops_text = f", {gflops:.2f} GFLOPs" if gflops else ""
            print(f"  tomanet-{task}-{scale} @ {size}px: {params / 1e6:.2f} M params{flops_text}")
```

- [ ] **Step 2: Update the call site in `main()`**

Replace:
```python
    check_standalone(args.scale)
    print()
    check_reparam_fusion()
    print()
    check_ultralytics(args.scale)
```
with:
```python
    check_standalone(args.scale)
    print()
    check_reparam_fusion()
    print()
    check_ultralytics()
```
(`--scale` still controls `check_standalone`, which profiles one scale of the standalone trunk by design; `check_ultralytics` now always checks all three.)

- [ ] **Step 3: Run it and confirm all six variants print with distinct params/GFLOPs**

```bash
cd tomanet
.venv/bin/python scripts/verify_setup.py
```
Expected: no `WARNING no model scale passed` lines, and six distinct lines under `== ultralytics integration (all variants) ==`, e.g.:
```
  tomanet-cls-n @ 224px: 1.32 M params, 0.37 GFLOPs
  tomanet-cls-s @ 224px: 3.xx M params, x.xx GFLOPs
  tomanet-cls-m @ 224px: x.xx M params, x.xx GFLOPs
  tomanet-det-n @ 640px: 2.92 M params, 8.58 GFLOPs
  tomanet-det-s @ 640px: x.xx M params, x.xx GFLOPs
  tomanet-det-m @ 640px: x.xx M params, x.xx GFLOPs
```
and the fusion check still prints `[OK]`.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_setup.py
git commit -m "$(cat <<'EOF'
Check all six TomaNet variants in verify_setup.py, not just scale n

check_ultralytics accepted a --scale argument but never used it - it
always built the unsuffixed YAML, which ultralytics silently resolves
to scale n. Write scale-suffixed copies like train_cls.py already
does, and loop n/s/m for both cls and det.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Add grouped split + Tomato-Village adapter as pure, unit-tested functions

**Files:**
- Modify: `tomanet/scripts/prepare_data.py` (add functions, no wiring into `main()` yet)
- Modify: `tomanet/tests/test_modules.py` (add tests)

**Interfaces:**
- Produces:
  - `group_key(path: Path) -> str` — strips a trailing `_aug\d+` suffix from the filename stem.
  - `grouped_stratified_split(items: list[tuple[Path, list[tuple[str, float, float, float, float]]]], ratios: list[float], seed: int) -> dict[Path, tuple[str, list[tuple[str, float, float, float, float]]]]` — maps each image path to `(split, boxes)`; guarantees every path sharing a `group_key` gets the same split.
  - `adapt_tomato_village_det(root: Path, mapping: dict) -> list[tuple[Path, list[tuple[str, float, float, float, float]]]]` — yields `(image_path, boxes)` where each box is `(canonical_class_name, cx, cy, w, h)`.
- Consumes: `images_under()` (already defined at `prepare_data.py:49`), `Counter`/`defaultdict` (already imported), `random` (already imported).
- Note: boxes are keyed by canonical **class name string** (e.g. `"early_blight"`), not a numeric id — matching how every existing classification adapter already represents classes (see Global Constraints).

- [ ] **Step 1: Add `import re` to `prepare_data.py`**

At the top of `tomanet/scripts/prepare_data.py`, in the stdlib import block:
```python
import argparse
import json
import random
import re
import shutil
import sys
```

- [ ] **Step 2: Write the failing tests**

In `tomanet/tests/test_modules.py`, add near the top (after the existing `sys.path.insert` line):
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prepare_data  # noqa: E402
```

Then add these test functions (anywhere after the existing tests, before the `if __name__ == "__main__":` block):
```python
def test_group_key_strips_aug_suffix():
    assert prepare_data.group_key(Path("IMG123_aug3.jpg")) == "IMG123"
    assert prepare_data.group_key(Path("IMG123_aug10.jpg")) == "IMG123"
    assert prepare_data.group_key(Path("IMG123.jpg")) == "IMG123"


def test_grouped_stratified_split_keeps_augmented_copies_together():
    items = []
    for i in range(20):
        base = f"IMG{i}"
        cls = "early_blight" if i % 2 == 0 else "healthy"
        box = (cls, 0.5, 0.5, 0.2, 0.2)
        items.append((Path(f"{base}.jpg"), [box]))
        items.append((Path(f"{base}_aug1.jpg"), [box]))
        items.append((Path(f"{base}_aug2.jpg"), [box]))

    assignment = prepare_data.grouped_stratified_split(items, ratios=[0.6, 0.2, 0.2], seed=0)

    groups = defaultdict(set)
    for path, (split, _boxes) in assignment.items():
        groups[prepare_data.group_key(path)].add(split)

    assert all(len(splits) == 1 for splits in groups.values())
    # sanity: all three splits actually got used with 20 groups at these ratios
    used_splits = {split for split, _ in assignment.values()}
    assert used_splits == {"train", "val", "test"}


def test_adapt_tomato_village_det_parses_yolo_labels():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "tomato_village"
        base = root / "Tomato-Village-main" / "Variant-c(Object Detection)"
        (base / "train" / "images").mkdir(parents=True)
        (base / "train" / "yolo").mkdir(parents=True)
        (base / "val" / "images").mkdir(parents=True)
        (base / "val" / "yolo").mkdir(parents=True)
        (base / "Varient-C Labels.txt").write_text(
            "0 : 'Early_blight'\n1 : 'Healthy'\n", encoding="utf-8"
        )
        (base / "train" / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff")
        (base / "train" / "yolo" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (base / "val" / "images" / "b.jpg").write_bytes(b"\xff\xd8\xff")
        (base / "val" / "yolo" / "b.txt").write_text("1 0.4 0.4 0.1 0.1\n", encoding="utf-8")

        mapping = {"Early_blight": "early_blight", "Healthy": "healthy"}
        items = prepare_data.adapt_tomato_village_det(root, mapping)

    items_by_name = {p.name: boxes for p, boxes in items}
    assert items_by_name["a.jpg"] == [("early_blight", 0.5, 0.5, 0.2, 0.2)]
    assert items_by_name["b.jpg"] == [("healthy", 0.4, 0.4, 0.1, 0.1)]
```

Also add the two needed imports at the top of `test_modules.py` (next to the existing `import sys` / `from pathlib import Path`):
```python
import tempfile
from collections import defaultdict
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
cd tomanet
.venv/bin/python -m pytest tests/test_modules.py -k "group_key or grouped_stratified or adapt_tomato_village" -v
```
Expected: FAIL — `AttributeError: module 'prepare_data' has no attribute 'group_key'` (and similarly for the other two).

- [ ] **Step 4: Implement `group_key`**

In `tomanet/scripts/prepare_data.py`, add near the top-level constants (after `SPLITS = ("train", "val", "test")`):
```python
_AUG_SUFFIX = re.compile(r"_aug\d+$")


def group_key(path: Path) -> str:
    """Base identity of a file, stripping this dataset's '_aug<N>' offline-augmentation suffix.

    Used to keep every augmented copy of one physical photo in the same split - phash
    dedup does not catch these (measured Hamming distance 26-36/64), so grouping by name
    is the only reliable way to prevent train/val/test leakage.
    """
    return _AUG_SUFFIX.sub("", path.stem)
```

- [ ] **Step 5: Implement `grouped_stratified_split`**

Add after the existing `stratified_split` function:
```python
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
```

- [ ] **Step 6: Implement `adapt_tomato_village_det`**

Add after `adapt_imagefolder` (near the other adapters), and register it in a new registry:
```python
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


DET_ADAPTERS = {"tomato_village": adapt_tomato_village_det}
```
Place the `DET_ADAPTERS = {...}` line right after the existing `ADAPTERS = {...}` block for discoverability.

- [ ] **Step 7: Run the tests to verify they pass**

```bash
cd tomanet
.venv/bin/python -m pytest tests/test_modules.py -k "group_key or grouped_stratified or adapt_tomato_village" -v
```
Expected: 3 passed.

- [ ] **Step 8: Run the full test suite to confirm no regression**

```bash
cd tomanet
.venv/bin/python -m pytest tests/test_modules.py -v
```
Expected: all tests pass (the 3 new + the existing ones).

- [ ] **Step 9: Commit**

```bash
git add scripts/prepare_data.py tests/test_modules.py
git commit -m "$(cat <<'EOF'
Add grouped split and tomato_village detection adapter

group_key/grouped_stratified_split split by base photo identity rather
than by image, so this dataset's baked-in offline augmentation can't
leak a photo's variants across train/val/test - phash dedup measurably
does not catch these (Hamming 26-36/64 between a photo and its aug
copies). adapt_tomato_village_det parses the source's YOLO-format
labels and translates through the canonical class mapping. Not yet
wired into prepare_data.py's CLI - that's the next task.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire `--task det` into `prepare_data.py`'s CLI, run against real data

**Files:**
- Modify: `tomanet/scripts/prepare_data.py` (add `--task`, `materialise_det`, `report_det`, branch in `main()`)
- Modify: `tomanet/configs/datasets.yaml` (document the leak, for future readers)
- Modify: `tomanet/README.md` (document the leak, next to Taiwan's existing entry)

**Interfaces:**
- Consumes: `DET_ADAPTERS`, `grouped_stratified_split`, `group_key` from Task 3; `load_classes()` (existing, `prepare_data.py:45`).
- Produces: `data/processed/tomato_village_det/{train,val,test}/{images,labels}/*`, `data/processed/tomato_village_det/data.yaml`, `data/processed/tomato_village_det/split.json` — this `data.yaml` path is what `train_det.py` (Task 5) points ultralytics at.

**Prerequisite:** `tomato_village` must already be downloaded (it is — `data/raw/tomato_village/Tomato-Village-main/` exists on this machine, 3.9G).

- [ ] **Step 1: Add `materialise_det`**

In `tomanet/scripts/prepare_data.py`, add after `materialise`:
```python
def materialise_det(assignment, out_dir: Path, copy: bool) -> list[str]:
    if out_dir.exists():
        shutil.rmtree(out_dir)

    class_names = sorted({name for _, boxes in assignment.values() for name, *_ in boxes})
    name_to_id = {name: i for i, name in enumerate(class_names)}

    for path, (split, boxes) in assignment.items():
        images_dir = out_dir / split / "images"
        labels_dir = out_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{abs(hash(str(path))) % 10**8}_{path.stem}"
        image_target = images_dir / f"{stem}{path.suffix}"
        if copy:
            shutil.copy2(path, image_target)
        else:
            image_target.symlink_to(path.resolve())

        lines = [f"{name_to_id[name]} {cx} {cy} {w} {h}" for name, cx, cy, w, h in boxes]
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    return class_names
```

- [ ] **Step 2: Add `report_det`**

Add after `report`:
```python
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
```

- [ ] **Step 3: Add `--task` and branch `main()`**

In `tomanet/scripts/prepare_data.py`, in the argparse block, add (next to `--dataset`):
```python
    parser.add_argument("--task", default="cls", choices=["cls", "det"],
                        help="cls: ImageFolder tree (default). det: YOLO-format boxes.")
```

Then restructure the body of `main()` from this point:
```python
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
```
into:
```python
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
```
This preserves the original error precedence exactly: for whichever task is active, the
adapter-membership check fires before the root/mapping checks, same as it did before this
change (an earlier draft of this task moved that check to fire after root/mapping checks
for the cls path, silently changing which error an unknown `--dataset` produces - that
would have been a real, if minor, regression). The rest of `main()` (from
`items = ADAPTERS[args.dataset](root, mapping)` onward, i.e. the existing classification
path) is unchanged.

Also update the `--list` block just above (it currently checks only `ADAPTERS`) — leave it as-is; it's cls-specific and `--task det --list` isn't a combination this plan needs to support (YAGNI).

- [ ] **Step 4: Run against the real downloaded dataset**

```bash
cd tomanet
.venv/bin/python scripts/prepare_data.py --dataset tomato_village --task det --copy
```
Expected: prints image/box/class counts (roughly 14,368 images, 8 classes), a per-split table, a per-class table, and writes `data/processed/tomato_village_det/`.

- [ ] **Step 5: Verify zero leakage on the real data (not just the synthetic test)**

```bash
cd tomanet
.venv/bin/python -c "
import json, re
from collections import defaultdict
from pathlib import Path

split_json = json.loads(Path('data/processed/tomato_village_det/split.json').read_text())
groups = defaultdict(set)
for path_str, split in split_json['splits'].items():
    base = re.sub(r'_aug\d+\$', '', Path(path_str).stem)
    groups[base].add(split)

leaking = {k: v for k, v in groups.items() if len(v) > 1}
assert not leaking, f'{len(leaking)} base photos leak across splits: {list(leaking)[:5]}'
print(f'OK: {len(groups)} base photos, zero leak across splits')
"
```
Expected: `OK: <N> base photos, zero leak across splits` with no assertion error.

- [ ] **Step 6: Confirm `data.yaml` is valid and matches the materialised tree**

```bash
cd tomanet
.venv/bin/python -c "
import yaml
from pathlib import Path
d = yaml.safe_load(Path('data/processed/tomato_village_det/data.yaml').read_text())
assert len(d['names']) == 8, d['names']
for split in ('train', 'val', 'test'):
    img_dir = Path(d['path']) / d[split]
    label_dir = img_dir.parent / 'labels'
    imgs = list(img_dir.iterdir())
    labels = list(label_dir.iterdir())
    assert len(imgs) == len(labels), (split, len(imgs), len(labels))
    print(split, len(imgs), 'images')
print('data.yaml OK:', d['names'])
"
```
Expected: three split counts printed (train the largest, val/test smaller — exact numbers vary because grouping splits by *group* count, not image count) and `data.yaml OK: {0: 'early_blight', ...}` with 8 entries.

- [ ] **Step 7: Document the leak finding in `configs/datasets.yaml`**

In `tomanet/configs/datasets.yaml`, find the `tomato_village:` entry's `caveats:` list:
```yaml
    caveats:
      - "Single region (Rajasthan) - do not treat as representative of all Indian fields."
```
Replace with:
```yaml
    caveats:
      - "Single region (Rajasthan) - do not treat as representative of all Indian fields."
      - "Variant-c's shipped train/val split leaks: every one of val's 1,484 base photos
         also has offline-augmented siblings ('_aug1'..'_aug7') sitting in train. phash
         dedup does not catch this (measured Hamming distance 26-36/64). prepare_data.py
         --task det re-splits by base-filename group instead of trusting the shipped split."
```

- [ ] **Step 8: Document it in `README.md`'s "Verified findings so far"**

In `tomanet/README.md`, find:
```markdown
- **TomatoEbola's Zenodo record (13324917) is access-restricted**, so it is registered as
  `manual`.
```
Replace with:
```markdown
- **TomatoEbola's Zenodo record (13324917) is access-restricted**, so it is registered as
  `manual`.
- **Tomato-Village's shipped train/val split leaks by construction.** Its offline
  augmentation means every one of val's 1,484 base photos has augmented siblings in
  train; phash dedup does not catch it (Hamming 26-36/64, threshold is 6).
  `prepare_data.py --task det` re-splits by base-filename group instead.
```

- [ ] **Step 9: Commit**

```bash
git add scripts/prepare_data.py configs/datasets.yaml README.md
git commit -m "$(cat <<'EOF'
Wire --task det into prepare_data.py, verify against real data

Ran against the downloaded tomato_village dataset: 14,368 images / 8
classes materialise into data/processed/tomato_village_det/ with zero
base-photo leakage across splits (verified programmatically, not just
via the synthetic unit test). Documents the dataset's split-leak
finding in datasets.yaml and README.md for future readers.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
Note: `data/processed/` and `data/raw/` should already be gitignored (same as the existing `taiwan_cls` output) - if `git status` shows the materialised dataset files as untracked-and-about-to-be-added, stop and check `.gitignore` before committing; don't commit dataset contents.

---

### Task 5: Create `scripts/train_det.py`, smoke-test against real data

**Files:**
- Create: `tomanet/scripts/train_det.py`
- Modify: `tomanet/RUNBOOK.md` (add a detection section)

**Interfaces:**
- Consumes: `tomanet.register.{is_patched, register}` (existing), `configs/models/tomanet-det.yaml` (existing), `data/processed/tomato_village_det/data.yaml` (Task 4).
- Produces: `runs/detect/<name>/weights/best.pt`, `runs/detect/<name>/run_args.json` — same conventions as `train_cls.py`. Calls `visualize.visualize_detection(...)` (Task 6) at the end, but this task's own test run uses `--no-viz` since that function doesn't exist yet.

- [ ] **Step 1: Write `scripts/train_det.py`**

```python
#!/usr/bin/env python3
"""Train a detection model (TomaNet-D or any baseline) on a prepared dataset.

    # smoke test - proves the pipeline works, ~2 min on a 12 GB GPU
    python scripts/train_det.py --data tomato_village --model tomanet --scale n --epochs 5 --smoke

    # real run
    python scripts/train_det.py --data tomato_village --model tomanet --scale n --epochs 100
    python scripts/train_det.py --data tomato_village --model yolo11n --epochs 100

Always writes verification visualizations (see scripts/visualize.py):
  train_batch*.jpg     what the model actually sees after augmentation
  results.png          loss and mAP curves
  confusion_matrix.png normalised confusion matrix
  predictions.png      test images with predicted vs ground-truth boxes
  metrics.json          mAP50, mAP50-95, per-class AP/P/R

Check `train_batch0.jpg` first. If the augmented boxes look wrong, nothing else matters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PROCESSED = REPO_ROOT / "data" / "processed"

# Augmentation presets.
#
# Unlike classification, rotating a detection image inflates its axis-aligned box, so
# `degrees=0.0` here (classification's --aug leaf uses 15.0). Mosaic/mixup are standard
# ultralytics detection augmentations with no leaf-specific reason to disable them.
AUG_PRESETS = {
    "leaf": dict(
        auto_augment=None, erasing=0.0,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=0.0, translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.3,
        mosaic=1.0, mixup=0.0, copy_paste=0.0,
    ),
    "ultralytics": dict(),   # library defaults, for the ablation
    "none": dict(
        auto_augment=None, erasing=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, fliplr=0.0, flipud=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
    ),
}


def build_model(name: str, scale: str, nc: int):
    from tomanet.register import is_patched, register
    from ultralytics import YOLO

    if name != "tomanet":
        return YOLO(f"{name}.pt" if not name.endswith(".yaml") else name)

    register()
    if not is_patched():
        sys.exit(
            "ultralytics is not patched - run `python scripts/patch_ultralytics.py`.\n"
            "Without it TomaLayer channels are not width-scaled and the model is wrong."
        )

    src = REPO_ROOT / "configs" / "models" / "tomanet-det.yaml"
    scaled = src.parent / f"tomanet-det{scale}.yaml"
    scaled.write_text(src.read_text().replace("nc: 10", f"nc: {nc}"), encoding="utf-8")
    return YOLO(str(scaled))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="prepared dataset name, e.g. tomato_village")
    parser.add_argument("--model", default="tomanet", help="tomanet | yolo11n | ...")
    parser.add_argument("--scale", default="n", choices=["n", "s", "m"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="16 fits 12 GB at 640px scale n")
    parser.add_argument("--device", default="0", help="'0' for GPU 0, 'cpu' for CPU")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=1e-3, help="AdamW initial LR")
    parser.add_argument("--patience", type=int, default=15, help="early-stop on val")
    parser.add_argument("--name", default=None, help="run name (default: auto)")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run to verify the pipeline: few epochs, small images")
    parser.add_argument("--aug", default="leaf", choices=["leaf", "ultralytics", "none"],
                        help="augmentation preset (default: leaf - see AUG_PRESETS)")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    data_dir = PROCESSED / f"{args.data}_det"
    data_yaml_path = data_dir / "data.yaml"
    if not data_yaml_path.exists():
        sys.exit(
            f"{data_yaml_path} not found - run: "
            f"python scripts/prepare_data.py --dataset {args.data} --task det"
        )

    data_cfg = yaml.safe_load(data_yaml_path.read_text())
    names = data_cfg["names"]
    classes = [names[i] for i in sorted(names)]
    nc = len(classes)
    print(f"dataset {args.data}: {nc} classes {classes}")

    if args.smoke:
        args.epochs = min(args.epochs, 5)
        args.imgsz = 320
        args.batch = min(args.batch, 8)
        print("SMOKE MODE: 5 epochs at 320px - checking the pipeline, not the accuracy")

    run_name = args.name or f"{args.data}_{args.model}{args.scale if args.model=='tomanet' else ''}"
    if args.smoke:
        run_name += "_smoke"

    print(f"augmentation preset: {args.aug}")
    model = build_model(args.model, args.scale, nc)

    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.05,
        warmup_epochs=3,
        patience=args.patience,
        project=str(REPO_ROOT / "runs" / "detect"),
        name=run_name,
        exist_ok=True,
        plots=True,
        val=True,
        **AUG_PRESETS[args.aug],
    )

    save_dir = Path(results.save_dir)
    (save_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"\nweights: {save_dir / 'weights' / 'best.pt'}")

    if not args.no_viz:
        from scripts.visualize import visualize_detection  # noqa: PLC0415

        visualize_detection(
            weights=save_dir / "weights" / "best.pt",
            data_yaml=data_yaml_path,
            out_dir=save_dir,
            split="test",
            imgsz=args.imgsz,
            device=args.device,
        )

    print(f"\nInspect these before trusting any number:")
    for artefact in ("train_batch0.jpg", "results.png", "confusion_matrix_normalized.png",
                     "PR_curve.png"):
        path = save_dir / artefact
        print(f"  {'OK ' if path.exists() else '(missing) '}{path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with `--no-viz` (visualize_detection doesn't exist until Task 6)**

```bash
cd tomanet
.venv/bin/python scripts/train_det.py --data tomato_village --model tomanet --scale n \
    --epochs 1 --smoke --device cpu --no-viz
```
Expected: prints `dataset tomato_village: 8 classes [...]`, `SMOKE MODE: ...`, trains 1 epoch (capped by `--smoke` at 5 max, but `--epochs 1` wins since it's the min), prints a `weights: .../best.pt` line, and the final artefact checklist shows `OK` for `train_batch0.jpg` and `results.png` at minimum (ultralytics writes these regardless of our own `--no-viz`).

- [ ] **Step 3: Add a detection section to `RUNBOOK.md`**

In `tomanet/RUNBOOK.md`, find the `## Not built yet` section:
```markdown
## Not built yet

`train_det.py`, cross-domain matrix (E3), SSL pre-training (E4), anomaly/open-set (E5),
XAI metrics (E6), efficiency benchmark (E7), results→LaTeX aggregator. Detection also
needs a box-format adapter in `prepare_data.py`, which must be written against
Tomato-Village's actual directory tree once it is downloaded.
```
Replace with:
```markdown
## Detection

```bash
python scripts/download_datasets.py --only tomato_village   # ~1.6 GB compressed
python scripts/prepare_data.py --dataset tomato_village --task det --copy
python scripts/train_det.py --data tomato_village --model tomanet --scale n \
    --epochs 5 --smoke --device 0
```

Tomato-Village's own train/val split leaks (every val base photo has augmented siblings
in train) - `prepare_data.py --task det` re-splits by base-filename group instead of
trusting the shipped one; see `configs/datasets.yaml`'s `tomato_village` caveats.

Full run:
```bash
python scripts/train_det.py --data tomato_village --model tomanet --scale n \
    --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8 --seed 0
```

## Not built yet

Cross-domain matrix (E3), SSL pre-training (E4), anomaly/open-set (E5), XAI metrics (E6),
efficiency benchmark (E7), results→LaTeX aggregator.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train_det.py RUNBOOK.md
git commit -m "$(cat <<'EOF'
Add train_det.py, mirroring train_cls.py's conventions

Same flag surface and run-artefact conventions as train_cls.py
(run_args.json, weights/best.pt, artefact checklist), sized for
detection: 640px/batch 16 default, degrees=0 in the leaf augmentation
preset (rotation inflates axis-aligned boxes, unlike classification).
Smoke-tested one real CPU epoch against tomato_village. Visualization
wiring uses --no-viz for now - scripts/visualize.py's
visualize_detection lands in the next task.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Add `visualize_detection` to `visualize.py`, enable it in `train_det.py`

**Files:**
- Modify: `tomanet/scripts/visualize.py` (add `visualize_detection`, add `--task` to its own CLI)
- Modify: `tomanet/scripts/train_det.py:167-176` (remove the temporary no-op path — the import already points at the right place, no change needed there; this task just makes it real)

**Interfaces:**
- Produces: `visualize_detection(weights: Path, data_yaml: Path, out_dir: Path, split: str = "test", imgsz: int = 640, device: str = "0") -> dict` — returns the same metrics dict it writes to `metrics.json`.
- Consumes: ultralytics' `model.val()`, whose returned `DetMetrics` object exposes `.results_dict` (overall `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`) and `.summary()` (list of per-class dicts with `Class`, `Images`, `Instances`, `Box-P`, `Box-R`, `Box-F1`, `mAP50`, `mAP50-95`) — confirmed against the installed ultralytics 8.4.137 source.

**Design note (simplification from the spec, disclosed here rather than silently deviating):** the spec called for a hand-built `plot_detection_predictions` overlay. Ultralytics' own `model.val(..., plots=True)` already writes `confusion_matrix(_normalized).png`, `PR_curve.png`, `F1_curve.png`, and predicted-vs-ground-truth box images (`val_batch*_pred.jpg` / `val_batch*_labels.jpg`) — hand-rolling IoU matching to reproduce what an already-installed dependency does for free would be needless custom code (ponytail rung 5: an already-installed dependency solves it). `visualize_detection` delegates the plotting to `model.val(plots=True)` and only adds the `metrics.json` extraction, matching the classification side's `metrics.json` convention. If a more tailored side-by-side prediction plot turns out to be needed later, it's a small addition on top of this, not a redesign.

- [ ] **Step 1: Add `visualize_detection` to `scripts/visualize.py`**

Add after `visualize_classification`:
```python
def visualize_detection(weights: Path, data_yaml: Path, out_dir: Path,
                        split: str = "test", imgsz: int = 640, device: str = "0") -> dict:
    weights, data_yaml, out_dir = Path(weights), Path(data_yaml), Path(out_dir)
    if not weights.exists():
        print(f"  ! no weights at {weights} - skipping visualizations")
        return {}

    model = _load_model(weights)
    print(f"  validating on {split} split")

    val_results = model.val(
        data=str(data_yaml), split=split, imgsz=imgsz, device=device,
        plots=True, project=str(out_dir.parent), name=out_dir.name, exist_ok=True,
        verbose=False,
    )

    overall = val_results.results_dict
    per_class = val_results.summary()
    metrics = {
        "split": split,
        "n_images": int(sum(row["Images"] for row in per_class)) if per_class else 0,
        "precision": overall.get("metrics/precision(B)", 0.0),
        "recall": overall.get("metrics/recall(B)", 0.0),
        "map50": overall.get("metrics/mAP50(B)", 0.0),
        "map50_95": overall.get("metrics/mAP50-95(B)", 0.0),
        "per_class": {row["Class"]: row for row in per_class},
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  mAP50 {metrics['map50']:.4f} | mAP50-95 {metrics['map50_95']:.4f} "
          f"| precision {metrics['precision']:.4f} | recall {metrics['recall']:.4f}")
    return metrics
```

- [ ] **Step 2: Add `--task` to `visualize.py`'s own CLI `main()`**

Replace:
```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="prepared dataset name, e.g. taiwan")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    weights = Path(args.weights)
    visualize_classification(
        weights=weights,
        data_dir=REPO_ROOT / "data" / "processed" / f"{args.data}_cls",
        out_dir=Path(args.out) if args.out else weights.parent.parent,
        split=args.split, imgsz=args.imgsz, device=args.device,
    )
```
with:
```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, help="prepared dataset name, e.g. taiwan")
    parser.add_argument("--task", default="cls", choices=["cls", "det"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    weights = Path(args.weights)
    out_dir = Path(args.out) if args.out else weights.parent.parent
    if args.task == "det":
        visualize_detection(
            weights=weights,
            data_yaml=REPO_ROOT / "data" / "processed" / f"{args.data}_det" / "data.yaml",
            out_dir=out_dir,
            split=args.split, imgsz=args.imgsz or 640, device=args.device,
        )
    else:
        visualize_classification(
            weights=weights,
            data_dir=REPO_ROOT / "data" / "processed" / f"{args.data}_cls",
            out_dir=out_dir,
            split=args.split, imgsz=args.imgsz or 224, device=args.device,
        )
```

- [ ] **Step 3: Test `visualize_detection` standalone, against the weights from Task 5's smoke run**

```bash
cd tomanet
.venv/bin/python scripts/visualize.py \
    --weights runs/detect/tomato_village_tomanetn_smoke/weights/best.pt \
    --data tomato_village --task det --split test --device cpu
```
Expected: prints `validating on test split`, then a line like `mAP50 0.0123 | mAP50-95 0.0045 | precision ... | recall ...` (numbers will be near-random after 1 epoch - that's expected and fine, this is a pipeline check not an accuracy check), and writes `metrics.json` plus ultralytics' own plot files into `runs/detect/tomato_village_tomanetn_smoke/`.

- [ ] **Step 4: Confirm `metrics.json` has the expected shape**

```bash
cd tomanet
.venv/bin/python -c "
import json
m = json.loads(open('runs/detect/tomato_village_tomanetn_smoke/metrics.json').read())
assert set(m) >= {'split', 'n_images', 'precision', 'recall', 'map50', 'map50_95', 'per_class'}
assert 0.0 <= m['map50'] <= 1.0
print('OK', m['split'], m['n_images'], 'images', len(m['per_class']), 'classes')
"
```
Expected: `OK test <N> images 8 classes` (or fewer than 8 if some classes had zero instances in this tiny smoke run - ultralytics' `summary()` only lists classes with predictions/instances).

- [ ] **Step 5: Re-run the full `train_det.py` smoke test WITHOUT `--no-viz`, confirming the end-to-end path**

```bash
cd tomanet
.venv/bin/python scripts/train_det.py --data tomato_village --model tomanet --scale n \
    --epochs 1 --smoke --device cpu
```
Expected: same as Task 5 Step 2, plus now also prints the `visualize_detection` output (`validating on test split`, `mAP50 ...`) and `runs/detect/tomato_village_tomanetn_smoke/metrics.json` exists.

- [ ] **Step 6: Commit**

```bash
git add scripts/visualize.py
git commit -m "$(cat <<'EOF'
Add visualize_detection, enable it in train_det.py

Delegates plotting to ultralytics' own model.val(plots=True) - it
already writes confusion matrix, PR/F1 curves and predicted-vs-ground-
truth box images, so hand-rolling IoU matching would just reproduce
what an already-installed dependency does. Extracts DetMetrics into
the same metrics.json convention train_cls.py's visualize_classification
already uses. Verified end-to-end against a real 1-epoch CPU run.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Final full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the unit test suite**

```bash
cd tomanet
.venv/bin/python -m pytest tests/test_modules.py -v
```
Expected: all tests pass, including the 3 new ones from Task 3.

- [ ] **Step 2: Run `verify_setup.py` end to end**

```bash
cd tomanet
.venv/bin/python scripts/verify_setup.py
```
Expected: standalone trunk section, fusion `[OK]`, and all six `tomanet-{cls,det}-{n,s,m}` lines with distinct params/GFLOPs, no `WARNING no model scale passed` lines.

- [ ] **Step 3: Confirm the real-data leakage check still passes (regression guard)**

```bash
cd tomanet
.venv/bin/python -c "
import json, re
from collections import defaultdict
from pathlib import Path
split_json = json.loads(Path('data/processed/tomato_village_det/split.json').read_text())
groups = defaultdict(set)
for path_str, split in split_json['splits'].items():
    base = re.sub(r'_aug\d+\$', '', Path(path_str).stem)
    groups[base].add(split)
leaking = {k: v for k, v in groups.items() if len(v) > 1}
assert not leaking
print('OK, zero leakage,', len(groups), 'base photos')
"
```

- [ ] **Step 4: Confirm `git status` is clean (no stray untracked files from testing)**

```bash
cd /home/devarshi/Desktop/tomato
git status --short
```
Expected: only files touched by this plan's commits are already committed; `tomanet/data/` outputs should not appear (check `.gitignore` covers `data/raw/` and `data/processed/`; if not, that's a pre-existing gap unrelated to this plan - flag it, don't silently commit dataset files).

- [ ] **Step 5: Report to the user**

Summarize: all 6 model variants verified buildable, tomato_village detection data prepared with zero-leakage split (real counts from Step 3), `train_det.py` smoke-tested end to end with real mAP/precision/recall numbers from Task 6 Step 3, RUNBOOK/README/datasets.yaml updated. No further commit needed for this step - it's a status report only.
