# TomaNet detection parity — design spec

Date: 2026-09-09
Status: approved (brainstorming), pending implementation plan

## Problem

TomaNet-C (classification) has a complete pipeline: `prepare_data.py` produces an
ImageFolder tree, `train_cls.py` trains at any scale (n/s/m), `visualize.py` writes
verification plots, and `scripts/verify_setup.py` proves the model builds correctly.

TomaNet-D (detection) has only the model YAML (`configs/models/tomanet-det.yaml`) and a
`verify_setup.py` check that it builds at scale n. There is no data adapter, no training
script, and no downloaded detection dataset. The RUNBOOK explicitly lists this as
"Not built yet" and notes the box-format adapter must be written against a real directory
tree, not guessed.

This spec brings detection to parity with classification: prepare → train → visualize,
for scales n/s/m, mirroring the existing classification pipeline's conventions rather than
inventing new ones.

## Scope

**In scope:**
- One real, auto-downloadable, box-annotated dataset: `tomato_village` (Variant-c, Object
  Detection). Verified via the GitHub API to ship YOLO-format labels already
  (`train|val/{images,pascal_voc,yolo}/`), 8 classes: Early_blight, Healthy, Late_blight,
  Leaf Miner, Magnesium Deficiency, Nitrogen Deficiency, Pottassium Deficiency, Spotted
  Wilt Virus.
- `scripts/prepare_data.py --task det` for this dataset: canonical class remap, phash
  dedup (reusing the existing classification dedup path), re-split (TV ships only
  train/val, no test), `data.yaml` generation for ultralytics' detection task.
- `scripts/train_det.py`: trains `tomanet-det{n,s,m}.yaml` or a YOLO baseline, at 640px,
  with a leaf-appropriate augmentation preset, writing the same kind of run artefacts
  (`run_args.json`, `metrics.json`, plots) as `train_cls.py`.
- `scripts/visualize.py`: add `visualize_detection()` — predicted vs. ground-truth boxes
  on test images, per-class AP bar chart, mAP50/mAP50-95 in `metrics.json`.
- `scripts/verify_setup.py`: extend to build and report all six variants (TomaNet-C and
  TomaNet-D, each at scale n/s/m), not just scale n.
- `configs/classes.yaml`: add 3 canonical classes (`magnesium_deficiency`,
  `nitrogen_deficiency`, `potassium_deficiency`, ids 18-20) and rewrite the
  `tomato_village` mapping to the dataset's real label strings (current mapping is wrong
  — it references `leaf_mold`, which does not exist in this dataset, and a single
  `nutrition_deficiency` key that doesn't match any real label).
- Two new tests in `tests/test_modules.py` against a small synthetic fixture (no download
  required): the box adapter parses YOLO-format labels correctly, and the canonical→
  contiguous class-id remap is a valid bijection.
- Real verification: run `verify_setup.py` (all 6 variants) and one real 1-epoch
  `train_det.py` run against the downloaded `tomato_village` data, on this machine.

**Out of scope** (explicitly deferred, per user's scope answer):
- Cross-domain matrix (E3), SSL pre-training (E4), anomaly/open-set (E5), XAI metrics
  (E6), efficiency benchmark (E7), results→LaTeX aggregator.
- Any other detection dataset (plantdoc's GitHub mirror has no annotation files at all —
  it is classification-only in practice; fieldplant/ccmt/etc. are manual-download only per
  `datasets.yaml` and out of scope here).
- A unified `train.py` for both tasks — rejected in brainstorming: ultralytics' cls vs.
  det argument surfaces genuinely diverge (`erasing`/`auto_augment` are cls-only;
  `mosaic`/`box`/`dfl`/`close_mosaic` are det-only), so a merged script would be a branchy
  file pretending to be one thing. Keeping `train_cls.py` and `train_det.py` as siblings
  with the same conventions (not shared code) is the deliberate choice.

## Data layer

### `configs/classes.yaml`

Add to `canonical`:
```yaml
18: magnesium_deficiency
19: nitrogen_deficiency
20: potassium_deficiency
```

Rewrite the `tomato_village` mapping under `mapping:` to the real label strings (replacing
the current wrong entry):
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
The existing classification-oriented `tomato_village` entry (if the codebase later adds a
classification variant of this dataset) is a separate mapping key or is understood to be
for Variant-a — out of scope here since this spec only touches the detection variant.

### `scripts/prepare_data.py`

Add `--task {cls,det}` (default `cls` — no change to any existing invocation).

New detection adapter registry, parallel to `ADAPTERS`:
```python
DET_ADAPTERS = {"tomato_village": adapt_tomato_village_det}
```

`adapt_tomato_village_det(root, mapping)` yields `(image_path, [(canonical_id, cx, cy, w,
h), ...])` — one entry per image, boxes in normalized YOLO xywh as the source ships them
(confirmed against real files: `class_id cx cy w h` per line, e.g. `3.0 0.395 0.291 0.131
0.102`; class id parsed as `int(float(...))`), class ids translated from the source's 0-7
to this project's canonical ids via `mapping`. Reads the source label id order from
`Varient-C Labels.txt` inside the dataset (not hardcoded), confirmed to read exactly:
`Early_blight, Healthy, Late_blight, Leaf Miner, Magnesium Deficiency, Nitrogen
Deficiency, Pottassium Deficiency, Spotted Wilt Virus`. Pools TV's `train` (11,493 images)
and `val` (2,875 images) directories together — TV ships no test split, and per-image
splitting must be redone here for a consistent 70/15/15 with the rest of the project.

**Verified data-integrity problem, changes the split design from the original draft:**
this dataset bakes in offline augmentation — filenames like `IMG20220323081448.jpg` and
`IMG20220323081448_aug3.jpg` … `_aug7.jpg` are rotated/cropped/color-shifted copies of one
physical photo. Measured on disk: train has 11,493 images over only 1,796 unique base
photos; val has 2,875 images over 1,484 unique base photos; **100% of val's 1,484 base
photos also have augmented siblings sitting in train** — the dataset's own shipped
train/val split leaks by construction, the same failure mode already flagged for Taiwan in
the RUNBOOK, but total rather than partial.

This defeats the dedup path originally planned: perceptual hashing was tested against
`IMG20220323081448` and its 5 `_aug` siblings and measured Hamming distances of 26-36 (out
of 64 bits, default duplicate threshold is 6) — the augmentation changes pixels enough
that `phash` does not see these as duplicates at all. So `drop_duplicates`/`dedup.py` is
**not used** for this adapter; it would silently do nothing about the actual leakage.

Fix: a new `grouped_stratified_split(items, ratios, seed)` in `prepare_data.py`, used only
for this adapter (the existing per-image `stratified_split` remains untouched and keeps
serving all classification adapters). It groups images by base filename with the
`_aug\d+` suffix stripped, then splits whole groups into train/val/test — stratified by
each group's most-frequent box class, ties broken by lowest canonical id for determinism.
No image is dropped; every augmented copy stays as legitimate training signal, but all
copies of one physical photo are guaranteed to land in the same split.

Output layout, mirroring the classification tree:
```
data/processed/tomato_village_det/
    train/{images,labels}/*
    val/{images,labels}/*
    test/{images,labels}/*
    data.yaml       # ultralytics detection format: path, train, val, test, names
    split.json      # same convention as classification: path + split + seed
```
`images/` follows `--copy`'s existing symlink-vs-copy behavior. `labels/*.txt` are always
written fresh (never symlinked) because ids are remapped from source 0-7 to this
project's canonical ids, then remapped again to a **contiguous 0..k-1** range for
ultralytics (canonical ids are sparse: 0, 2, 3, 11, 14, 18, 19, 20). `data.yaml` records
both the contiguous `names:` list ultralytics needs and a `canonical_ids:` list at the
same index, so a later cross-dataset script can map back to canonical space without
re-deriving it.

## Training layer

### `scripts/train_det.py`

Sibling of `train_cls.py`: same flag names and defaults where they overlap (`--data`,
`--model`, `--scale`, `--epochs`, `--device`, `--workers`, `--seed`, `--lr0`,
`--patience`, `--name`, `--smoke`, `--no-viz`), same `build_model` pattern (reusing
`tomanet-det.yaml`, same patched-ultralytics guard), same run-artefact conventions
(`run_args.json` written into `save_dir`, `weights/best.pt` printed, an artefact-existence
checklist printed at the end).

Differences from `train_cls.py`:
- `--imgsz` default 640 (not 224).
- `--batch` default 16 (not 64) — matches the RUNBOOK's 12GB VRAM table for TomaNet-D-n.
- `--smoke` forces 5 epochs at 320px, batch capped at 8.
- `data=str(data_dir / "data.yaml")` (ultralytics detection task needs the yaml, not a
  bare directory).
- Its own `AUG_PRESETS["leaf"]`: `degrees=0.0` (rotating an image inflates its axis-
  aligned box, unlike classification where the whole image is the label), `mosaic=1.0`,
  `mixup=0.0`, `copy_paste=0.0`, same `hsv_*` lighting-drift values as the cls preset for
  consistency. `"ultralytics"` preset stays `dict()` for the ablation, `"none"` disables
  all augmentation same as cls.
- `project=str(REPO_ROOT / "runs" / "detect")` (separate from `runs/classify`).
- Calls `visualize_detection(...)` instead of `visualize_classification(...)`.
- `BASELINES` set uses detection model names: `yolo11n`, `yolo11s`, `yolov8n`, `yolov8s`
  (no `-cls` suffix — these are ultralytics' detection weights).

### `scripts/verify_setup.py`

`check_ultralytics` currently loops `for task, size in (("cls", 224), ("det", 640))` at a
single scale (whatever `--scale` was passed). Change it to loop all three scales for both
tasks and print a table:
```
== ultralytics integration (all variants) ==
  tomanet-cls-n @ 224px: 1.32 M params, 0.37 GFLOPs
  tomanet-cls-s @ 224px: ... 
  tomanet-cls-m @ 224px: ...
  tomanet-det-n @ 640px: 2.92 M params, 8.58 GFLOPs
  tomanet-det-s @ 640px: ...
  tomanet-det-m @ 640px: ...
```
`--scale` flag stays (limits `check_standalone`, which profiles the standalone trunk at
one scale by design) but `check_ultralytics` always checks all three — that's the "all
variants" verification this spec is for.

## Visualization layer

### `scripts/visualize.py`

Add `visualize_detection(weights, data_dir, out_dir, split="test", imgsz=640, device="0")`:
- Runs `model.val(data=data_dir/"data.yaml", split=split, ...)` — ultralytics already
  computes mAP50, mAP50-95, per-class AP, precision, recall; this function extracts those
  into the same `metrics.json` convention as classification (top-level scalar metrics +
  `per_class` dict), so downstream tooling that reads `metrics.json` does not need to
  branch on task type.
- `plot_detection_predictions`: grid of test images with predicted boxes (green, labeled
  `class conf`) overlaid on ground-truth boxes (red outline only), mirroring
  `plot_predictions`'s "wrong cases first" framing — here, images with any false positive
  or missed detection are shown first.
- `plot_per_class` is reused as-is (already takes a generic `report`/`classes` pair); the
  per-class AP dict is shaped to match what it expects.

## Tests

`tests/test_modules.py` gets three new tests, all against an in-memory / tmp_path
fixture — no dataset download required:
1. A small synthetic directory shaped like `train/{images,yolo}/` with 2-3 fake YOLO
   label files is fed to the adapter; assert the returned boxes and remapped class ids
   match what was written.
2. The canonical→contiguous remap: given a sparse canonical id list, assert the resulting
   mapping is a bijection onto `0..k-1` and that decoding a contiguous prediction back via
   `data.yaml`'s recorded `canonical_ids` returns the original canonical id.
3. `grouped_stratified_split`: given a synthetic set of base names each with 2-6 `_augN`
   variants, assert every variant of a base name lands in the same split — i.e. for every
   group, `{split for (path, _) in group}` has exactly one element. This is the test that
   would have caught the leakage found during design verification.

## Verification plan (before claiming this works)

1. `python scripts/verify_setup.py` — all 6 variants build, print params/GFLOPs, fusion
   check still `[OK]`.
2. `python scripts/download_datasets.py --only tomato_village` — done: 3.9 GB extracted to
   `data/raw/tomato_village/Tomato-Village-main/`.
3. `python scripts/prepare_data.py --dataset tomato_village --task det` against the real
   downloaded tree.
4. After prepare, a leakage check: for every base name, assert all its splits are
   identical (the same property tested synthetically in test 3, checked here against the
   real 14,368-image dataset as the final proof the fix works, not just the fixture).
5. `python scripts/train_det.py --data tomato_village --model tomanet --scale n --epochs 1
   --device cpu` — one real epoch, confirms the full path (data.yaml, dataloader, loss,
   mAP computation, visualize_detection) runs end to end.
6. `pytest tests/test_modules.py` — new tests pass.

## Verified during design (resolved, kept for record)

- Zip extraction root is `Tomato-Village-main/` — confirmed on disk.
- `Varient-C Labels.txt` reads cleanly as `<int> : '<Name>'` per line, all 8 lines
  consistent: Early_blight, Healthy, Late_blight, Leaf Miner, Magnesium Deficiency,
  Nitrogen Deficiency, Pottassium Deficiency, Spotted Wilt Virus.
- YOLO label format confirmed: `<class_id:float> <cx> <cy> <w> <h>` per line, normalized.
- train: 11,493 images / 1,796 unique base photos. val: 2,875 images / 1,484 unique base
  photos. 100% base-photo overlap between train and val (see Data layer section above) —
  this is why the adapter does its own grouped split rather than trusting the shipped one.
