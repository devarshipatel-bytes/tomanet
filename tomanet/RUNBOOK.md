# Runbook — 12 GB GPU server

Everything below has been executed end-to-end on CPU; only the batch sizes are tuned
for a 12 GB card.

---

## 0. One-time setup (~10 min)

```bash
cd tomanet
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # makes `tomanet` importable inside ultralytics
python scripts/patch_ultralytics.py    # REQUIRED - see note
python scripts/verify_setup.py
```

Expected output:

```
== standalone trunk (scale n) ==
  224px -> (1, 256, 7, 7)    0.34 GFLOPs
  640px -> (1, 256, 20, 20)  2.73 GFLOPs
  params: 0.97 M
== re-parameterisation ==
  max |unfused - fused| = 2.38e-07   [OK]
== ultralytics integration ==
  tomanet-cls @ 224px: 1.32 M params, 0.37 GFLOPs
  tomanet-det @ 640px: 2.92 M params, 8.58 GFLOPs
```

If the fusion line is not `[OK]`, stop — the model would silently degrade at inference.

**About the patch.** Ultralytics has no plugin registry, and `parse_model` decides channel
injection from frozensets that are local variables. The patch adds our modules there.
It is idempotent, backs the file up, and `--revert`s. **Re-run it after every
`pip install -U ultralytics`** — without it `TomaLayer` channels are not width-scaled and
you train a silently wrong model. `verify_setup.py` fails loudly if it is missing.

```bash
python scripts/patch_ultralytics.py --check    # exit 0 = patched
```

---

## 1. Get data

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py --tier core --dry-run
python scripts/download_datasets.py --only taiwan          # 46 MB, start here
python scripts/download_datasets.py --only plantvillage    # 2.0 GB
```

### Real sizes (measured from the source APIs, 2026-09-01)

| Dataset | Download | Auto? | Notes |
|---|---:|---|---|
| **taiwan** | **45.8 MB** | yes | 622 raw field images, 6 classes. Start here. |
| plantvillage | 2.0 GB | yes | whole repo (all crops); tomato = 18,160 of 54,306 |
| plantdoc | 933 MB | yes | 2,598 images, boxes, ~700 tomato |
| tomato_village | 3.2 GB | yes | field India; cls + multi-label + detection |
| plantseg | 1.6 GB | yes | pixel lesion masks (for XAI / severity) |
| plantvillage_tomato_zenodo | 183 MB | yes | 11k PlantVillage re-split; small stand-in |
| laboro_tomato | 61.5 MB | yes | fruit ripeness, instance masks |
| tuta_tanzania | 2.2 GB | yes | pest, 4,341 images |
| eight_tomato_pests | 66.3 MB | yes | 8 pest species |
| ccmt | ~4 GB | **no** | Mendeley API lists no files — browser download |
| fieldplant | — | **no** | Roboflow, needs API key |
| aiub_bangladesh, daffodil_iu, bangladesh_field_cls, damage_progression, sichuan_greenhouse, tom2024 | — | **no** | same Mendeley API limitation |
| tomato_ebola | — | **no** | Zenodo record is access-restricted |

**Auto-downloadable core total: ~7.8 GB.** Everything with `auto: NO` prints its URL and
the reason; nothing fails silently.

Then always deduplicate:

```bash
python scripts/dedup.py --report
```

---

## 2. Prepare (harmonise labels, dedup, split 70/15/15)

```bash
python scripts/prepare_data.py --dataset taiwan
```

Verified output:

```
622 images, 6 mapped classes
dedup: dropped 1 near-duplicate image(s) in 1 cluster(s)
TOTAL                          432     90     99     621
imbalance ratio (largest/smallest class): 2.3
```

Writes `data/processed/taiwan_cls/{train,val,test}/<class>/` as symlinks (no disk copy)
plus `split.json` recording the seed, so runs are reproducible.

---

## 3. First model — smoke test (~2 min)

**Dataset: Taiwan.** 622 real field images, 6 classes, already downloaded, trains in
minutes. The point is to prove the pipeline, not to get a number.

```bash
python scripts/train_cls.py --data taiwan --model tomanet --scale n \
    --epochs 5 --smoke --device 0
```

Expect ~45–55% accuracy. That is *correct* for 5 epochs from scratch on 432 images.

### Then look at the pictures, in this order

| File | What it proves |
|---|---|
| **`train_batch0.jpg`** | **Check this first.** Lesions visible, colours natural, no black boxes. |
| `results.png` | train loss falling, val accuracy rising |
| `predictions.png` | mistakes plausible (bacterial spot vs black mold), not absurd |
| `confusion_matrix_normalized.png` | errors concentrated in look-alike pairs |
| `per_class.png` | which classes actually fail |
| `metrics.json` | accuracy, macro-F1, kappa, MCC — written by code, never typed |

All in `runs/classify/taiwan_tomanetn_smoke/`.

**Why `train_batch0.jpg` matters.** Ultralytics' default classification augmentation is
wrong for this task and we measured it: `erasing=0.4` blanks a rectangle that often covers
the lesion, and `auto_augment=randaugment` applies solarize/posterize/invert, destroying
the colour that *is* the diagnostic signal. The default preset here is `--aug leaf`, which
removes both and keeps only realistic lighting drift, rotation and flips. Same 3 epochs:
45.5% → 49.5%. Compare yourself with `--aug ultralytics`.

---

## 4. First real run

```bash
# Taiwan, full schedule (~15 min on a 12 GB card)
python scripts/train_cls.py --data taiwan --model tomanet --scale n \
    --epochs 100 --imgsz 224 --batch 128 --device 0 --workers 8 --seed 0

# The reference baseline everyone reports
python scripts/download_datasets.py --only plantvillage
python scripts/prepare_data.py --dataset plantvillage
python scripts/train_cls.py --data plantvillage --model tomanet --scale n \
    --epochs 100 --batch 128 --device 0
```

Expect ~99% on PlantVillage. **That is not a result** — it is saturated in the literature
(99.4–99.96%). The real experiment is what happens on field data and across datasets.

### Baseline comparison (identical splits and recipe)

```bash
for m in yolo11n-cls yolo11s-cls yolov8n-cls; do
  python scripts/train_cls.py --data taiwan --model $m --epochs 100 --batch 128 --device 0
done
```

### 12 GB batch sizes

| Model | imgsz | batch | approx. VRAM |
|---|---|---|---|
| TomaNet-C-n | 224 | 128 | ~4 GB |
| TomaNet-C-s | 224 | 96 | ~6 GB |
| TomaNet-D-n | 640 | 32 | ~8 GB |
| TomaNet-D-s | 640 | 16 | ~10 GB |

Halve `--batch` on CUDA OOM; `--workers 8` suits most servers.

---

## Architecture as built

| | TomaNet-C | TomaNet-D |
|---|---|---|
| Input | 224×224 | 640×640 |
| Params (n) | **1.32 M** | **2.92 M** |
| GFLOPs (n) | **0.37** | **8.58** |
| Head | Conv1×1→1280, GAP, FC | SPPF + PAFPN + anchor-free DFL |
| Backbone | layers 0–8, **byte-identical** (enforced by a test) | same |

Reference points: YOLO11n-cls 1.6 M / 0.5 G; YOLO11n 2.6 M / 6.5 G; YOLOv8n 3.2 M / 8.7 G.
So TomaNet-C is smaller than YOLO11n-cls; TomaNet-D sits between YOLO11n and YOLOv8n.
It is **not** lighter than YOLO11n — the claim to defend is Pareto efficiency plus
cross-domain robustness, and it has to be measured, not asserted.

The neck uses `expansion=1.0` (measured: 10.33 → 8.58 GFLOPs, backbone untouched).
CoordAtt is kept in the neck because it measured at <0.05 M params and 0.00 GFLOPs.

---

## Not built yet

`train_det.py`, cross-domain matrix (E3), SSL pre-training (E4), anomaly/open-set (E5),
XAI metrics (E6), efficiency benchmark (E7), results→LaTeX aggregator. Detection also
needs a box-format adapter in `prepare_data.py`, which must be written against
Tomato-Village's actual directory tree once it is downloaded.
