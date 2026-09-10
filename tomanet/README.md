# TomaNet

One block family, two architectures: `TomaNet-C` (classification) and `TomaNet-D`
(detection) for tomato leaf disease. Built on Ultralytics so the assigner, losses,
augmentation, mAP evaluation and export come for free.

See `../main.pdf` for the full research plan (audit, datasets, metrics, experiments).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/patch_ultralytics.py       # registers TomaNet modules into ultralytics
python scripts/verify_setup.py            # builds both models, prints params/GFLOPs
```

`patch_ultralytics.py` edits the *installed* ultralytics package. Ultralytics has no
plugin registry, so custom YAML modules must be visible inside `nn/tasks.py`. The patch
is idempotent and must be re-run after every `pip install -U ultralytics`.

## Datasets

```bash
python scripts/download_datasets.py --list                  # show registry
python scripts/download_datasets.py --tier core             # auto-downloadable ones
python scripts/download_datasets.py --only taiwan plantseg
```

Zenodo, Mendeley and GitHub sets download automatically. Kaggle needs
`~/.kaggle/kaggle.json`. Roboflow and a few others are manual - the script prints the URL
and the reason.

**Read `configs/datasets.yaml` before using any of these.** Licences differ (CC0 through
CC BY-NC-ND), Tomato-Village has no stated licence, and most Kaggle "tomato" sets are
re-splits of PlantVillage - training on one and testing on another is *not* a
cross-domain experiment.

Then always deduplicate:

```bash
python scripts/dedup.py --report            # counts, nothing touched
python scripts/dedup.py --write-manifest    # emits data/interim/duplicates.json
```

Nothing is ever deleted; the manifest is an exclusion list the loaders read.

### Verified findings so far

- **Taiwan ships a leaking split.** 619 of 623 duplicate clusters span `data augmentation/`
  and `Preprocessed data/`, and `Preprocessed data/Test/.../Bs1.JPG` reappears as
  `data augmentation/Train/.../Bs1.JPG`. Use `Preprocessed data` only and re-split the
  622 raw images yourself.
- **TomatoEbola's Zenodo record (13324917) is access-restricted**, so it is registered as
  `manual`.
- **Tomato-Village's shipped train/val split leaks by construction.** Its offline
  augmentation means every one of val's 1,484 base photos has augmented siblings in
  train; phash dedup does not catch it (Hamming 26-36/64, threshold is 6).
  `prepare_data.py --task det` re-splits by base-filename group instead.

## Layout

```
configs/
  datasets.yaml        dataset registry: source, licence, classes, caveats
  classes.yaml         canonical class vocabulary + per-dataset mapping
  models/*.yaml        TomaNet-C / TomaNet-D architectures
tomanet/
  modules.py           PConv, CoordAtt, TomaBlock, TomaLayer
  register.py          injects the modules into ultralytics' namespace
scripts/
  patch_ultralytics.py one-time source patch of the installed package
  download_datasets.py registry-driven downloader
  dedup.py             perceptual-hash dedup within and across datasets
  verify_setup.py      smoke test
tests/
  test_modules.py      shape + reparam-fusion equivalence
```
