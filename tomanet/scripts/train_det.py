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
