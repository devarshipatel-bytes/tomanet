#!/usr/bin/env python3
"""Verification visualizations: proof that a model is training correctly.

Run automatically at the end of scripts/train_cls.py, or standalone:

    python scripts/visualize.py --weights runs/classify/taiwan_tomanetn/weights/best.pt \
                                --data taiwan --split test

Produces, in the run directory:
  predictions.png   a grid of test images, green border = correct, red = wrong,
                    captioned "pred (conf) / true"
  per_class.png     precision, recall and F1 per class, with support counts
  metrics.json      accuracy, macro-P/R/F1, per-class values - the numbers that go in
                    the paper, written by code so they cannot be mistyped

What to look for
  - train_batch0.jpg (written by ultralytics): are the augmented images still
    recognisable leaves? Over-aggressive augmentation is the most common silent failure.
  - results.png: train loss falling, val accuracy rising. A flat val curve with falling
    train loss means overfitting; both flat means the LR or the data is wrong.
  - predictions.png: are the mistakes plausible (early vs late blight) or absurd
    (healthy vs mosaic)? Absurd mistakes usually mean a label-mapping bug.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_model(weights: Path):
    from tomanet.register import register
    from ultralytics import YOLO

    register()  # custom modules must exist before torch.load rebuilds the graph
    return YOLO(str(weights))


def _collect(data_dir: Path, split: str) -> list[tuple[Path, str]]:
    root = data_dir / split
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    return [
        (path, class_dir.name)
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir())
        for path in sorted(class_dir.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]


def _predict(model, paths: list[Path], imgsz: int, device: str, names: dict):
    preds, confs = [], []
    for i in range(0, len(paths), 32):
        batch = [str(p) for p in paths[i : i + 32]]
        for result in model.predict(batch, imgsz=imgsz, device=device, verbose=False):
            top = int(result.probs.top1)
            preds.append(names[top])
            confs.append(float(result.probs.top1conf))
    return preds, confs


def plot_predictions(paths, trues, preds, confs, out_path: Path, n: int = 20, seed: int = 0):
    """Grid of sample predictions. Deliberately shows every wrong case first."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    wrong = [i for i in range(len(paths)) if trues[i] != preds[i]]
    right = [i for i in range(len(paths)) if trues[i] == preds[i]]
    rng = random.Random(seed)
    rng.shuffle(right)
    chosen = (wrong[: n // 2] + right)[:n] if wrong else right[:n]

    cols = 5
    rows = max(1, (len(chosen) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, chosen):
        ok = trues[idx] == preds[idx]
        ax.imshow(Image.open(paths[idx]).convert("RGB"))
        ax.set_title(
            f"{preds[idx]} ({confs[idx]:.2f})\ntrue: {trues[idx]}",
            fontsize=8, color="darkgreen" if ok else "darkred",
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("green" if ok else "red")
            spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")

    n_wrong = len(wrong)
    fig.suptitle(
        f"{len(paths)} test images - {n_wrong} wrong ({100*n_wrong/len(paths):.1f}%). "
        f"Mistakes shown first.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(report: dict, classes: list[str], out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(classes))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(classes)), 4.2))
    for offset, key, colour in ((-width, "precision", "#4C72B0"),
                                (0.0, "recall", "#DD8452"),
                                (width, "f1-score", "#55A868")):
        ax.bar(x + offset, [report[c][key] for c in classes], width, label=key, color=colour)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n={int(report[c]['support'])})" for c in classes], rotation=30, ha="right", fontsize=8
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("score")
    ax.set_title("Per-class performance (low bars = where the model actually fails)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def visualize_classification(weights: Path, data_dir: Path, out_dir: Path,
                             split: str = "test", imgsz: int = 224, device: str = "0") -> dict:
    from sklearn.metrics import classification_report, cohen_kappa_score, matthews_corrcoef

    weights, data_dir, out_dir = Path(weights), Path(data_dir), Path(out_dir)
    if not weights.exists():
        print(f"  ! no weights at {weights} - skipping visualizations")
        return {}

    model = _load_model(weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    items = _collect(data_dir, split)
    paths = [p for p, _ in items]
    trues = [c for _, c in items]
    print(f"  visualizing {len(paths)} {split} images")

    preds, confs = _predict(model, paths, imgsz, device, names)
    classes = sorted(set(trues) | set(preds))

    report = classification_report(trues, preds, labels=classes, output_dict=True, zero_division=0)
    metrics = {
        "split": split,
        "n_images": len(paths),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "cohen_kappa": cohen_kappa_score(trues, preds),
        "mcc": matthews_corrcoef(trues, preds),
        "per_class": {c: report[c] for c in classes},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_predictions(paths, trues, preds, confs, out_dir / "predictions.png")
    plot_per_class(report, classes, out_dir / "per_class.png")

    print(f"  accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} "
          f"| kappa {metrics['cohen_kappa']:.4f} | MCC {metrics['mcc']:.4f}")
    return metrics


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


if __name__ == "__main__":
    main()
