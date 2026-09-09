#!/usr/bin/env python3
"""Smoke test: build both TomaNet models and report parameters, GFLOPs and shapes.

    python scripts/verify_setup.py            # default scale n
    python scripts/verify_setup.py --scale s

Runs standalone (plain PyTorch) whether or not ultralytics is installed. If it is
installed and patched, the YAML models are built too, which is the real check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tomanet.modules import TomaLayer, fuse_reparam  # noqa: E402

# (channels, blocks, kernel) per stage - mirrors configs/models/tomanet-*.yaml
STAGES = [(128, 2, 3), (256, 4, 3), (512, 4, 5), (1024, 2, 5)]
SCALES = {"n": (0.50, 0.25), "s": (0.50, 0.50), "m": (1.00, 0.75)}
MAX_CHANNELS = 1024


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class Trunk(torch.nn.Module):
    """Standalone copy of the YAML backbone, for profiling without ultralytics."""

    def __init__(self, scale: str = "n"):
        super().__init__()
        depth, width = SCALES[scale]
        conv = lambda c1, c2: torch.nn.Sequential(  # noqa: E731
            torch.nn.Conv2d(c1, c2, 3, 2, 1, bias=False),
            torch.nn.BatchNorm2d(c2),
            torch.nn.SiLU(inplace=True),
        )

        c_stem = make_divisible(min(64, MAX_CHANNELS) * width)
        c_p2 = make_divisible(min(128, MAX_CHANNELS) * width)
        self.stem = torch.nn.Sequential(conv(3, c_stem), conv(c_stem, c_p2))

        layers, c_in = [], c_p2
        for i, (channels, blocks, kernel) in enumerate(STAGES):
            c_out = make_divisible(min(channels, MAX_CHANNELS) * width)
            n = max(1, round(blocks * depth))
            stage = [TomaLayer(c_in, c_out, n, kernel)]
            if i < len(STAGES) - 1:
                stage.append(conv(c_out, make_divisible(min(STAGES[i + 1][0], MAX_CHANNELS) * width)))
                c_in = make_divisible(min(STAGES[i + 1][0], MAX_CHANNELS) * width)
            layers.append(torch.nn.Sequential(*stage))
        self.stages = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_gflops(model: torch.nn.Module, size: int) -> float | None:
    """GFLOPs = 2 x MACs, matching the Ultralytics convention."""
    try:
        from thop import profile
    except ImportError:
        return None
    macs, _ = profile(model, inputs=(torch.zeros(1, 3, size, size),), verbose=False)
    return 2 * macs / 1e9


def check_standalone(scale: str) -> None:
    print(f"== standalone trunk (scale {scale}) ==")
    trunk = Trunk(scale).eval()

    for size in (224, 640):
        with torch.no_grad():
            out = trunk(torch.zeros(1, 3, size, size))
        gflops = count_gflops(Trunk(scale).eval(), size)
        flops_text = f"{gflops:.2f} GFLOPs" if gflops else "GFLOPs n/a (pip install thop)"
        print(f"  {size}px -> {tuple(out.shape)}   {flops_text}")

    print(f"  params: {count_params(trunk) / 1e6:.2f} M")


def check_reparam_fusion() -> None:
    print("== re-parameterisation ==")
    layer = TomaLayer(32, 32, n=2, k=3).eval()
    x = torch.randn(2, 32, 32, 32)

    with torch.no_grad():
        before = layer(x)
        after = fuse_reparam(layer)(x)

    delta = (before - after).abs().max().item()
    status = "OK" if delta < 1e-4 else "MISMATCH"
    print(f"  max |unfused - fused| = {delta:.2e}   [{status}]")
    if delta >= 1e-4:
        sys.exit("re-parameterisation is not equivalent - do not train until this is fixed")


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
            # Write scale-specific YAML with depth_multiple and width_multiple instead of scales dict
            yaml_text = src.read_text()
            depth, width = SCALES[scale]
            # Replace the scales dict with depth_multiple and width_multiple
            yaml_text = yaml_text.replace(
                "scales: # [depth, width, max_channels]\n  n: [0.50, 0.25, 1024]\n  s: [0.50, 0.50, 1024]\n  m: [1.00, 0.75, 768]",
                f"depth_multiple: {depth}\nwidth_multiple: {width}"
            )
            scaled.write_text(yaml_text, encoding="utf-8")
            model = YOLO(str(scaled))
            params = count_params(model.model)
            gflops = count_gflops(model.model, size)
            flops_text = f", {gflops:.2f} GFLOPs" if gflops else ""
            print(f"  tomanet-{task}-{scale} @ {size}px: {params / 1e6:.2f} M params{flops_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", default="n", choices=list(SCALES), help="model scale")
    args = parser.parse_args()

    check_standalone(args.scale)
    print()
    check_reparam_fusion()
    print()
    check_ultralytics()


if __name__ == "__main__":
    main()
