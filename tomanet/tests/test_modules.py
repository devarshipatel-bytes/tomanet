"""Unit tests for the TomaNet blocks.

    pytest tests/ -q          (or: python tests/test_modules.py)

The fusion-equivalence test is the important one: if re-parameterisation is not exactly
equivalent, a trained model silently degrades at inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tomanet.modules import CoordAtt, PConv, RepPConv, TomaBlock, TomaLayer, fuse_reparam


def test_coordatt_preserves_shape():
    x = torch.randn(2, 32, 16, 16)
    assert CoordAtt(32)(x).shape == x.shape


def test_coordatt_is_not_identity():
    """A CoordAtt with random weights must actually change the input."""
    torch.manual_seed(0)
    x = torch.randn(2, 32, 16, 16)
    assert not torch.allclose(CoordAtt(32).eval()(x), x)


def test_pconv_preserves_shape_and_passes_channels_through():
    layer = PConv(32, ratio=0.25).eval()
    x = torch.randn(2, 32, 16, 16)
    y = layer(x)
    assert y.shape == x.shape
    # channels beyond the convolved slice must be untouched
    assert torch.allclose(y[:, layer.c_conv :], x[:, layer.c_conv :])


def test_tomablock_residual_only_when_channels_match():
    assert TomaBlock(32, 32).residual is True
    assert TomaBlock(32, 64).residual is False


def test_tomablock_shapes():
    x = torch.randn(2, 32, 16, 16)
    assert TomaBlock(32, 32).eval()(x).shape == (2, 32, 16, 16)
    assert TomaBlock(32, 64).eval()(x).shape == (2, 64, 16, 16)


def test_tomalayer_maps_channels():
    x = torch.randn(2, 16, 32, 32)
    assert TomaLayer(16, 48, n=3, k=5).eval()(x).shape == (2, 48, 32, 32)


def test_reppconv_fusion_is_equivalent():
    """The whole point of re-parameterisation: identical output, one branch."""
    torch.manual_seed(0)
    layer = RepPConv(64, ratio=0.25).eval()
    # push BN away from its identity init so the test is meaningful
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_(0, 0.5)
            module.running_var.uniform_(0.5, 1.5)
            module.weight.data.normal_(1.0, 0.2)
            module.bias.data.normal_(0, 0.2)

    x = torch.randn(4, 64, 16, 16)
    with torch.no_grad():
        before = layer(x)
        layer.fuse()
        after = layer(x)

    assert torch.allclose(before, after, atol=1e-4), (before - after).abs().max()


def test_fuse_is_idempotent():
    layer = RepPConv(32).eval()
    x = torch.randn(1, 32, 8, 8)
    with torch.no_grad():
        layer.fuse()
        once = layer(x)
        layer.fuse()
        twice = layer(x)
    assert torch.allclose(once, twice)


def test_fuse_reparam_walks_nested_modules():
    layer = TomaLayer(32, 32, n=2).eval()
    torch.manual_seed(0)
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        before = layer(x)
        after = fuse_reparam(layer)(x)
    assert torch.allclose(before, after, atol=1e-4)
    assert all(m.fused is not None for m in layer.modules() if isinstance(m, RepPConv))


def test_backbone_yamls_share_an_identical_backbone():
    """TomaNet-C and TomaNet-D must have byte-identical backbone lists."""
    configs = Path(__file__).resolve().parent.parent / "configs" / "models"

    def backbone(name: str) -> str:
        text = (configs / name).read_text()
        return text.split("backbone:")[1].split("head:")[0].strip()

    assert backbone("tomanet-cls.yaml") == backbone("tomanet-det.yaml")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
