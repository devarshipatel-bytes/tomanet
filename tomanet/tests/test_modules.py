"""Unit tests for the TomaNet blocks.

    pytest tests/ -q          (or: python tests/test_modules.py)

The fusion-equivalence test is the important one: if re-parameterisation is not exactly
equivalent, a trained model silently degrades at inference.
"""

from __future__ import annotations

import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tomanet.modules import CoordAtt, PConv, RepPConv, TomaBlock, TomaLayer, fuse_reparam

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prepare_data  # noqa: E402


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


def test_cls_backbone_transfers_into_the_detector():
    """train_det.py --init-from relies on every backbone tensor matching by name AND shape."""
    from tomanet.register import is_patched, register

    register()
    if not is_patched():
        return  # ultralytics not patched; build would be wrong anyway
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import intersect_dicts

    configs = Path(__file__).resolve().parent.parent / "configs" / "models"
    with tempfile.TemporaryDirectory() as tmp:
        built = {}
        for kind in ("cls", "det"):
            path = Path(tmp) / f"tomanet-{kind}n.yaml"
            path.write_text((configs / f"tomanet-{kind}.yaml").read_text())
            built[kind] = YOLO(str(path)).model.state_dict()

    backbone = tuple(f"model.{i}." for i in range(9))
    shared = intersect_dicts(built["cls"], built["det"])
    det_backbone = {k for k in built["det"] if k.startswith(backbone)}
    assert det_backbone <= set(shared), "backbone tensors would not transfer"
    assert not (set(shared) - det_backbone), "non-backbone tensors transferred unexpectedly"


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
