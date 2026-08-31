"""Building blocks for TomaNet.

One block (`TomaBlock`) is reused in the classifier trunk, the detector trunk and the
detector neck. Design rationale is in main.pdf section 7; in short:

  PConv      - spatial mixing on 1/4 of the channels keeps memory traffic low, which is
               the real bottleneck on edge devices (FasterNet, arXiv:2303.03667).
  DWConv 5x5 - lesion-scale receptive field in deep stages for no extra parameters.
  CoordAtt   - encodes position along H and W. Lesions are small and position-specific;
               channel-only attention (SE/ECA) discards exactly what matters
               (arXiv:2103.02907).
  Rep branch - multi-branch at train time, single 3x3 at inference (arXiv:2101.03697).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["CoordAtt", "PConv", "RepPConv", "TomaBlock", "TomaLayer"]


def autopad(k: int, p: int | None = None) -> int:
    """Padding that keeps spatial size unchanged for odd kernels."""
    return k // 2 if p is None else p


class ConvBNAct(nn.Module):
    """Conv -> BN -> activation. Mirrors ultralytics.nn.modules.Conv defaults."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CoordAtt(nn.Module):
    """Coordinate attention (Hou et al., CVPR 2021).

    Pools separately along H and W so the attention weights retain positional
    information, then re-weights the input by both maps.
    """

    def __init__(self, c: int, reduction: int = 32):
        super().__init__()
        mid = max(8, c // reduction)
        self.conv1 = nn.Conv2d(c, mid, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mid, c, 1)
        self.conv_w = nn.Conv2d(mid, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)                      # (n, c, h, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (n, c, w, 1)

        y = self.act(self.bn(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        return x * self.conv_h(y_h).sigmoid() * self.conv_w(y_w).sigmoid()


class PConv(nn.Module):
    """Partial convolution: convolve the first `ratio` of channels, pass the rest through."""

    def __init__(self, c: int, k: int = 3, ratio: float = 0.25):
        super().__init__()
        self.c_conv = max(1, int(c * ratio))
        self.c_pass = c - self.c_conv
        self.conv = nn.Conv2d(self.c_conv, self.c_conv, k, 1, autopad(k), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.split(x, [self.c_conv, self.c_pass], dim=1)
        return torch.cat([self.conv(a), b], dim=1)


class RepPConv(nn.Module):
    """Re-parameterisable partial convolution.

    Train time: 3x3 + 1x1 + identity branches (each with BN) on the convolved channels.
    Inference: `fuse()` collapses them into a single 3x3 convolution, so the deployed
    model pays for one branch only.
    """

    def __init__(self, c: int, k: int = 3, ratio: float = 0.25):
        super().__init__()
        if k != 3:
            raise ValueError(f"RepPConv supports k=3 only, got {k}")
        self.c_conv = max(1, int(c * ratio))
        self.c_pass = c - self.c_conv

        self.conv_k = nn.Conv2d(self.c_conv, self.c_conv, 3, 1, 1, bias=False)
        self.bn_k = nn.BatchNorm2d(self.c_conv)
        self.conv_1 = nn.Conv2d(self.c_conv, self.c_conv, 1, 1, 0, bias=False)
        self.bn_1 = nn.BatchNorm2d(self.c_conv)
        self.bn_id = nn.BatchNorm2d(self.c_conv)

        self.fused: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = torch.split(x, [self.c_conv, self.c_pass], dim=1)
        if self.fused is not None:
            return torch.cat([self.fused(a), b], dim=1)
        y = self.bn_k(self.conv_k(a)) + self.bn_1(self.conv_1(a)) + self.bn_id(a)
        return torch.cat([y, b], dim=1)

    @staticmethod
    def _fuse_conv_bn(weight: torch.Tensor, bn: nn.BatchNorm2d):
        std = (bn.running_var + bn.eps).sqrt()
        scale = (bn.weight / std).reshape(-1, 1, 1, 1)
        return weight * scale, bn.bias - bn.running_mean * bn.weight / std

    def _identity_kernel(self) -> torch.Tensor:
        kernel = torch.zeros(self.c_conv, self.c_conv, 3, 3, device=self.bn_id.weight.device)
        for i in range(self.c_conv):
            kernel[i, i, 1, 1] = 1.0
        return kernel

    @torch.no_grad()
    def fuse(self) -> None:
        """Collapse the three branches into `self.fused`. Idempotent."""
        if self.fused is not None:
            return

        w_k, b_k = self._fuse_conv_bn(self.conv_k.weight, self.bn_k)
        w_1, b_1 = self._fuse_conv_bn(self.conv_1.weight, self.bn_1)
        w_id, b_id = self._fuse_conv_bn(self._identity_kernel(), self.bn_id)

        weight = w_k + nn.functional.pad(w_1, [1, 1, 1, 1]) + w_id
        bias = b_k + b_1 + b_id

        fused = nn.Conv2d(self.c_conv, self.c_conv, 3, 1, 1, bias=True)
        fused.weight.copy_(weight)
        fused.bias.copy_(bias)
        self.fused = fused

        for name in ("conv_k", "bn_k", "conv_1", "bn_1", "bn_id"):
            delattr(self, name)


class TomaBlock(nn.Module):
    """PConv -> expand 1x1 -> DWConv kxk -> CoordAtt -> project 1x1, with residual.

    The residual is used only when `c1 == c2`, so the same block serves both the trunk
    (channel-preserving) and the neck (channel-changing fusion nodes).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        expansion: float = 2.0,
        ratio: float = 0.25,
        attn: bool = True,
        rep: bool = True,
    ):
        super().__init__()
        c_hidden = int(c1 * expansion)

        self.spatial = RepPConv(c1, 3, ratio) if rep else PConv(c1, 3, ratio)
        self.expand = ConvBNAct(c1, c_hidden, 1)
        self.depthwise = ConvBNAct(c_hidden, c_hidden, k, g=c_hidden)
        self.attn = CoordAtt(c_hidden) if attn else nn.Identity()
        self.project = nn.Sequential(
            nn.Conv2d(c_hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.residual = c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.attn(self.depthwise(self.expand(self.spatial(x)))))
        return x + y if self.residual else y


class TomaLayer(nn.Module):
    """`n` stacked TomaBlocks. First block maps c1 -> c2, the rest are c2 -> c2.

    This is the unit referenced from the model YAMLs, so it follows the ultralytics
    signature convention `(c1, c2, n, *args)`.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        k: int = 3,
        expansion: float = 2.0,
        attn: bool = True,
        rep: bool = True,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                TomaBlock(c1 if i == 0 else c2, c2, k, expansion, attn=attn, rep=rep)
                for i in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


def fuse_reparam(model: nn.Module) -> nn.Module:
    """Fuse every RepPConv in `model` for inference. Call after loading weights."""
    for module in model.modules():
        if isinstance(module, RepPConv):
            module.fuse()
    return model
