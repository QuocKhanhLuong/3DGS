"""Checkpoint-compatible, feature-only MedicalNet ResNet10.

This module intentionally implements only the ResNet10 feature extractor used
by the point-guided semantic prior.  Tencent MedicalNet's legacy ``conv_seg``
decoder is not recreated: a caller supplies its own explicitly scoped head.
The backbone layout and its state-dict names otherwise follow MedicalNet so a
locally supplied ResNet10 checkpoint can be validated and loaded exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Final
import torch
from torch import Tensor, nn

__all__ = [
    "BasicBlock",
    "MedicalNetCheckpointError",
    "MedicalNetCheckpointProvenance",
    "MedicalNetResNet10",
    "MedicalNetResNet10Backbone",
    "APPROVED_OFFICIAL_MEDICALNET_RESNET10_SHA256",
    "ResNet10Backbone",
    "adapt_medicalnet_input_conv_weight",
    "load_medicalnet_checkpoint",
    "resnet10",
    "sha256_file",
]


# The repository intentionally ships no checkpoint and therefore no unaudited
# digest is treated as official.  A future, separately reviewed provenance
# update may add the digest of an upstream ResNet10 release here.  Until then
# all successfully loaded local files are accurately recorded as custom or
# unverified checkpoints rather than being called pretrained MedicalNet.
APPROVED_OFFICIAL_MEDICALNET_RESNET10_SHA256: Final[frozenset[str]] = frozenset()


class MedicalNetCheckpointError(ValueError):
    """Raised when a local MedicalNet checkpoint fails validation."""


@dataclass(frozen=True)
class MedicalNetCheckpointProvenance:
    """Immutable evidence for a checkpoint actually loaded into a backbone."""

    checkpoint_path: str
    sha256: str
    source_input_channels: int
    adapted_input_channels: int
    input_conv_adapted: bool
    source_state_dict_key_count: int
    loaded_backbone_key_count: int
    integrity_verified: bool
    official_pretrained_verified: bool

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready provenance without exposing checkpoint contents."""

        return asdict(self)


def _conv3x3x3(
    in_planes: int,
    out_planes: int,
    *,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Conv3d:
    """MedicalNet's padded 3-D convolution helper."""

    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


class BasicBlock(nn.Module):
    """The unmodified 3-D residual block used by MedicalNet ResNet10."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3x3(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        return self.relu(out)


class MedicalNetResNet10(nn.Module):
    """The full MedicalNet ResNet10 backbone without its legacy decoder.

    ``in_channels=1`` preserves the upstream checkpoint architecture.  The
    point-guided semantic prior builds the same layout with ``in_channels=3``
    and uses :func:`adapt_medicalnet_input_conv_weight` when loading a native
    one-channel checkpoint.
    """

    feature_channels = 512
    layers = (1, 1, 1, 1)
    shortcut_type = "B"

    def __init__(self, *, in_channels: int = 1) -> None:
        super().__init__()
        if in_channels not in (1, 3):
            raise ValueError("MedicalNet ResNet10 supports only one or three input channels")

        self.in_channels = int(in_channels)
        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            self.in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, self.layers[0])
        self.layer2 = self._make_layer(128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(256, self.layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(512, self.layers[3], stride=1, dilation=4)
        self._initialize_weights()

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        *,
        stride: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            # ResNet10 is locked to MedicalNet shortcut type B.
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * BasicBlock.expansion),
            )

        blocks_list: list[nn.Module] = [
            BasicBlock(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            blocks_list.append(BasicBlock(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*blocks_list)

    def _initialize_weights(self) -> None:
        # This matches the upstream 3-D ResNet initialisation convention.
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """Return the final, stride-eight, 512-channel feature volume."""

        if not isinstance(x, Tensor):
            raise TypeError("MedicalNet ResNet10 expects a torch.Tensor input")
        if x.ndim != 5:
            raise ValueError("MedicalNet ResNet10 input must have shape [B, C, D, H, W]")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"MedicalNet ResNet10 expects {self.in_channels} input channels, "
                f"received {x.shape[1]}"
            )

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x: Tensor) -> Tensor:
        """Alias for :meth:`forward_features` for normal ``nn.Module`` use."""

        return self.forward_features(x)


# A descriptive alias for integration code that wants to emphasize the
# feature-only boundary.
ResNet10Backbone = MedicalNetResNet10
MedicalNetResNet10Backbone = MedicalNetResNet10


def resnet10(*, in_channels: int = 1) -> MedicalNetResNet10:
    """Construct the locked shortcut-B MedicalNet ResNet10 backbone."""

    return MedicalNetResNet10(in_channels=in_channels)


def sha256_file(checkpoint_path: str | Path) -> str:
    """Calculate a checkpoint digest without loading its contents."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"MedicalNet checkpoint does not exist: {path}. Automatic downloads are disabled."
        )

    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _sha256_stream(handle: BinaryIO) -> str:
    """Hash an already-open checkpoint stream without opening a second path."""

    digest = sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _normalise_expected_sha256(expected_sha256: str | None) -> str | None:
    if expected_sha256 is None:
        return None
    digest = expected_sha256.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise MedicalNetCheckpointError("expected_sha256 must be a SHA-256 hex digest")
    return digest


def adapt_medicalnet_input_conv_weight(
    weight: Tensor,
    *,
    target_input_channels: int = 3,
) -> Tensor:
    """Adapt a native one-channel MedicalNet stem deterministically to RGB-like MRI.

    Repeating the one-channel kernel and dividing by three preserves the stem
    response for identical T1, T2, and FLAIR values.  This is the only input
    convolution adaptation permitted by the locked frontend.
    """

    if not isinstance(weight, Tensor) or weight.ndim != 5:
        raise MedicalNetCheckpointError("MedicalNet conv1.weight must be a rank-five tensor")
    if weight.shape[1] != 1:
        raise MedicalNetCheckpointError(
            "MedicalNet input-conv adaptation requires exactly one source channel"
        )
    if target_input_channels != 3:
        raise MedicalNetCheckpointError(
            "MedicalNet input-conv adaptation is locked to three T1/T2/FLAIR channels"
        )
    return weight.repeat(1, target_input_channels, 1, 1, 1) / float(target_input_channels)


def _load_checkpoint_object(
    source: BinaryIO | Path,
    *,
    source_label: Path | None = None,
) -> object:
    """Load tensor-only data from an already-open stream when available."""

    if isinstance(source, Path):
        with source.open("rb") as handle:
            return _load_checkpoint_object(handle, source_label=source)
    label = str(source_label) if source_label is not None else "open checkpoint stream"

    try:
        # ``weights_only`` is available on supported modern PyTorch releases.
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise MedicalNetCheckpointError(
            "This PyTorch build lacks torch.load(weights_only=True); refusing unsafe checkpoint deserialization"
        ) from error
    except Exception as error:  # pragma: no cover - message depends on PyTorch version.
        raise MedicalNetCheckpointError(
            f"Unable to load tensor-only MedicalNet checkpoint: {label}"
        ) from error


def _extract_state_dict(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise MedicalNetCheckpointError("MedicalNet checkpoint must be a mapping")

    candidate: object = checkpoint.get("state_dict", checkpoint)
    if not isinstance(candidate, Mapping):
        raise MedicalNetCheckpointError("MedicalNet checkpoint state_dict must be a mapping")
    if not candidate:
        raise MedicalNetCheckpointError("MedicalNet checkpoint state_dict must not be empty")

    state_dict: dict[str, Tensor] = {}
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise MedicalNetCheckpointError(
                "MedicalNet checkpoint state_dict must map string keys to tensors"
            )
        state_dict[key] = value

    prefixed = [key.startswith("module.") for key in state_dict]
    if any(prefixed):
        if not all(prefixed):
            raise MedicalNetCheckpointError(
                "MedicalNet checkpoint has a mixed DataParallel module. prefix"
            )
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _is_legacy_decoder_key(key: str) -> bool:
    """Recognise only actual keys emitted by MedicalNet's excluded conv_seg."""

    return key.startswith((
        "conv_seg.0.",
        "conv_seg.1.",
        "conv_seg.3.",
        "conv_seg.4.",
        "conv_seg.6.",
    ))


def _validated_backbone_state_dict(
    backbone: MedicalNetResNet10,
    source_state_dict: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], int, bool]:
    """Drop only known legacy decoder keys and strictly validate all backbone keys."""

    unexpected_legacy_keys = [
        key for key in source_state_dict if key.startswith("conv_seg.") and not _is_legacy_decoder_key(key)
    ]
    if unexpected_legacy_keys:
        raise MedicalNetCheckpointError(
            "MedicalNet checkpoint contains unknown legacy decoder keys: "
            f"{sorted(unexpected_legacy_keys)}"
        )

    candidate = {
        key: value
        for key, value in source_state_dict.items()
        if not key.startswith("conv_seg.")
    }
    expected = backbone.state_dict()
    missing = sorted(set(expected) - set(candidate))
    unexpected = sorted(set(candidate) - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing keys={missing}")
        if unexpected:
            details.append(f"unexpected keys={unexpected}")
        raise MedicalNetCheckpointError(
            "MedicalNet backbone state_dict does not match ResNet10: " + "; ".join(details)
        )

    source_stem = candidate["conv1.weight"]
    target_stem = expected["conv1.weight"]
    source_input_channels = int(source_stem.shape[1]) if source_stem.ndim == 5 else -1
    adapted = False
    if tuple(source_stem.shape) != tuple(target_stem.shape):
        can_adapt_stem = (
            source_stem.ndim == 5
            and target_stem.ndim == 5
            and source_stem.shape[0] == target_stem.shape[0]
            and tuple(source_stem.shape[2:]) == tuple(target_stem.shape[2:])
            and source_stem.shape[1] == 1
            and target_stem.shape[1] == 3
        )
        if not can_adapt_stem:
            raise MedicalNetCheckpointError(
                "MedicalNet conv1.weight shape is incompatible with the locked ResNet10 stem: "
                f"checkpoint={tuple(source_stem.shape)}, expected={tuple(target_stem.shape)}"
            )
        candidate["conv1.weight"] = adapt_medicalnet_input_conv_weight(source_stem)
        adapted = True

    for key, target_tensor in expected.items():
        source_tensor = candidate[key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise MedicalNetCheckpointError(
                f"MedicalNet tensor shape mismatch for {key}: "
                f"checkpoint={tuple(source_tensor.shape)}, expected={tuple(target_tensor.shape)}"
            )
        if source_tensor.dtype != target_tensor.dtype:
            raise MedicalNetCheckpointError(
                f"MedicalNet tensor dtype mismatch for {key}: "
                f"checkpoint={source_tensor.dtype}, expected={target_tensor.dtype}"
            )

    return candidate, source_input_channels, adapted


def load_medicalnet_checkpoint(
    backbone: MedicalNetResNet10,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_official_pretrained: bool = False,
) -> MedicalNetCheckpointProvenance:
    """Verify and load a local MedicalNet checkpoint with no fallback path.

    The load is strict for every feature-backbone key.  A standard MedicalNet
    checkpoint may include its historical ``conv_seg`` decoder; those known
    decoder tensors are ignored because this module deliberately exposes
    features only.  No other missing or unexpected tensor is accepted.
    """

    if not isinstance(backbone, MedicalNetResNet10):
        raise TypeError("backbone must be a MedicalNetResNet10 instance")

    if not isinstance(require_official_pretrained, bool):
        raise TypeError("require_official_pretrained must be a bool")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"MedicalNet checkpoint does not exist: {path}. Automatic downloads are disabled."
        )
    required_sha256 = _normalise_expected_sha256(expected_sha256)
    # Snapshot bytes once before hashing.  The digest and deserialization then
    # consume independent cursors over precisely those immutable bytes, so
    # even an in-place write to the original path cannot misbind provenance.
    with path.open("rb") as checkpoint_stream:
        checkpoint_snapshot = checkpoint_stream.read()
    actual_sha256 = _sha256_stream(BytesIO(checkpoint_snapshot))
    if required_sha256 is not None and actual_sha256 != required_sha256:
        raise MedicalNetCheckpointError(
            "MedicalNet checkpoint SHA-256 mismatch: "
            f"expected={required_sha256}, actual={actual_sha256}"
        )
    source_state_dict = _extract_state_dict(
        _load_checkpoint_object(BytesIO(checkpoint_snapshot), source_label=path)
    )
    loadable_state_dict, source_input_channels, adapted = _validated_backbone_state_dict(
        backbone,
        source_state_dict,
    )
    official_pretrained_verified = (
        required_sha256 is not None
        and actual_sha256 in APPROVED_OFFICIAL_MEDICALNET_RESNET10_SHA256
        and source_input_channels == 1
        and adapted
    )
    if require_official_pretrained and not official_pretrained_verified:
        raise MedicalNetCheckpointError(
            "official pretrained MedicalNet requires an approved one-channel ResNet10 SHA-256 digest; "
            "this repository currently has no approved digest"
        )
    incompatible = backbone.load_state_dict(loadable_state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:  # Defensive; strict=True should raise first.
        raise MedicalNetCheckpointError(
            "MedicalNet strict load unexpectedly reported incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    return MedicalNetCheckpointProvenance(
        checkpoint_path=str(path.resolve()),
        sha256=actual_sha256,
        source_input_channels=source_input_channels,
        adapted_input_channels=backbone.in_channels,
        input_conv_adapted=adapted,
        source_state_dict_key_count=len(source_state_dict),
        loaded_backbone_key_count=len(loadable_state_dict),
        integrity_verified=required_sha256 is not None,
        official_pretrained_verified=official_pretrained_verified,
    )
