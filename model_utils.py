from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import warnings

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms


CLASS_NAMES = [
    "Anaheim",
    "Bakersfield",
    "Los_Angeles",
    "Riverside",
    "SLO",
    "San_Diego",
]

DEFAULT_IMAGE_SIZE = (224, 384)
SUPPORTED_MODELS = (
    "mobilenet_v3_small",
    "efficientnet_b0",
)


def infer_label(path: Path) -> str:
    return path.name.split("-", 1)[0]


def list_image_paths(data_dir: str | Path) -> list[Path]:
    return sorted(Path(data_dir).glob("*.jpg"))


def split_paths(
    data_dir: str | Path,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[Path], list[Path]]:
    image_paths = list_image_paths(data_dir)
    labels = [infer_label(path) for path in image_paths]
    train_paths, val_paths = train_test_split(
        image_paths,
        test_size=val_size,
        random_state=random_state,
        stratify=labels,
    )
    return sorted(train_paths), sorted(val_paths)


def build_train_transform(image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.65, 1.0),
                ratio=(1.3, 2.0),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.12,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_eval_transform(image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


class SoCalGuessrDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        transform,
        class_names: Sequence[str] = CLASS_NAMES,
        include_labels: bool = True,
    ) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform
        self.class_names = list(class_names)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        if not self.include_labels:
            return image_tensor, image_path.name

        label = infer_label(image_path)
        return image_tensor, self.class_to_idx[label]


def build_model(
    num_classes: int,
    model_name: str = "mobilenet_v3_small",
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model_builder = models.mobilenet_v3_small
        warning_name = "MobileNetV3-Small"
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model_builder = models.efficientnet_b0
        warning_name = "EfficientNet-B0"
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    try:
        model = model_builder(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        warnings.warn(
            f"Falling back to randomly initialized {warning_name}: {exc}",
            stacklevel=2,
        )
        model = model_builder(weights=None)

    if model_name == "mobilenet_v3_small":
        in_features = model.classifier[0].in_features
        hidden_features = model.classifier[0].out_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_features, num_classes),
        )
    elif model_name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
    return model


def freeze_backbone(model: nn.Module, frozen: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = not frozen


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


@dataclass
class CheckpointMetadata:
    class_names: list[str]
    image_size: tuple[int, int]
    model_name: str = "mobilenet_v3_small"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: CheckpointMetadata,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": metadata.class_names,
            "image_size": metadata.image_size,
            "model_name": metadata.model_name,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    image_size = tuple(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    model_name = checkpoint.get("model_name", "mobilenet_v3_small")
    model = build_model(
        num_classes=len(class_names),
        model_name=model_name,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    metadata = CheckpointMetadata(
        class_names=list(class_names),
        image_size=(int(image_size[0]), int(image_size[1])),
        model_name=model_name,
    )
    return model, metadata
