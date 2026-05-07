from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from model_utils import load_checkpoint


MODEL_GLOB = "model_weights*.pt"
NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def _to_tensor(image: Image.Image) -> torch.Tensor:
    return NORMALIZE(TF.to_tensor(image))


class MultiViewDataset:
    def __init__(self, image_paths: list[Path], image_size: tuple[int, int]) -> None:
        self.image_paths = image_paths
        self.image_size = image_size
        self.wide_size = (image_size[0], image_size[1] + 64)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            base = TF.resize(image, self.image_size, antialias=True)
            wide = TF.resize(image, self.wide_size, antialias=True)
            crop_left = TF.crop(
                wide,
                top=0,
                left=0,
                height=self.image_size[0],
                width=self.image_size[1],
            )
            crop_right = TF.crop(
                wide,
                top=0,
                left=self.wide_size[1] - self.image_size[1],
                height=self.image_size[0],
                width=self.image_size[1],
            )
            views = torch.stack(
                [
                    _to_tensor(base),
                    _to_tensor(TF.hflip(base)),
                    _to_tensor(crop_left),
                    _to_tensor(TF.hflip(crop_left)),
                    _to_tensor(crop_right),
                    _to_tensor(TF.hflip(crop_right)),
                ]
            )
        return views, path.name


def predict(image_path: str) -> dict[str, str]:
    image_dir = Path(image_path)
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        return {}

    model_paths = sorted(Path(__file__).parent.glob(MODEL_GLOB))
    if not model_paths:
        raise FileNotFoundError(
            "Expected at least one model_weights*.pt file next to predict.py. "
            "Zip predict.py together with the saved model weights at the top level."
        )

    combined_logits: dict[str, torch.Tensor] = {}
    class_names: list[str] | None = None

    with torch.no_grad():
        for model_path in model_paths:
            model, metadata = load_checkpoint(model_path, map_location="cpu")
            model.eval()

            if class_names is None:
                class_names = metadata.class_names
            elif class_names != metadata.class_names:
                raise ValueError("All checkpoints in the ensemble must use the same class order.")

            dataset = MultiViewDataset(
                image_paths,
                image_size=metadata.image_size,
            )
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

            for views, filenames in loader:
                batch_size, num_views, channels, height, width = views.shape
                logits = model(views.view(batch_size * num_views, channels, height, width))
                logits = logits.view(batch_size, num_views, -1).mean(dim=1)
                for filename, logit in zip(filenames, logits.cpu()):
                    if filename not in combined_logits:
                        combined_logits[filename] = logit
                    else:
                        combined_logits[filename] += logit

    assert class_names is not None
    predictions = {
        filename: class_names[int(logits.argmax().item())]
        for filename, logits in combined_logits.items()
    }
    return predictions
