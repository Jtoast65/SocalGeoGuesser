from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_utils import SoCalGuessrDataset, build_eval_transform, load_checkpoint


MODEL_FILENAME = "model_weights.pt"


def predict(image_path: str) -> dict[str, str]:
    image_dir = Path(image_path)
    model_path = Path(__file__).with_name(MODEL_FILENAME)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected model weights at {model_path}. "
            "Zip predict.py together with model_weights.pt at the top level."
        )

    model, metadata = load_checkpoint(model_path, map_location="cpu")
    model.eval()

    dataset = SoCalGuessrDataset(
        sorted(image_dir.glob("*.jpg")),
        transform=build_eval_transform(metadata.image_size),
        class_names=metadata.class_names,
        include_labels=False,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    predictions: dict[str, str] = {}
    with torch.no_grad():
        for images, filenames in loader:
            logits = model(images)
            logits = (logits + model(torch.flip(images, dims=[3]))) / 2
            predicted_indices = logits.argmax(dim=1).tolist()
            for filename, index in zip(filenames, predicted_indices):
                predictions[filename] = metadata.class_names[index]

    return predictions
