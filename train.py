from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model_utils import (
    CLASS_NAMES,
    CheckpointMetadata,
    SoCalGuessrDataset,
    SUPPORTED_MODELS,
    build_eval_transform,
    build_model,
    build_train_transform,
    freeze_backbone,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    split_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SoCalGuessr classifier.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--weights-name", default="model_weights.pt")
    parser.add_argument("--model-name", choices=SUPPORTED_MODELS, default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--tta", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(args: argparse.Namespace):
    train_paths, val_paths = split_paths(
        args.data_dir,
        val_size=args.val_size,
        random_state=args.split_seed if args.split_seed is not None else args.seed,
    )
    train_dataset = SoCalGuessrDataset(train_paths, build_train_transform())
    val_dataset = SoCalGuessrDataset(val_paths, build_eval_transform())
    pin_memory = resolve_device(args.device).type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    use_tta: bool = False,
):
    model.eval()
    losses = []
    all_true = []
    all_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            if use_tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2
            loss = criterion(logits, labels)
            losses.append(loss.item())
            predictions = logits.argmax(dim=1)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(predictions.cpu().tolist())

    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(all_true, all_pred),
        "confusion_matrix": confusion_matrix(all_true, all_pred),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
):
    model.train()
    losses = []
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": correct / total,
    }


def save_history(history: list[dict], path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "elapsed_seconds",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_curve_plot(history: list[dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Empirical risk")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / args.weights_name
    history_path = output_dir / "training_history.csv"
    curve_path = output_dir / "training_curve.png"

    train_loader, val_loader = make_loaders(args)

    if args.resume:
        model, metadata = load_checkpoint(args.resume, map_location=device)
        class_names = metadata.class_names
        model_name = metadata.model_name
    else:
        class_names = CLASS_NAMES
        model_name = args.model_name
        model = build_model(
            num_classes=len(class_names),
            model_name=model_name,
            pretrained=not args.no_pretrained,
            dropout=args.dropout,
        )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": args.backbone_learning_rate,
            },
            {
                "params": model.classifier.parameters(),
                "lr": args.learning_rate,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history: list[dict] = []
    best_val_accuracy = -1.0
    patience_left = args.patience
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        freeze_backbone(model, frozen=epoch <= args.freeze_epochs)
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_tta=args.tta,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "elapsed_seconds": time.time() - epoch_start,
        }
        history.append(row)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f}",
            flush=True,
        )

        if row["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = row["val_accuracy"]
            patience_left = args.patience
            save_checkpoint(
                checkpoint_path,
                model,
                CheckpointMetadata(
                    class_names=class_names,
                    image_size=(224, 384),
                    model_name=model_name,
                ),
            )
            np.savetxt(
                output_dir / "validation_confusion_matrix.csv",
                val_metrics["confusion_matrix"],
                delimiter=",",
                fmt="%d",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    total_training_time = time.time() - training_start
    save_history(history, history_path)
    save_curve_plot(history, curve_path)

    print(f"best_val_accuracy={best_val_accuracy:.4f}", flush=True)
    print(f"training_time_seconds={total_training_time:.2f}", flush=True)
    print(f"saved_weights={checkpoint_path}", flush=True)
    print(f"saved_history={history_path}", flush=True)
    print(f"saved_curve={curve_path}", flush=True)


if __name__ == "__main__":
    main()
