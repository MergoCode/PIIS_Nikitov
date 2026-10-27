import argparse
import json
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.data import (
    DEFAULT_LABELS,
    build_collate_fn,
    get_speechcommands_subset,
)
from src.model import AudioCNN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = (correct / total * 100) if total > 0 else 0.0
    avg_loss = (total_loss / total) if total > 0 else 0.0
    return accuracy, avg_loss


def train(args):
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    print(f"Loading SpeechCommands dataset for classes: {args.classes}...", flush=True)
    sub_dataset = get_speechcommands_subset(args.data_dir, args.classes, download=True)
    print(f"Found {len(sub_dataset)} matching audio samples.", flush=True)

    train_size = int(0.8 * len(sub_dataset))
    test_size = len(sub_dataset) - train_size
    train_set, test_set = random_split(
        sub_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    collate_fn = build_collate_fn(args.classes)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)

    model = AudioCNN(num_classes=len(args.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training for {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 40 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}",
                    flush=True,
                )

        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}", flush=True)

    accuracy, test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {accuracy:.2f}% | Test Loss: {test_loss:.4f}", flush=True)

    # Save artifacts
    model_path = os.path.join(args.out, "model.pt")
    torch.save(model.state_dict(), model_path)
    model_size_kb = os.path.getsize(model_path) / 1024

    labels_path = os.path.join(args.out, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(args.classes, f, indent=2)

    metrics = {
        "accuracy": round(accuracy, 2),
        "test_loss": round(test_loss, 4),
        "model_size_kb": round(model_size_kb, 2),
        "epochs": args.epochs,
        "seed": args.seed,
        "classes": args.classes,
        "architecture": "AudioCNN",
        "model_version": "1.0.0",
    }
    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Artifacts successfully saved to {args.out}:", flush=True)
    print(f" - {model_path} ({model_size_kb:.2f} KB)")
    print(f" - {labels_path}")
    print(f" - {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Audio Keyword Spotting CNN")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_LABELS,
        help="List of classes to train on",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", type=str, default="./data", help="Root directory for dataset")
    parser.add_argument(
        "--out",
        type=str,
        default="./artifacts",
        help="Output directory for artifacts",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
