import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.data import (
    DEFAULT_LABELS,
    build_collate_fn,
    get_speechcommands_subset,
)
from src.model import AudioCNN


def evaluate_model(args):
    labels = DEFAULT_LABELS
    if args.labels_path and os.path.exists(args.labels_path):
        with open(args.labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
    elif args.classes:
        labels = args.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")
    print(f"Target classes: {labels}")

    model = AudioCNN(num_classes=len(labels)).to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at: {args.model_path}")

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    sub_dataset = get_speechcommands_subset(args.data_dir, labels, download=False)
    train_size = int(0.8 * len(sub_dataset))
    test_size = len(sub_dataset) - train_size
    _, test_set = random_split(
        sub_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    collate_fn = build_collate_fn(labels)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()
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

    print("--- Evaluation Results ---")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Loss: {avg_loss:.4f}")

    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate Audio Keyword Spotting Model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./artifacts/model.pt",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default="./artifacts/labels.json",
        help="Path to labels.json",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class list override",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root directory for dataset",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Dataset split seed")

    args = parser.parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
