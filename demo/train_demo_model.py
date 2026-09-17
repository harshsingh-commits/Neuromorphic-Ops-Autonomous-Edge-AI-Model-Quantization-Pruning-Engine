from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from demo.demo_model import DemoCNN


DEVICE = "cpu"

DATA_ROOT = "data"

BATCH_SIZE = 64

TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000

EPOCHS = 3

LEARNING_RATE = 0.001


def build_datasets():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    if TRAIN_SAMPLES < len(train_dataset):
        train_dataset = torch.utils.data.Subset(
            train_dataset,
            range(TRAIN_SAMPLES),
        )

    if TEST_SAMPLES < len(test_dataset):
        test_dataset = torch.utils.data.Subset(
            test_dataset,
            range(TEST_SAMPLES),
        )

    return train_dataset, test_dataset


def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> float:
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.size(0)

    if total == 0:
        return 0.0

    return (
        correct / total
    ) * 100.0


def main():
    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    model_path = (
        project_root
        / "demo"
        / "trained_demo_model.pt"
    )

    data_root = (
        project_root
        / DATA_ROOT
    )

    print("=" * 60)
    print("Neuromorphic-Ops Demo Model Training")
    print("=" * 60)

    print("\nLoading CIFAR-10...")

    train_dataset, test_dataset = build_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Evaluation samples: {len(test_dataset)}"
    )

    model = DemoCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nTraining started...")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        for batch_index, (
            inputs,
            targets,
        ) in enumerate(train_loader):

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = (
            running_loss
            / len(train_loader)
        )

        accuracy = evaluate(
            model,
            test_loader,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Loss: {average_loss:.4f} "
            f"| Test Accuracy: {accuracy:.2f}%"
        )

    final_accuracy = evaluate(
        model,
        test_loader,
    )

    model.eval()

    torch.save(
        model,
        model_path,
    )

    print("\nTraining completed.")
    print(
        f"Final accuracy: {final_accuracy:.2f}%"
    )

    print(
        f"Model parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        f"Saved model: {model_path}"
    )

    print(
        f"Dataset directory: {data_root}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()