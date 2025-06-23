import argparse

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def train_model(learning_rate, batch_size):
    print(f"Training with learning_rate={learning_rate}, batch_size={batch_size}")

    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)

        # Prepare dataset
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Model, loss, optimizer
        model = SimpleNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        model.train()
        for epoch in range(5):
            total_loss = 0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("loss", avg_loss, step=epoch)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # Save model
        model_path = "mnist_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST with MLflow tracking")

    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    args = parser.parse_args()

    train_model(args.learning_rate, args.batch_size)
