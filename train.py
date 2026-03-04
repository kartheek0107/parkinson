import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.loader import DQLCTDataset
from src.models.simple_cnn import SimpleDQLCTNet


def train():
    # 1. Setup
    print("🚀 Starting Training Pipeline...")
    device = torch.device("cuda")
    print(f"   Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 4
    LR = 0.001
    EPOCHS = 10

    # 2. Load Data
    # Point this to where make_dataset.py saved your .pt files
    dataset = DQLCTDataset(processed_dir='./data/processed')

    # Safety Check
    if len(dataset) == 0:
        print("❌ Error: No .pt files found in data/processed. Did you run make_dataset.py?")
        return

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"   Loaded {len(dataset)} samples.")

    # 3. Initialize Model
    model = SimpleDQLCTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)

            # Backward
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        acc = 100. * correct / total
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  Loss: {total_loss / len(train_loader):.4f}  Acc: {acc:.2f}%")

    print("✅ Training Complete.")

    # 5. Quick Test (Inference)
    print("\n🔮 Inference Test on one sample:")
    test_data = dataset[0][0].unsqueeze(0).to(device)  # Get first sample
    output = model(test_data)
    pred = output.argmax(dim=1).item()
    print(f"   Predicted Class: {pred} ({'PD' if pred == 1 else 'Healthy'})")
    print(f"   Actual Label:    {dataset[0][1]}")


if __name__ == "__main__":
    train()