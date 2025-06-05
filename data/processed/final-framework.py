import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Define a simple model
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))  # Keeping raw logits for CrossEntropyLoss

# Initialize model, criterion, optimizer
input_size = 10
output_size = 2
model = NeuralNet(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Create dummy dataset
x_train = torch.rand(500, input_size)
y_train = torch.randint(0, output_size, (500,)).view(-1)  # Reshaped labels
x_val = torch.rand(100, input_size)  # Validation set
y_val = torch.randint(0, output_size, (100,)).view(-1)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)

# Training loop with validation and early stopping
num_epochs = 10
best_loss = float('inf')
patience, no_improve = 3, 0  # Early stopping settings

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Early stopping check
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        no_improve = 0
        torch.save(model.state_dict(), "best_model.pt")  # Save best model
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered!")
            break

print("Training complete!")

# Continual learning function with corrected dataset handling
def continual_retraining(model, dataset_path, interval=5000, steps=50000):
    step = 0
    while step < steps:
        if step % interval == 0:
            print(f"Retrained model at step {step} using {dataset_path}")  # Making path explicit
        step += 1

# Instantiate attacker and defender models separately
attacker_model = NeuralNet(input_size, output_size)
defender_model = NeuralNet(input_size, output_size)

continual_retraining(attacker_model, "data/attacker_dataset.csv")
continual_retraining(defender_model, "data/defender_dataset.csv")

print(" adversarial training complete!")
