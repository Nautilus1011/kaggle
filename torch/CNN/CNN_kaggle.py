import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split



trainPath = "/home/jinysd/workspace/datasetFolder/digit/train.csv"
testPath = "/home/jinysd/workspace/datasetFolder/digit/test.csv"


train_df = pd.read_csv(trainPath)
test_df = pd.read_csv(testPath)

x = train_df.iloc[:, 1:].values
y = train_df.iloc[:, 0].values

x = x / 255.0

x = x.reshape(-1, 28, 28)



# Create PyTorch datasets and dataloaders
class DigitDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(28, 28, 1).astype('float32')
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        else:
            return img

# Define transformations
transform = transforms.ToTensor()

# Create datasets and dataloaders for PyTorch
train_dataset_pt = DigitDataset(x, y, transform=transform)
train_loader_pt = DataLoader(train_dataset_pt, batch_size=64, shuffle=True)

# Create test dataset and dataloader for PyTorch
test_dataset_pt = DigitDataset(test_df.values, transform=transform)
test_loader_pt = DataLoader(test_dataset_pt, batch_size=64, shuffle=False)








# Create CNN Model
class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# Create PyTorch model instance
model_pt = DigitModel()



# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_pt.to(device)

# Define loss function and optimizer for PyTorch
criterion_pt = nn.CrossEntropyLoss()
optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001)

# Training loop for PyTorch model
num_epochs = 10
for epoch in range(num_epochs):
    model_pt.train()
    running_loss = 0.0
    for imgs, labels in train_loader_pt:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_pt.zero_grad()
        outputs = model_pt(imgs)
        loss = criterion_pt(outputs, labels)
        loss.backward()
        optimizer_pt.step()
        running_loss += loss.item()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader_pt):.4f}')



    # Evaluation on validation data for PyTorch model
model_pt.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in train_loader_pt:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model_pt(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'PyTorch Model Accuracy on Training Data: {100 * correct / total:.2f}%')


# Make predictions on test data using PyTorch model
model_pt.eval()
predictions_pt = []
with torch.no_grad():
    for imgs in test_loader_pt:  # Use test_loader_pt here
        imgs = imgs.to(device)
        outputs = model_pt(imgs)
        _, predicted = torch.max(outputs, 1)
        predictions_pt.extend(predicted.cpu().numpy())

# Ensure predictions_pt has 28000 rows
assert len(predictions_pt) == 28000, "Number of predictions should be 28000"
