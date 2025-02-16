import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CustomNet  # Import the CustomNet model
from tqdm import tqdm
from torchsummary import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Calculate the mean and std of the CIFAR-10 dataset for normalization
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# Define the transformation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=mean),
    A.Normalize(mean=mean, std=std),  # Normalize the image
    ToTensorV2()
])

# Wrapper class for Albumentations transformations
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        # Ensure the image is passed as a named argument
        return self.transforms(image=np.array(img))['image']

# Load CIFAR-10 dataset
transform_wrapper = Transforms(transform)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_wrapper, download=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize the model
model = CustomNet().to(device)

# Print model summary
summary(model, input_size=(3, 32, 32))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, device, train_loader, test_loader, optimizer, criterion, num_epochs=35):
    for epoch in range(num_epochs):
        model.train()
        correct_train = 0
        total_train = 0
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", mininterval=1)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update progress bar with current training accuracy
            train_accuracy = 100 * correct_train / total_train
            progress_bar.set_postfix(loss=running_loss/total_train, accuracy=train_accuracy)

        # Calculate test accuracy after each epoch
        test_accuracy = test(model, device, test_loader)
        print(f"Test Accuracy after Epoch {epoch+1}: {test_accuracy:.2f}%")

# Testing loop
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    train(model, device, train_loader, test_loader, optimizer, criterion)