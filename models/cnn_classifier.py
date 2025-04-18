import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import datasets, transforms

class ConvBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
            nn.Dropout(0.5),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out


class CNNClassifier(nn.Module):
    def __init__(self, in_channel=3, channel=128, n_conv_block=2, n_conv_channel=32, num_classes=10):
        super().__init__()
        # structure similar to S-VQVAE encoder
        blocks = [
            nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), 
            nn.Conv2d(channel, channel, 3, padding=1),
        ]

        for _ in range(n_conv_block):
            blocks.append(ConvBlock(channel, n_conv_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.feature_extractor = nn.Sequential(*blocks)

        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), 
            nn.Linear(channel, num_classes),
            nn.Dropout(0.5), 
        )

    def forward(self, input):
        features = self.feature_extractor(input) 
        logits = self.classifier(features)
        return logits


def train_classifier(model, train_loader, val_loader, num_epochs=100, lr=1e-3, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    best_val_loss = float('inf')
    
    model = model.to(device)
    patience_counter = 0  # counter for early stopping

    # create the folder to save the best model
    save_dir = "best_model"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "cnn_classifier.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with Validation Loss: {best_val_loss:.4f} at epoch {epoch+1}")
            patience_counter = 0  # reset counter on improvement
        else:
            patience_counter += 1

        # check for early stopping
        if patience_counter >= 100:
            print(f"No improvement in accuracy for 50 epochs. Early stopping at epoch {epoch+1}.")
            break


def get_data_loaders(data_path, batch_size=128, train_split=0.8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize CIFAR-10 images
    ])

    # load CIFAR-10 dataset
    dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    # parameters
    data_path = './data'  # path to CIFAR-10 dataset
    batch_size = 128
    num_epochs = 1000
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get data loaders for CIFAR-10
    train_loader, val_loader = get_data_loaders(data_path, batch_size=batch_size)

    # instantiate the CNN model
    model = CNNClassifier(num_classes=10)  # CIFAR-10 has 10 classes

    # train the classifier
    train_classifier(model, train_loader, val_loader, num_epochs=num_epochs, lr=learning_rate, device=device)
