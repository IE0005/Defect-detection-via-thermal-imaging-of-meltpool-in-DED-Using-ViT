
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import pandas as pd
from PIL import Image
import os
import timm
import numpy as np

# Dataset class
class ImageDatasetWithLabels(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

# MLP Head for Classification
class MLPHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(MLPHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct = 0.0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
    return running_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    return running_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    dataset = ImageDatasetWithLabels(args.csv_file, args.image_folder, transform=transform)
    kfold = KFold(n_splits=6, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}")
        train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx),
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx),
                                batch_size=args.batch_size, shuffle=False)

        # Load pretrained MAE encoder and freeze
        encoder = timm.create_model('vit_base_patch16_224', pretrained=False)
        checkpoint = torch.load(args.mae_weights_path, map_location=device)
        encoder.load_state_dict(checkpoint['model'], strict=False)
        encoder.to(device)
        for param in encoder.parameters():
            param.requires_grad = False

        # Only use encoder forward and attach MLP head
        class EncoderWithMLP(nn.Module):
            def __init__(self, encoder, num_classes):
                super().__init__()
                self.encoder = encoder
                self.head = MLPHead(in_features=768, num_classes=num_classes)

            def forward(self, x):
                x = self.encoder.forward_features(x)
                return self.head(x)

        model = EncoderWithMLP(encoder, args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAE + MLP Classification with 6-fold CV")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with labels")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder with images")
    parser.add_argument("--mae_weights_path", type=str, required=True, help="Path to pretrained MAE weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size (default 224)")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    args = parser.parse_args()
    main(args)
