
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='ViT Classifier with Cross Validation')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--label_csv', type=str, required=True, help='Path to CSV file with labels')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_folds', type=int, default=6, help='Number of cross-validation folds')
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    image_folder = args.image_dir
    csv_file = args.label_csv
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    num_folds = args.num_folds

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms
from transformers import ViTModel
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),   # Resize images to 224x224 for ViT
    transforms.ToTensor(),          # Convert images to tensors
])

# Custom dataset to load images and labels from CSV
class ImageDatasetWithLabels(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the ViT-based classifier model with dropout
class ViTClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout_rate=0.5):
        super(ViTClassifier, self).__init__()
        self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, x):
        outputs = self.encoder(x)
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

# Load the dataset
image_folder = '/home/jovyan/stephenie-storage/israt_files/Thermal_images/all_images_cropped'
csv_file = '/home/jovyan/stephenie-storage/israt_files/Thermal_images/modified_labels_09_10.csv'
dataset = ImageDatasetWithLabels(csv_file, image_folder, transform=transform)

# Initialize cross-validation parameters
kf = KFold(n_splits=6, shuffle=True, random_state=42)
num_epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training function for each fold
def train_classifier(model, train_loader, val_loader, num_epochs, best_model_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

    return train_losses, val_losses

# Testing function
def test_model(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_test_loss = 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return avg_test_loss, accuracy, precision, recall, f1, conf_matrix

# Perform 6-fold cross-validation
fold = 1
for train_idx, test_idx in kf.split(dataset):
    # Split data into train, validation, and test sets
    fold_train_val_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)
    train_size = int(0.85 * len(fold_train_val_dataset))
    val_size = len(fold_train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(fold_train_val_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize and load model
    model = ViTClassifier(num_labels=2, dropout_rate=0.5).to(device)
    mae_weights_path = '/home/jovyan/stephenie-storage/israt_files/Thermal_images/best_mae_vit_model_7800_first.pth'
    mae_model = torch.load(mae_weights_path, map_location=device)
    model.encoder.load_state_dict(mae_model, strict=False)

    # Train model
    best_model_path = f'/home/jovyan/stephenie-storage/israt_files/Thermal_images/vit_class_6_cross_val/best_vit_classification_fold_{fold}.pth'
    train_losses, val_losses = train_classifier(model, train_loader, val_loader, num_epochs, best_model_path)

    # Plot train and val loss for each fold
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold} Training and Validation Loss')
    plt.legend()
    plt.savefig(f'/home/jovyan/stephenie-storage/israt_files/Thermal_images/vit_class_6_cross_val/loss_curve_fold_{fold}.png')
    plt.show()

    # Load best model and test
    model.load_state_dict(torch.load(best_model_path))
    avg_test_loss, accuracy, precision, recall, f1, conf_matrix = test_model(model, test_loader)

    print(f'Fold {fold} Test Loss: {avg_test_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    fold += 1
