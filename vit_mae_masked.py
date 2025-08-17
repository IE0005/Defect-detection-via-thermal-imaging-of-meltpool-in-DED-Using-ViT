import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MAE ViT Masked Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='./mae_model.pth', help='Path to save the model')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    save_path = args.save_path
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
    import os
    from transformers import ViTMAEForPreTraining
    import matplotlib.pyplot as plt
    import random

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 for ViT
        transforms.ToTensor(),          # Convert images to tensors
    ])

    # Custom dataset to load images directly from a folder
    class ImageDatasetFromFolder(Dataset):
        def __init__(self, image_folder, transform=None):
            self.image_folder = image_folder
            self.image_filenames = sorted([
                f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            self.transform = transform

            if not os.path.isdir(image_folder):
                raise ValueError(f"Directory '{image_folder}' does not exist or cannot be accessed.")

        def __len__(self):
            return len(self.image_filenames)

        def __getitem__(self, idx):
            img_name = os.path.join(self.image_folder, self.image_filenames[idx])  # Get image path
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.image_filenames[idx]

    # Usage
    # Use args.data_dir instead of hardcoded path
    image_dir = data_dir
    dataset = ImageDatasetFromFolder(image_folder, transform=transform)

    # Split dataset into train/val sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Define the masking function
    def mask_patches(images, mask_ratio=0.75, patch_size=16):
        batch_size, channels, height, width = images.shape
        num_patches = (height // patch_size) * (width // patch_size)
        num_masked_patches = int(mask_ratio * num_patches)

        masked_images = images.clone()
        masks = torch.zeros(batch_size, num_patches, dtype=torch.bool)

        for i in range(batch_size):
            mask_indices = random.sample(range(num_patches), num_masked_patches)
            masks[i, mask_indices] = True

            for idx in mask_indices:
                row = (idx // (width // patch_size)) * patch_size
                col = (idx % (width // patch_size)) * patch_size
                masked_images[i, :, row:row+patch_size, col:col+patch_size] = 0

        return masked_images, masks

    # Initialize the ViTMAEForPreTraining model
    mae_model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')

    # Move model to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mae_model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mae_model.parameters(), lr=1e-4)

    # Function to reconstruct the image from patches
    def reconstruct_from_patches(patches, img_size=224, patch_size=16):
        batch_size = patches.size(0)
        num_patches_per_row = img_size // patch_size
        patches = patches.view(batch_size, num_patches_per_row, num_patches_per_row, 3, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        images = patches.contiguous().view(batch_size, 3, img_size, img_size)
        return images

    # Function to save reconstructed images
    def save_reconstructed_images(reconstructed_images, filenames, save_folder):
        os.makedirs(save_folder, exist_ok=True)
        for i, image_tensor in enumerate(reconstructed_images):
            img = transforms.ToPILImage()(image_tensor.cpu())
            img.save(os.path.join(save_folder, filenames[i]))

    # Train MAE for image reconstruction
    def train_mae(mae_model, train_loader, val_loader, num_epochs, save_path, loss_plot_path, recon_save_folder, mask_ratio=0.75):
        best_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            total_train_loss = 0
            total_val_loss = 0

            # Training phase
            mae_model.train()
            for images, _ in train_loader:
                images = images.to(device)
                masked_images, _ = mask_patches(images, mask_ratio)  # Apply masking

                outputs = mae_model(masked_images).logits
                reconstructed_images = reconstruct_from_patches(outputs)
                loss = criterion(reconstructed_images, images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

            # Validation phase
            mae_model.eval()
            reconstructed_images = []
            filenames = []
            with torch.no_grad():
                for images, filename in val_loader:
                    images = images.to(device)
                    masked_images, _ = mask_patches(images, mask_ratio)  # Apply masking for validation as well

                    outputs = mae_model(masked_images).logits
                    reconstructed_batch = reconstruct_from_patches(outputs)
                    reconstructed_images.extend(reconstructed_batch)
                    filenames.extend(filename)

                    val_loss = criterion(reconstructed_batch, images)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

            # Save the best model and its reconstructed images
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(mae_model.state_dict(), save_path)
                print("Best model saved!")
                best_recon_folder = os.path.join(recon_save_folder, 'best_epoch')
                save_reconstructed_images(reconstructed_images, filenames, best_recon_folder)

        # Plot and save the loss curve
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
        plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(loss_plot_path)
        plt.show()

    # Set parameters and train the MAE model
    num_epochs = 100
    train_mae(mae_model, train_loader, val_loader, num_epochs, save_path, loss_plot_path, recon_save_folder, mask_ratio=0.75)
