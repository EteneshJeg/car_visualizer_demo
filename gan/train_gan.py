import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from gan_model import Generator, Discriminator

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize(72),
    transforms.CenterCrop((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Path to dataset
dataset_path = "../c5-RunYOLO/cropped_cars"

# Check if the dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The dataset folder '{dataset_path}' does not exist. Ensure you have cropped car images.")

# Load dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Instantiate models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Hyperparameters
learning_rate = 0.0002
betas = (0.5, 0.999)  # Adam optimizer parameters
num_epochs = 50

optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)

# Loss function
criterion = torch.nn.BCELoss()

# Function to display generated images
def show_generated_images(images, epoch, num_images=16):
    """
    Display generated images during training.
    """
    images = (images + 1) / 2  # Rescale to [0, 1]
    grid = vutils.make_grid(images[:num_images], nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Generated Images at Epoch {epoch + 1}")
    plt.axis("off")
    plt.show()

def train_gan(num_epochs=50):
    """
    Train the GAN for a given number of epochs.
    """
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            # Move real images to the device
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Train Discriminator
            noise = torch.randn(batch_size, 100).to(device)  # Random noise for generator input
            fake_imgs = generator(noise)

            # Real and fake labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Forward pass through discriminator
            real_output = discriminator(real_imgs)
            fake_output = discriminator(fake_imgs.detach())

            # Discriminator loss (real vs. fake)
            loss_d_real = criterion(real_output, real_labels)
            loss_d_fake = criterion(fake_output, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2

            # Backpropagate and update discriminator
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            fake_output = discriminator(fake_imgs)
            loss_g = criterion(fake_output, real_labels)

            # Backpropagate and update generator
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            # Print batch progress
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(dataloader)}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

        # Display images every 10 epochs
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                noise = torch.randn(16, 100).to(device)  # Generate 16 fake images for display
                fake_imgs = generator(noise)
                show_generated_images(fake_imgs, epoch)
