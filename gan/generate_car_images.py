import torch
import matplotlib.pyplot as plt
import torchvision


def show_generated_images(images, epoch, num_images=16):
    """
    Display generated images during training.

    Args:
        images (Tensor): Generated images tensor with shape (B, C, H, W).
        epoch (int): Current training epoch.
        num_images (int): Number of images to display in the grid.
    """
    images = (images + 1) / 2  # Rescale to [0, 1]
    grid = torchvision.utils.make_grid(images[:num_images], nrow=4, normalize=True, value_range=(0, 1))
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Generated Images at Epoch {epoch + 1}")
    plt.axis("off")
    plt.show()


# Example training loop with epoch definition
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dummy example for the generator
class DummyGenerator(torch.nn.Module):
    def __init__(self):
        super(DummyGenerator, self).__init__()
        self.fc = torch.nn.Linear(100, 3 * 64 * 64)

    def forward(self, x):
        return self.fc(x).view(-1, 3, 64, 64)


# Initialize generator and move to device
generator = DummyGenerator().to(device)

# Training loop
num_epochs = 50
batch_size = 16

for epoch in range(num_epochs):
    # Generate dummy data for training (replace this with real training data)
    real_imgs = torch.randn(batch_size, 3, 64, 64).to(device)  # Replace with dataset batch

    # Generate fake images using the generator
    noise = torch.randn(batch_size, 100).to(device)  # Noise vector
    fake_imgs = generator(noise)

    # Visualize generated images every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            show_generated_images(fake_imgs, epoch)
