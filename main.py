from gan.train_gan import train_gan
from gan.gan_model import Generator
import torch
from gan.train_gan import show_generated_images

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_images():
    """
    Generate and display images using a pre-trained generator.
    """
    print("Loading pre-trained generator model...")
    generator = Generator().to(device)

    print("Pre-trained generator model loaded.")

    # Generate random noise
    noise = torch.randn(16, 100).to(device)

    # Generate images
    with torch.no_grad():
        fake_images = generator(noise)

    # Display generated images
    show_generated_images(fake_images, epoch=0)


if __name__ == "__main__":
    print("Choose an option:")
    print("1: Train GAN")
    print("2: Generate Images")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        print("Starting GAN training...")
        train_gan()
    elif choice == "2":
        print("Generating images...")
        generate_images()
    else:
        print("Invalid choice. Exiting.")
