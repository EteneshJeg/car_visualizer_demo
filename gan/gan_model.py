import torch
import torch.nn as nn


# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), 3, 64, 64)  # Reshape to (Batch, Channels, Height, Width)
        return x


# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # Output: Single value (real or fake)
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.model(x)


# Weight Initialization Function
def weights_init(m):
    """Initialize weights for Generator and Discriminator."""
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# Instantiate models
generator = Generator()
discriminator = Discriminator()

# Apply weight initialization
generator.apply(weights_init)
discriminator.apply(weights_init)

# Print models (optional)
print(generator)
print(discriminator)
