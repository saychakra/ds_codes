## import the necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

########################################################## Discriminator ##########################################################
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)

############################################################ Generator ############################################################
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)

########################################################### Weight Initialization #################################################
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

############################################################ Hyperparameters ######################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

lr = 1e-4
z_dim = 64
img_dim = 28 * 28
batch_size = 32
num_epochs = 50

# model setup
disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
initialize_weights(disc)
initialize_weights(gen)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# data loading
dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# optimizers and loss
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

# tensorboard
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
writer_loss = SummaryWriter(f"runs/GAN_MNIST/loss")
step = 0

# training loop
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, img_dim).to(device)
        batch_size = real.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake.detach()).view(-1)
        lossD_real = criterion(disc_real, torch.full_like(disc_real, 0.9))  # label smoothing
        lossD_fake = criterion(disc_fake, torch.full_like(disc_fake, 0.1))  # noisy label
        lossD = (lossD_real + lossD_fake) / 2

        opt_disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # Train Generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))  # pretend real
        opt_gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # TensorBoard logging
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                real_img = real.reshape(-1, 1, 28, 28)

                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True, nrow=8)
                img_grid_real = torchvision.utils.make_grid(real_img, normalize=True, nrow=8)
                comparison_grid = torch.cat((img_grid_real, img_grid_fake), dim=1)  # stack vertically

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Real Images", img_grid_real, global_step=step)
                writer_fake.add_image("Real vs Fake", comparison_grid, global_step=step)

                writer_loss.add_scalar("Generator Loss", lossG.item(), global_step=step)
                writer_loss.add_scalar("Discriminator Loss", lossD.item(), global_step=step)
                step += 1

# close writers
writer_fake.close()
writer_real.close()
writer_loss.close()