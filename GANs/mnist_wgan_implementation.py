import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


######################################################## Critic ########################################################
class Critic(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),  # no Sigmoid
        )

    def forward(self, x):
        return self.model(x)

######################################################## Generator ########################################################
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

##################################################### Weight Initialization #################################################
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

######################################################## Hyperparameters ########################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 5e-5
z_dim = 64
img_dim = 28 * 28
batch_size = 64
num_epochs = 50
n_critic = 5
weight_clip = 0.01

# Setup
critic = Critic(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
initialize_weights(critic)
initialize_weights(gen)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizers
opt_critic = optim.RMSprop(critic.parameters(), lr=lr)
opt_gen = optim.RMSprop(gen.parameters(), lr=lr)

# TensorBoard
writer_fake = SummaryWriter("runs/WGAN_MNIST/fake")
writer_real = SummaryWriter("runs/WGAN_MNIST/real")
writer_loss = SummaryWriter("runs/WGAN_MNIST/loss")
step = 0

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, img_dim).to(device)
        batch_size = real.size(0)

        # Train Critic
        for _ in range(n_critic):
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            critic_real = critic(real).view(-1)
            critic_fake = critic(fake.detach()).view(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))

            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

            # Weight clipping
            for p in critic.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

        # Train Generator
        output = critic(fake).view(-1)
        loss_gen = -torch.mean(output)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Logging
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Loss Critic: {loss_critic:.4f}, Loss Gen: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                real_img = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True, nrow=8)
                img_grid_real = torchvision.utils.make_grid(real_img, normalize=True, nrow=8)
                comparison_grid = torch.cat((img_grid_real, img_grid_fake), dim=1)

                writer_fake.add_image("Fake Images", img_grid_fake, global_step=step)
                writer_real.add_image("Real Images", img_grid_real, global_step=step)
                writer_fake.add_image("Real vs Fake", comparison_grid, global_step=step)
                writer_loss.add_scalar("Generator Loss", loss_gen.item(), global_step=step)
                writer_loss.add_scalar("Critic Loss", loss_critic.item(), global_step=step)
                step += 1

# Close writers
writer_fake.close()
writer_real.close()
writer_loss.close()
