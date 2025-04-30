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
            nn.LeakyReLU(0.1), ## leaky ReLU is often times a better choice in GANs - currently the slope is set to 0.1 (can play around with it if required)
            nn.Linear(128, 1), ## output of the final value
            nn.Sigmoid(), # fake = 0 and real = 1. To bring values within a probability range we're calling sigmoid
        )

    def forward(self, x):
        return self.disc(x)

############################################################ Generator ############################################################
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(), ## for normalizing the input within the range of -1 to 1
        )

    def forward(self, x):
        return self.gen(x)

############################################################ Hyperparameters #######################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 ## since karpathay said so!
z_dim = 64 # 128, 256 etc
img_dim = 28 * 28 * 1 # 784
batch_size = 32
num_epochs = 50

## calling stuff
disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ## train for discriminator - max log(D(real)) + log(1-D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ## train the generator and minimize log(1 - D(G(z))) --> this expression leads to weak gradients and leads to slower / no training sometimes. Better is to max log(D(G(z))) 
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        ## additional code for tensorboard
        if batch_idx == 0:
            print(
                f"Epoch [{epoch} / {num_epochs}] \n "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "MNIST Fake Images", img_grid_real, global_step=step
                )
                
                step += 1
