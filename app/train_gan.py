
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.gan_model import Generator, Discriminator


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    epochs = 10

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        total_g_loss = 0
        total_d_loss = 0

        for real_images, _ in loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train the Discriminator
            optimizer_d.zero_grad()

            real_outputs = discriminator(real_images)
            real_loss = criterion(real_outputs, real_labels)

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)

            fake_outputs = discriminator(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train the Generator
            optimizer_g.zero_grad()

            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)

            g_loss.backward()
            optimizer_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"D Loss: {total_d_loss / len(loader):.4f}, "
            f"G Loss: {total_g_loss / len(loader):.4f}",
            flush=True
        )

    torch.save(generator.state_dict(), "app/gan_generator.pth")
    print("Generator saved to app/gan_generator.pth")


if __name__ == "__main__":
    train()
