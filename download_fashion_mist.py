import os
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from PIL import Image

save_path = 'data/fashion_mnist/images'
os.makedirs(save_path, exist_ok=True)

# Resizing the images to 512x512 and convert to RGB as StyleGAN2 expects RGB
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3)
])

dataset = FashionMNIST(root='data', train=True, download=True)

print(f"Saving {len(dataset)} images to {save_path}...")

for i, (img, label) in enumerate(dataset):
    img = transform(img)
    img.save(os.path.join(save_path, f'{i:05d}.png'))

print("Done!")