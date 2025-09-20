import torch
from PIL import Image
from torchvision import transforms
from encoder import VAE_Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess image
image_path = "./dog.jpg"
img = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])
img_tensor = transform(img).unsqueeze(0).to(device)  # (1,3,512,512)

# Initialize encoder
encoder = VAE_Encoder().to(device)
encoder.eval()

# Create noise tensor for sampling
noise = torch.randn(1, 4, 64, 64).to(device)

# Forward pass
with torch.no_grad():
    output = encoder(img_tensor, noise)

print("Output shape:", output.shape)  # (1,4,64,64)
