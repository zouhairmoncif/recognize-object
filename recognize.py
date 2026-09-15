import torch

from torchvision import models
from torchvision.models import ResNet18_Weights
from PIL import Image

# Load pretrained weights
weights = ResNet18_Weights.IMAGENET1K_V1

# Load ResNet18
model = models.resnet18(weights=weights)
model.eval()

# Preprocessing
preprocess = weights.transforms()

# Load image
img = Image.open("image.jpg")

# Prepare image
batch = preprocess(img).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(batch)

# Convert output to probabilities
probs = torch.nn.functional.softmax(output[0], dim=0)

# Get predicted class
class_id = probs.argmax().item()

# Get class name
class_name = weights.meta["categories"][class_id]

# Get confidence
confidence = probs[class_id].item()

print(f"Prediction: {class_name} ({confidence * 100:.2f}%)")