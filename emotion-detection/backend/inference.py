import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model import EmotionCNN
import torch
from torchvision.transforms import v2
from PIL import Image
# Detect device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Create model
model = EmotionCNN(num_classes=7)

# Load weights with device mapping (handles CPU/GPU differences)
model.load_state_dict(torch.load('../model/best_model.pth', map_location=device))

# Move model to device
model.to(device)
model.eval()

print(f"✓ Model loaded on {device}")

# Transform (same as validation)
inference_transform = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])

# Class names (must match training order!)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load and predict
image_path = "../images/train/angry/0.jpg"
image = Image.open(image_path)
image_tensor = inference_transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    predicted_class = predicted_class * 100
    
    print(logits)
    print(probabilities)
    print(predicted_class)

# Display
print(predicted_class.item())

print(f"Predicted: {class_names[predicted_class.item()]}")
print(f"Confidence: {probabilities[0, predicted_class].item():.2%}")