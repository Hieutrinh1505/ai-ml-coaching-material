"""
Inference script for testing the trained Emotion Detection model.

This script demonstrates how to:
1. Load a pre-trained EmotionCNN model
2. Preprocess a single image for inference
3. Run prediction and interpret results

Usage:
    python inference.py

Note: Modify 'image_path' variable to test different images.
"""

import sys
from pathlib import Path

# Add parent directory to path for importing custom modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model import EmotionCNN
import torch
from torchvision.transforms import v2
from PIL import Image

# ============================================================
# MODEL SETUP
# ============================================================

# Detect available device (Apple Silicon MPS or CPU fallback)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Initialize model architecture with 7 emotion classes
model = EmotionCNN(num_classes=7)

# Load pre-trained weights
# map_location ensures weights load correctly regardless of training device
model.load_state_dict(torch.load('../model/best_model.pth', map_location=device))

# Move model to device and set to evaluation mode
# eval() disables dropout and uses running stats for batch normalization
model.to(device)
model.eval()

print(f"✓ Model loaded on {device}")

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

# Transform pipeline - MUST match validation transforms used during training
# Key steps:
# 1. Convert to grayscale (model expects single channel input)
# 2. Resize to 224x224 (model's expected input size)
# 3. Convert to tensor and normalize to [-1, 1] range
inference_transform = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])

# Emotion class names - ORDER MUST MATCH training dataset folder order (alphabetical)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ============================================================
# INFERENCE
# ============================================================

# Load test image
image_path = "../images/train/angry/0.jpg"
image = Image.open(image_path)

# Preprocess: apply transforms and add batch dimension
# unsqueeze(0) adds batch dimension: (1, 224, 224) -> (1, 1, 224, 224)
image_tensor = inference_transform(image).unsqueeze(0).to(device)

# Run inference with gradient computation disabled (faster, less memory)
with torch.no_grad():
    # Get raw model outputs (logits)
    logits = model(image_tensor)

    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(logits, dim=1)

    # Get predicted class index (highest probability)
    predicted_class = torch.argmax(probabilities, dim=1)

    # Debug outputs
    print(f"Logits: {logits}")
    print(f"Probabilities: {probabilities}")
    print(f"Predicted class index: {predicted_class.item()}")

# ============================================================
# DISPLAY RESULTS
# ============================================================

print(f"\nPredicted Emotion: {class_names[predicted_class.item()]}")
print(f"Confidence: {probabilities[0, predicted_class].item():.2%}")