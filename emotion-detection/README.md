# Facial Emotion Detection

A real-time facial emotion detection system using a custom CNN architecture. The project includes training scripts, inference utilities, and a Streamlit web application for live webcam-based emotion recognition.

## Project Structure

```
emotion-detection/
├── backend/
│   ├── train.py              # Model training script
│   └── inference.py          # Single image inference demo
├── frontend/
│   └── app.py                # Streamlit web app with live webcam
├── utils/
│   ├── __init__.py
│   ├── model.py              # EmotionCNN architecture
│   └── processing.py         # Data loading & preprocessing
├── model/
│   └── best_model.pth        # Trained model weights (~52MB)
├── images/
│   ├── train/                # Training images by emotion class
│   └── validation/           # Validation images by emotion class
└── notebook/
    ├── data-from-kaggle.ipynb
    ├── eda.ipynb
    ├── image-processing-basic.ipynb
    └── notebook-v1.ipynb
```

## Emotion Classes

The model classifies faces into **7 emotion categories** (alphabetical order):

| Class | Description |
|-------|-------------|
| angry | Anger expression |
| disgust | Disgust expression |
| fear | Fear expression |
| happy | Happiness/smile |
| neutral | Neutral/no expression |
| sad | Sadness expression |
| surprise | Surprise expression |

---

## Model Architecture

### EmotionCNN

A custom 3-block CNN designed for grayscale facial images.

```
Input: (batch, 1, 224, 224) - Grayscale images

┌─────────────────────────────────────────────────────────┐
│ ConvBlock 1: 1 → 32 channels                            │
│   Conv2d(3x3, padding=1) → BatchNorm → ReLU → MaxPool   │
│   Output: (batch, 32, 112, 112)                         │
├─────────────────────────────────────────────────────────┤
│ ConvBlock 2: 32 → 64 channels                           │
│   Conv2d(3x3, padding=1) → BatchNorm → ReLU → MaxPool   │
│   Output: (batch, 64, 56, 56)                           │
├─────────────────────────────────────────────────────────┤
│ ConvBlock 3: 64 → 128 channels                          │
│   Conv2d(3x3, padding=1) → BatchNorm → ReLU → MaxPool   │
│   Output: (batch, 128, 28, 28)                          │
├─────────────────────────────────────────────────────────┤
│ Flatten: 128 × 28 × 28 = 100,352 features               │
├─────────────────────────────────────────────────────────┤
│ FC1: 100,352 → 128 + BatchNorm + ReLU + Dropout(0.5)    │
├─────────────────────────────────────────────────────────┤
│ FC2: 128 → 7 (raw logits, no softmax)                   │
└─────────────────────────────────────────────────────────┘

Output: (batch, 7) - Logits for each emotion class
```

### Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Input | Grayscale | Emotion is shape-based, color not needed |
| Kernel size | 3×3 | Standard for feature extraction |
| Pooling | MaxPool 2×2 | Reduces dimensions while preserving features |
| Dropout | 50% | Prevents overfitting on limited data |
| Output | Raw logits | CrossEntropyLoss applies softmax internally |

---

## Data Pipeline

### Preprocessing Transforms

**Training** (with augmentation):
```python
Grayscale → Resize(224×224) → RandomHorizontalFlip(0.5) →
ColorJitter(brightness=0.2, contrast=0.2) → ToTensor → Normalize(0.5, 0.5)
```

**Validation/Inference** (no augmentation):
```python
Grayscale → Resize(224×224) → ToTensor → Normalize(0.5, 0.5)
```

### Dataset Structure

Images are organized using PyTorch's `ImageFolder` convention:
```
images/
├── train/
│   ├── angry/       # Training images for "angry" class
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── validation/
    ├── angry/       # Validation images for "angry" class
    └── ...
```

---

## Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 50 |
| Batch Size | 32 |
| Loss Function | CrossEntropyLoss |
| Device | MPS (Apple Silicon) / CPU |

### Run Training

```bash
cd backend
python train.py
```

The script will:
1. Load train/validation datasets from `images/` folder
2. Train for 50 epochs with progress bars
3. Save the best model (by validation accuracy) as `best_model.pth`
4. Print epoch-by-epoch loss and accuracy metrics

### Output Example

```
Using device: mps
Number of classes: 7
Training samples: 28709
Validation samples: 7178

Epoch [1/50]
Training: 100%|██████████| 897/897 [01:23<00:00, loss: 1.2345, acc: 45.67%]
Validation: 100%|██████████| 225/225 [00:15<00:00, loss: 1.0123, acc: 52.34%]
✓ New best model saved! (Val Acc: 52.34%)
```

---

## Inference

### Single Image Testing

```bash
cd backend
python inference.py
```

Modify `image_path` in the script to test different images.

### Programmatic Usage

```python
from utils.model import EmotionCNN
import torch
from torchvision.transforms import v2
from PIL import Image

# Load model
model = EmotionCNN(num_classes=7)
model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu'))
model.eval()

# Preprocess image
transform = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])

image = Image.open('path/to/face.jpg')
tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    predicted = torch.argmax(probs, dim=1).item()

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
print(f"Emotion: {emotions[predicted]}, Confidence: {probs[0, predicted]:.2%}")
```

---

## Real-Time Web Application

### Features

- Live webcam feed with face detection
- Real-time emotion probability bar chart
- Exponential smoothing for stable predictions
- Multi-face detection support

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web UI | Streamlit | Interactive interface |
| Video Streaming | streamlit_webrtc | Webcam access |
| Face Detection | MTCNN | Locate faces in frame |
| Emotion Model | EmotionCNN | Classify emotions |

### Run the App

```bash
cd frontend
streamlit run app.py
```

### Performance Optimizations

1. **Inference Interval**: Runs model every 3 frames (not every frame)
2. **Dual Smoothing**:
   - Model-side: EMA with α=0.7 for raw predictions
   - Display-side: EMA with α=0.85 for chart updates
3. **Thread Safety**: Locks protect shared emotion probability data

---

## Key Takeaways

### Architecture Decisions

1. **Grayscale Input**: Emotions are conveyed through facial structure, not color. Using grayscale reduces model complexity and training data requirements.

2. **Progressive Channel Increase** (1→32→64→128): Standard CNN pattern that captures increasingly complex features at each layer.

3. **BatchNorm After Every Conv**: Stabilizes training and allows higher learning rates.

4. **High Dropout (50%)**: Essential for small-to-medium datasets to prevent overfitting.

5. **No Softmax in Model**: CrossEntropyLoss expects raw logits; applying softmax manually during inference.

### Data Augmentation Strategy

- **RandomHorizontalFlip**: Faces can appear from either side
- **ColorJitter**: Simulates lighting variations
- **No Rotation**: Extreme rotations are uncommon in real use cases

### Real-Time Processing

1. **MTCNN for Face Detection**: Robust multi-task CNN that handles varying face sizes and angles.

2. **Exponential Moving Average**: Prevents jittery predictions by smoothing over time:
   ```
   smoothed = α × previous + (1-α) × current
   ```
   Higher α = slower, smoother transitions.

3. **Separate Model/Display Smoothing**: Decouples inference stability from UI aesthetics.

### Common Pitfalls Avoided

| Pitfall | Solution in This Project |
|---------|-------------------------|
| Loading model per frame | Model loaded once in `__init__` |
| Thread-unsafe data access | Lock protects `emotion_probs` dict |
| Inconsistent transforms | Same validation transforms for inference |
| Class order mismatch | Alphabetical folder order = class indices |

---

## Dependencies

```
torch
torchvision
streamlit
streamlit-webrtc
facenet-pytorch
opencv-python
pandas
tqdm
Pillow
av
```

---

## File Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `backend/train.py` | Training loop | Epoch iteration, model saving |
| `backend/inference.py` | Demo inference | Single image prediction |
| `frontend/app.py` | Web application | Webcam + real-time display |
| `utils/model.py` | CNN architecture | `ConvBlock`, `EmotionCNN` |
| `utils/processing.py` | Data loading | `ImageProcessing` class |

---

## Future Improvements

- [ ] Add confusion matrix visualization after training
- [ ] Implement early stopping
- [ ] Add learning rate scheduler
- [ ] Support for multiple faces with individual emotion labels
- [ ] Export model to ONNX for deployment
- [ ] Add data augmentation options (rotation, scaling)
