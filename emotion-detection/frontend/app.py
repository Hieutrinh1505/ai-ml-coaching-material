"""
Real-time Emotion Detection Web Application using Streamlit.

This application provides a live webcam feed with real-time facial emotion detection.
It uses:
- Streamlit for the web UI
- streamlit_webrtc for webcam streaming
- MTCNN (Multi-task Cascaded Convolutional Networks) for face detection
- Custom EmotionCNN for emotion classification

Features:
- Real-time face detection with bounding boxes
- Live emotion probability visualization
- Exponential smoothing for stable predictions

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Add parent directory to path for importing custom modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Streamlit and webcam streaming
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

# Face detection
from facenet_pytorch import MTCNN

# Threading for concurrent access to shared data
import threading

# UI and visualization
import streamlit as st
import pandas as pd
import time

# Emotion detection model
from utils.model import EmotionCNN
import torch
from torchvision.transforms import v2
from PIL import Image


class MyProcessor(VideoProcessorBase):
    """
    Video processor for real-time emotion detection.

    Processes each video frame to:
    1. Detect faces using MTCNN
    2. Run emotion classification on detected faces
    3. Apply exponential smoothing for stable predictions
    4. Draw bounding boxes around detected faces

    Attributes:
        mtcnn: MTCNN face detector
        emotion_probs: Dictionary storing smoothed emotion probabilities
        smoothing_factor: Factor for exponential moving average (0.5-0.9)
        inference_interval: Run inference every N frames for performance
    """
    def __init__(self):
        """Initialize the video processor with face detector and emotion model."""
        super().__init__()

        # ============================================================
        # FACE DETECTION SETUP
        # ============================================================
        # MTCNN: Multi-task Cascaded Convolutional Networks for face detection
        # keep_all=True: Detect all faces in frame (not just the largest)
        self.mtcnn = MTCNN(keep_all=True, device="cpu")

        # ============================================================
        # EMOTION PROBABILITY STORAGE
        # ============================================================
        # Smoothing factor for exponential moving average
        # Higher values (0.7-0.9) = smoother but slower response
        # Lower values (0.3-0.5) = faster response but more jittery
        self.smoothing_factor = 0.7

        # Dictionary to store smoothed emotion probabilities
        # Thread-safe access required since recv() runs in separate thread
        self.emotion_probs = {
            "angry": 0,
            "disgust": 0,
            "fear": 0,
            "happy": 0,
            "neutral": 0,
            "sad": 0,
            "surprise": 0,
        }

        # Lock for thread-safe access to emotion_probs
        self.lock = threading.Lock()

        # ============================================================
        # MODEL SETUP
        # ============================================================
        # Detect device: Apple Silicon MPS or CPU fallback
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Image preprocessing pipeline (must match training transforms)
        self.transform = v2.Compose(
            [
                v2.Grayscale(num_output_channels=1),  # Convert to grayscale
                v2.Resize((224, 224)),  # Resize to model input size
                v2.ToTensor(),  # Convert to tensor
                v2.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
            ]
        )

        # Performance optimization: run inference every N frames
        self.inference_interval = 3

        # Emotion labels in alphabetical order (matches training folder structure)
        self.emotion_labels = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ]

        # Load emotion detection model ONCE during initialization
        # Critical: Loading in __init__ avoids repeated loading per frame
        self.model = EmotionCNN(num_classes=7)
        self.model.load_state_dict(
            torch.load("../model/best_model.pth", map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode (disables dropout)

    def model_inference(self, img):
        """
        Run emotion classification on a face image.

        Applies preprocessing, runs model inference, and updates
        the emotion probabilities using exponential moving average.

        Args:
            img: Face image as PIL Image or numpy array
        """
        # Convert numpy array to PIL Image if needed
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        # Preprocess and add batch dimension
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Run inference without gradient computation
        with torch.no_grad():
            logits = self.model(img_tensor)
            probabilities = torch.softmax(logits, dim=1)
            raw_probs = probabilities.squeeze().cpu().tolist()

            # Apply exponential moving average smoothing
            # Thread-safe update of emotion probabilities
            with self.lock:
                for i, emotion in enumerate(self.emotion_labels):
                    # EMA formula: new_value = α * old_value + (1-α) * current_value
                    self.emotion_probs[emotion] = (
                        self.smoothing_factor * self.emotion_probs[emotion]
                        + (1 - self.smoothing_factor) * raw_probs[i]
                    )

    def recv(self, frame):
        """
        Process each video frame from the webcam.

        This method is called for each frame and:
        1. Detects faces using MTCNN
        2. Crops detected faces and runs emotion inference
        3. Draws bounding boxes around detected faces

        Args:
            frame: Input video frame from webcam

        Returns:
            Processed frame with face bounding boxes drawn
        """
        # Convert frame to numpy array (BGR format for OpenCV)
        img = frame.to_ndarray(format="bgr24")

        # Convert BGR to RGB for MTCNN (expects RGB input)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces - returns bounding boxes and landmarks
        boxes, _ = self.mtcnn.detect(img_rgb)

        # Process each detected face
        if boxes is not None:
            for box in boxes:
                # Extract bounding box coordinates
                # MTCNN returns [x1, y1, x2, y2] format
                x1, y1, x2, y2 = [int(b) for b in box]

                # Crop face region for emotion detection
                face = img_rgb[y1:y2, x1:x2]

                # Run emotion detection if face crop is valid
                if face.size > 0:
                    self.model_inference(face)

                # Draw green bounding box around detected face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Return processed frame (convert back to av.VideoFrame)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ============================================================
# STREAMLIT UI LAYOUT
# ============================================================

# Page title
st.title("Real-time Emotion Detection")

# Create two-column layout: video feed (left) and chart (right)
col1, col2 = st.columns([2, 1])

# Left column: Live webcam feed with face detection
with col1:
    st.subheader("Live Video")
    # Initialize webcam streamer with custom video processor
    ctx = webrtc_streamer(key="emotion-detection", video_processor_factory=MyProcessor)

# Right column: Real-time emotion probability chart
with col2:
    st.subheader("Emotion Probabilities")

    # Placeholder for dynamically updating chart
    chart_placeholder = st.empty()

    # Display-side smoothing (separate from model-side smoothing)
    # This provides additional visual smoothing for the chart
    display_probs = {
        "angry": 0,
        "disgust": 0,
        "fear": 0,
        "happy": 0,
        "neutral": 0,
        "sad": 0,
        "surprise": 0,
    }

    # Higher smoothing = slower but smoother chart transitions
    smoothing_factor = 0.85

    # Main update loop - runs while video is playing
    if ctx.video_processor:
        while ctx.state.playing:
            # Thread-safe copy of current emotion probabilities
            with ctx.video_processor.lock:
                current_probs = ctx.video_processor.emotion_probs.copy()

            # Apply display-side smoothing for smooth chart animations
            for emotion in display_probs:
                display_probs[emotion] = (
                    smoothing_factor * display_probs[emotion]
                    + (1 - smoothing_factor) * current_probs[emotion]
                )

            # Create DataFrame for Streamlit bar chart
            df = pd.DataFrame(
                {
                    "Emotion": list(display_probs.keys()),
                    "Probability": list(display_probs.values()),
                }
            )

            # Update the chart in place
            chart_placeholder.bar_chart(df.set_index("Emotion"))

            # Short sleep for responsive updates (20 FPS)
            time.sleep(0.05)
