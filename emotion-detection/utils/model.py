"""
CNN Model Architecture for Facial Emotion Recognition.

This module defines the neural network architecture used for classifying
facial expressions into 7 emotion categories:
- angry, disgust, fear, happy, neutral, sad, surprise

Architecture Overview:
- 3 convolutional blocks for feature extraction
- 2 fully connected layers for classification
- Input: 224x224 grayscale images
- Output: 7-class probability distribution

Classes:
    ConvBlock: Reusable convolutional building block
    EmotionCNN: Main emotion classification network
"""

from torch import nn


class ConvBlock(nn.Module):
    """
    Reusable convolutional block for feature extraction.
    
    Each block contains:
    - Conv2d: Extracts features
    - BatchNorm2d: Normalizes activations for stable training
    - ReLU: Non-linear activation
    - MaxPool2d: Reduces spatial dimensions by half
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output feature maps
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Store for reference (optional, not strictly needed)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Define the sequential block
        self.block = nn.Sequential(
            # Convolutional layer
            # padding=1 keeps spatial size same (H, W unchanged)
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            
            # Batch normalization - normalizes conv output
            # Speeds up training and adds regularization
            nn.BatchNorm2d(num_features=out_channels),
            
            # ReLU activation - introduces non-linearity
            # inplace=True saves memory
            nn.ReLU(inplace=True),
            
            # Max pooling - reduces spatial dimensions by 2
            # (H, W) -> (H/2, W/2)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        """Forward pass through the block"""
        return self.block(x)


class EmotionCNN(nn.Module):
    """
    CNN for facial emotion recognition.
    
    Architecture:
    - Input: (batch, 1, 224, 224) - grayscale images
    - 3 convolutional blocks with progressive channel increase
    - 2 fully connected layers for classification
    - Output: (batch, 7) - probabilities for 7 emotions
    
    Args:
        num_classes: Number of emotion classes (default: 7)
    """
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction blocks
        # Each block: Conv -> BatchNorm -> ReLU -> MaxPool
        
        # Block 1: 1 -> 32 channels, 224x224 -> 112x112
        self.conv1 = ConvBlock(in_channels=1, out_channels=32)
        
        # Block 2: 32 -> 64 channels, 112x112 -> 56x56
        self.conv2 = ConvBlock(in_channels=32, out_channels=64)
        
        # Block 3: 64 -> 128 channels, 56x56 -> 28x28
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        
        # Combine all feature extraction blocks
        self.features = nn.Sequential(
            self.conv1,  # Output: (batch, 32, 112, 112)
            self.conv2,  # Output: (batch, 64, 56, 56)
            self.conv3   # Output: (batch, 128, 28, 28)
        )
        
        # Classification head
        # Input: 128 * 28 * 28 = 100,352 features
        self.classifier = nn.Sequential(
            # Flatten 3D feature maps to 1D vector
            # (batch, 128, 28, 28) -> (batch, 100352)
            nn.Flatten(),
            
            # First fully connected layer
            # Reduces from 100,352 to 128 features
            nn.Linear(128 * 28 * 28, 128),
            
            # Normalize FC layer output
            nn.BatchNorm1d(128),
            
            # ReLU activation
            nn.ReLU(inplace=True),
            
            # Dropout for regularization (prevent overfitting)
            nn.Dropout(0.5),  # Drop 50% of neurons during training
            
            # Final classification layer
            # 128 -> num_classes (7 emotions)
            # NO Softmax here! CrossEntropyLoss applies it internally
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 1, 224, 224)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Extract features through conv blocks
        x = self.features(x)      # (batch, 128, 28, 28)
        
        # Classify
        x = self.classifier(x)    # (batch, 7)
        
        return x  # Return raw logits (no softmax!)
