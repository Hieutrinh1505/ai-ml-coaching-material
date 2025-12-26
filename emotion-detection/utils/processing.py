"""
Image Processing and Data Loading Utilities for Emotion Detection.

This module handles all data preprocessing and loading operations:
- Image transformations (with/without data augmentation)
- Dataset loading using PyTorch's ImageFolder
- DataLoader creation for training and validation

Key Design Decisions:
- Training data uses augmentation (flips, jitter) for better generalization
- Validation data uses no augmentation for consistent evaluation
- Images are converted to grayscale and normalized to [-1, 1]

Classes:
    ImageProcessing: Main class for dataset management and preprocessing
"""

from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# ============================================================
# TRANSFORM DEFINITIONS
# ============================================================
# IMPORTANT: Different transforms for train vs validation

# Training transforms - WITH data augmentation
transform_train = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.Resize((224, 224)),
    # Data augmentation (only for training!)
    v2.RandomHorizontalFlip(p=0.5),
    # v2.RandomRotation(15),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])

# Validation transforms - NO augmentation
transform_val = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.Resize((224, 224)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5], std=[0.5])
])


class ImageProcessing:
    """
    Handles dataset loading and dataloader creation.
    Uses different transforms for training and validation.
    """
    def __init__(self, train_path, val_path, 
                 transform_train=transform_train, 
                 transform_val=transform_val):
        self.train_path = train_path
        self.val_path = val_path
        self.transform_train = transform_train  # Augmentation
        self.transform_val = transform_val      # No augmentation
        
        # Will be populated after load_datasets()
        self.train_dataset = None 
        self.val_dataset = None 
        self.num_classes = None 
        self.class_names = None
        
    def load_datasets(self):
        """Load train and validation datasets with appropriate transforms"""
        # Train with augmentation
        self.train_dataset = ImageFolder(
            self.train_path,
            transform=self.transform_train  # Uses augmentation
        )
        
        # Validation without augmentation
        self.val_dataset = ImageFolder(
            self.val_path,
            transform=self.transform_val  # No augmentation
        )
        
        # Store metadata
        self.num_classes = len(self.train_dataset.classes)
        self.class_names = self.train_dataset.classes
        
        print(f"✓ Loaded {len(self.train_dataset)} training images")
        print(f"✓ Loaded {len(self.val_dataset)} validation images")
        print(f"✓ Classes ({self.num_classes}): {self.class_names}")
    
    def create_loaders(self, batch_size=32, num_workers=4):
        """Create train and validation dataloaders"""
        # Ensure datasets are loaded
        if self.train_dataset is None:
            self.load_datasets()
        
        # Training loader - shuffle=True
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Randomize order each epoch
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Validation loader - shuffle=False
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep consistent order
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader

    def get_num_classes(self):
        return self.num_classes