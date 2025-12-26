import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model import EmotionCNN
from utils.processing import ImageProcessing
import torch.optim as optim 
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch
from collections import defaultdict

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.parent

    # Build paths relative to script location
    train_path = script_dir / "images" / "train"
    val_path = script_dir / "images" / "validation"
    # ===== SETUP =====
    # Define parameters 
    processing = ImageProcessing(
        train_path=train_path,
        val_path=val_path
    )
    
    # Device setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Hyperparameters
    learning_rate = 0.001
    epochs = 50
    
    # Load data
    train_loader, val_loader = processing.create_loaders()
    num_classes = processing.get_num_classes()
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {processing.class_names}")
    print(f"Training samples: {len(processing.train_dataset)}")
    print(f"Validation samples: {len(processing.val_dataset)}")
    
    # Initialize model
    model = EmotionCNN(num_classes=num_classes)
    model.to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    
    # Tracking
    metrics = defaultdict(list)
    best_val_acc = 0.0
    best_epoch = 0
    
    
    # ===== TRAINING LOOP =====
    print("\n" + "="*60)
    print(f"Starting Training for {epochs} Epochs")
    print("="*60)
    
    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print("-" * 60)
        
        # ===== TRAINING PHASE =====
        model.train()  # Set model to training mode
        
        train_running_loss = 0.0
        train_correct = 0 
        train_total = 0
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc="Training")
        
        for images, labels in train_pbar:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)  # Raw logits
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            # torch.max returns (values, indices)
            # We use indices as predicted classes
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Accumulate loss
            train_running_loss += loss.item()

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Calculate training averages
        train_avg_loss = train_running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        
        # ===== VALIDATION PHASE =====
        model.eval()  # Set model to evaluation mode (disables dropout, batchnorm)
        
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # No gradient computation during validation
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            
            for images, labels in val_pbar:
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass only (no backward)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Accumulate loss
                val_running_loss += loss.item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        # Calculate validation averages
        val_avg_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        
        # ===== SAVE METRICS =====
        metrics['train_loss'].append(train_avg_loss)
        metrics['train_acc'].append(train_accuracy)
        metrics['val_loss'].append(val_avg_loss)
        metrics['val_acc'].append(val_accuracy)
        
        
        # ===== PRINT EPOCH SUMMARY =====
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_avg_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss:   {val_avg_loss:.4f} | Val Acc:   {val_accuracy:.2f}%")
        print(f"{'='*60}")
        
        
        # ===== SAVE BEST MODEL =====
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ New best model saved! (Val Acc: {val_accuracy:.2f}%)")
    
    
    # ===== TRAINING COMPLETE =====
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"{'='*60}")
    
    
    # # ===== PLOT TRAINING HISTORY =====
    # import matplotlib.pyplot as plt
    
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # # Loss plot
    # ax1.plot(metrics['train_loss'], label='Train Loss', marker='o')
    # ax1.plot(metrics['val_loss'], label='Val Loss', marker='s')
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss')
    # ax1.set_title('Training and Validation Loss')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)
    
    # # Accuracy plot
    # ax2.plot(metrics['train_acc'], label='Train Acc', marker='o')
    # ax2.plot(metrics['val_acc'], label='Val Acc', marker='s')
    # ax2.set_xlabel('Epoch')
    # ax2.set_ylabel('Accuracy (%)')
    # ax2.set_title('Training and Validation Accuracy')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.savefig('training_history.png', dpi=150)
    # print("\n✓ Training history plot saved as 'training_history.png'")
    # plt.show()
    
    
   