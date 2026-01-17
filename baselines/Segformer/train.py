"""
Complete PyTorch Lightning Training Script for Binary Semantic Segmentation using SegFormer
Supports multi-GPU training, mixed precision, early stopping, and comprehensive logging.

Author: Claude
Date: 2025-10-30
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import random

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

# Hugging Face Transformers
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Torchmetrics
from torchmetrics import JaccardIndex
from torchmetrics.segmentation import DiceScore


# ============================================================================
# Set Random Seeds for Reproducibility
# ============================================================================
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    # Make CUDNN deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Custom Dataset Class
# ============================================================================
class SegmentationDataset(Dataset):
    """
    Custom Dataset for binary semantic segmentation.
    
    Args:
        image_dir: Path to directory containing images
        mask_dir: Path to directory containing masks
        processor: SegformerImageProcessor for preprocessing
        image_size: Target size for images and masks (height, width)
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        processor: SegformerImageProcessor,
        image_size: Tuple[int, int] = (512, 512)
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.processor = processor
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = sorted(list(self.image_dir.glob("*.png")))
        
        # Verify that corresponding masks exist
        for img_path in self.image_files:
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found for image: {img_path.name}")
        
        print(f"Loaded {len(self.image_files)} image-mask pairs from {self.image_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> dict:
        # Load image and mask
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale
        
        # Resize to target size
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)
        
        # Convert mask to numpy array and normalize to 0/1
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0).astype(np.float32)  # Ensure binary: 0 or 1
        
        # Process image using SegformerImageProcessor
        # This handles normalization and tensor conversion
        encoded_inputs = self.processor(image, return_tensors="pt")
        
        # Remove batch dimension added by processor
        pixel_values = encoded_inputs["pixel_values"].squeeze(0)
        
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return {
            "pixel_values": pixel_values,
            "labels": mask_tensor,
            "image_path": str(img_path)
        }


# ============================================================================
# Loss Functions
# ============================================================================
class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Predicted logits/probabilities [B, 1, H, W]
            targets: Ground truth binary masks [B, 1, H, W]
        
        Returns:
            Dice loss value
        """
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1.0 - dice_score


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss for binary segmentation.
    
    Args:
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
    """
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output logits [B, 1, H, W]
            targets: Ground truth binary masks [B, 1, H, W]
        
        Returns:
            Combined loss value
        """
        bce = self.bce_loss(logits, targets)
        
        # Apply sigmoid for Dice loss
        probs = torch.sigmoid(logits)
        dice = self.dice_loss(probs, targets)
        
        return self.bce_weight * bce + self.dice_weight * dice


# ============================================================================
# PyTorch Lightning DataModule
# ============================================================================
class SegmentationDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for handling train/val/test dataloaders.
    
    Args:
        data_root: Root directory containing train/val/test folders
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        image_size: Target size for images (height, width)
        model_name: Pretrained SegFormer model name for processor
    """
    
    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (512, 512),
        model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640"
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # Initialize SegFormer image processor
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage (fit/test)."""
        
        if stage == "fit" or stage is None:
            # Training dataset
            self.train_dataset = SegmentationDataset(
                image_dir=self.data_root / "train" / "images_rgb",
                mask_dir=self.data_root / "train" / "masks_binary",
                processor=self.processor,
                image_size=self.image_size
            )
            
            # Validation dataset
            self.val_dataset = SegmentationDataset(
                image_dir=self.data_root / "val" / "images_rgb",
                mask_dir=self.data_root / "val" / "masks_binary",
                processor=self.processor,
                image_size=self.image_size
            )
        
        if stage == "test" or stage is None:
            # Test dataset
            self.test_dataset = SegmentationDataset(
                image_dir=self.data_root / "test" / "images_rgb",
                mask_dir=self.data_root / "test" / "masks_binary",
                processor=self.processor,
                image_size=self.image_size
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


# ============================================================================
# PyTorch Lightning Module
# ============================================================================
class SegFormerLightningModule(pl.LightningModule):
    """
    Lightning Module for SegFormer binary semantic segmentation.
    
    Args:
        model_name: Pretrained SegFormer model name
        learning_rate: Learning rate for optimizer
        num_labels: Number of output classes (1 for binary)
        image_size: Input image size (height, width)
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        learning_rate: float = 1e-4,
        num_labels: int = 1,
        image_size: Tuple[int, int] = (512, 512)
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.image_size = image_size
        
        # Load pretrained SegFormer model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True  # Allow different num_labels
        )
        
        # Loss function
        self.criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        
        # Metrics for binary segmentation
        self.train_dice = DiceScore(num_classes=1, average='macro')
        self.val_dice = DiceScore(num_classes=1, average='macro')
        self.test_dice = DiceScore(num_classes=1, average='macro')

        self.train_iou = JaccardIndex(task='binary', num_classes=2)
        self.val_iou = JaccardIndex(task='binary', num_classes=2)
        self.test_iou = JaccardIndex(task='binary', num_classes=2)
        
        # Store predictions for visualization
        self.test_predictions = []
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        outputs = self.model(pixel_values=pixel_values)
        
        # Upsample logits to match input size
        logits = outputs.logits
        upsampled_logits = F.interpolate(
            logits,
            size=self.image_size,
            mode="bilinear",
            align_corners=False
        )
        
        return upsampled_logits
    
    def shared_step(self, batch: dict, stage: str):
        """Shared step for train/val/test."""
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        
        # Forward pass
        logits = self(pixel_values)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Apply sigmoid and threshold for metrics
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Convert to integer labels for metrics
        preds_int = preds.long().squeeze(1)
        labels_int = labels.long().squeeze(1)
        
        # Update metrics based on stage
        if stage == "train":
            self.train_dice(preds, labels)
            self.train_iou(preds_int, labels_int)
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_dice", self.train_dice, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_iou", self.train_iou, on_step=False, on_epoch=True)
        elif stage == "val":
            self.val_dice(preds, labels)
            self.val_iou(preds_int, labels_int)
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_iou", self.val_iou, on_step=False, on_epoch=True)
        elif stage == "test":
            self.test_dice(preds, labels)
            self.test_iou(preds_int, labels_int)
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True)
            self.log(f"{stage}_dice", self.test_dice, on_step=False, on_epoch=True)
            self.log(f"{stage}_iou", self.test_iou, on_step=False, on_epoch=True)
            
            # Store predictions for visualization
            self.test_predictions.append({
                "predictions": preds.cpu(),
                "labels": labels.cpu(),
                "image_paths": batch["image_path"]
            })
        
        return loss
    
    def training_step(self, batch: dict, batch_idx: int):
        """Training step."""
        return self.shared_step(batch, "train")
    
    def validation_step(self, batch: dict, batch_idx: int):
        """Validation step."""
        return self.shared_step(batch, "val")
    
    def test_step(self, batch: dict, batch_idx: int):
        """Test step."""
        return self.shared_step(batch, "test")
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


# ============================================================================
# Visualization Functions
# ============================================================================
def save_predictions(
    model: SegFormerLightningModule,
    dataloader: DataLoader,
    output_dir: Path,
    device: torch.device,
    num_samples: int = 10
):
    """
    Save predicted masks and create visualizations.
    
    Args:
        model: Trained model
        dataloader: DataLoader for predictions
        output_dir: Directory to save predictions
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir.parent / "visualizations" / output_dir.name
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    model.to(device)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]
            image_paths = batch["image_path"]
            
            # Get predictions
            logits = model(pixel_values)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Move to CPU for saving
            preds = preds.cpu().numpy()
            labels = labels.numpy()
            
            # Save each prediction
            for i in range(len(image_paths)):
                img_path = Path(image_paths[i])
                img_name = img_path.stem
                
                # Save prediction mask
                pred_mask = (preds[i, 0] * 255).astype(np.uint8)
                pred_img = Image.fromarray(pred_mask)
                pred_img.save(output_dir / f"{img_name}_pred.png")
                
                # Create visualization for first N samples
                if sample_count < num_samples:
                    visualize_prediction(
                        image_path=img_path,
                        pred_mask=preds[i, 0],
                        gt_mask=labels[i, 0],
                        save_path=vis_dir / f"{img_name}_comparison.png"
                    )
                    sample_count += 1
            
            if sample_count >= num_samples:
                break
    
    print(f"Predictions saved to: {output_dir}")
    print(f"Visualizations saved to: {vis_dir}")


def visualize_prediction(
    image_path: Path,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    save_path: Path
):
    """
    Create a visualization comparing original image, ground truth, and prediction.
    
    Args:
        image_path: Path to original image
        pred_mask: Predicted mask [H, W]
        gt_mask: Ground truth mask [H, W]
        save_path: Path to save visualization
    """
    # Load original image
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Ground truth mask
    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    # Predicted mask
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")
    
    # Overlay: Original + Prediction
    overlay = image.copy()
    # Create red overlay for predicted mask
    mask_overlay = np.zeros_like(image)
    mask_overlay[pred_mask > 0.5] = [255, 0, 0]  # Red for predictions
    overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
    
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (Prediction in Red)")
    axes[3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Import cv2 for overlay
import cv2


# ============================================================================
# Main Training Function
# ============================================================================
def main():
    """Main training pipeline."""
    
    # ========================================================================
    # Argument Parsing
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="Train SegFormer for Binary Semantic Segmentation"
    )
    
    # Data arguments
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing train/val/test folders")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints and predictions")
    
    # Model arguments
    parser.add_argument("--model_name", type=str,
                        default="nvidia/segformer-b5-finetuned-ade-640-640",
                        help="Pretrained SegFormer model name")
    parser.add_argument("--num_labels", type=int, default=1,
                        help="Number of output classes (1 for binary)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Input image size (square)")
    
    # Training configuration
    parser.add_argument("--precision", type=str, default="16-mixed",
                        choices=["32", "16-mixed", "bf16-mixed"],
                        help="Training precision")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Checkpoint arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    
    # Evaluation arguments
    parser.add_argument("--num_vis_samples", type=int, default=10,
                        help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    # ========================================================================
    # Setup
    # ========================================================================
    print("=" * 80)
    print("SegFormer Binary Semantic Segmentation Training")
    print("=" * 80)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    pred_dir = output_dir / "predictions"
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # ========================================================================
    # Data Module
    # ========================================================================
    print("\n" + "=" * 80)
    print("Initializing Data Module")
    print("=" * 80)
    
    image_size = (args.image_size, args.image_size)
    
    data_module = SegmentationDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
        model_name=args.model_name
    )
    
    # ========================================================================
    # Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("Initializing Model")
    print("=" * 80)
    
    model = SegFormerLightningModule(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        num_labels=args.num_labels,
        image_size=image_size
    )
    
    # Print model summary
    print(f"\nModel: {args.model_name}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ========================================================================
    # Callbacks
    # ========================================================================
    print("\n" + "=" * 80)
    print("Setting up Callbacks")
    print("=" * 80)
    
    # Model checkpoint callback - save best model based on validation Dice
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best_model",
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_dice",
        patience=args.early_stopping_patience,
        mode="max",
        verbose=True
    )
    
    # Progress bar
    progress_bar = TQDMProgressBar(refresh_rate=10)
    
    callbacks = [checkpoint_callback, early_stop_callback, progress_bar]
    
    # ========================================================================
    # Logger
    # ========================================================================
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="segformer_training",
        default_hp_metric=False
    )
    
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"View logs with: tensorboard --logdir {log_dir}")
    
    # ========================================================================
    # Trainer
    # ========================================================================
    print("\n" + "=" * 80)
    print("Initializing Trainer")
    print("=" * 80)
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Image Size: {image_size}")
    print(f"  Precision: {args.precision}")
    print(f"  Gradient Accumulation: {args.accumulate_grad_batches}")
    
    # ========================================================================
    # Training
    # ========================================================================
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best validation Dice: {checkpoint_callback.best_model_score:.4f}")
    
    # ========================================================================
    # Testing
    # ========================================================================
    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)
    
    # Load best model for testing
    best_model = SegFormerLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    
    # Test
    trainer.test(best_model, datamodule=data_module)
    
    # ========================================================================
    # Save Predictions
    # ========================================================================
    print("\n" + "=" * 80)
    print("Generating Predictions")
    print("=" * 80)
    
    # Setup data module for test
    data_module.setup("test")
    
    # Save validation predictions
    print("\nGenerating validation predictions...")
    data_module.setup("fit")
    save_predictions(
        model=best_model,
        dataloader=data_module.val_dataloader(),
        output_dir=pred_dir / "val",
        device=device,
        num_samples=args.num_vis_samples
    )
    
    # Save test predictions
    print("\nGenerating test predictions...")
    data_module.setup("test")
    save_predictions(
        model=best_model,
        dataloader=data_module.test_dataloader(),
        output_dir=pred_dir / "test",
        device=device,
        num_samples=args.num_vis_samples
    )
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Pipeline Complete!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Predictions: {pred_dir}")
    print(f"  Logs: {log_dir}")
    print(f"\nTo view training logs, run:")
    print(f"  tensorboard --logdir {log_dir}")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()