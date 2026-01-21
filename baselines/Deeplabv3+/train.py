import os
import datetime
import argparse
from PIL import Image
import torch
import segmentation_models_pytorch as smp #type: ignore
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms #type: ignore
import torch.nn as nn
from torch.optim.adamw import AdamW
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.segmentation import DiceScore as Dice
import surface_distance as surfdist  # type: ignore
import numpy as np


# ======================
# Dataset Class
# ======================
class BinaryChestXRaySegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform
        self.mask_transform = mask_transform

        image_folder = os.path.join(root_dir, "images")
        mask_folder = os.path.join(root_dir, "masks")

        for img_name in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_name)
            mask_path = os.path.join(mask_folder, img_name)
            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


# ======================
# Transforms
# ======================
image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze().long())
])


# ======================
# Loss Function
# ======================
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return bce_loss + dice_loss


# ======================
# NSD Metric
# ======================
def compute_nsd(pred, target):
    pred_np = pred.cpu().numpy().astype(bool)
    target_np = target.cpu().numpy().astype(bool)
    if pred_np.sum() == 0 and target_np.sum() == 0:
        return 1.0
    distances = surfdist.compute_surface_distances(target_np, pred_np, spacing_mm=(1.0, 1.0))
    nsd = surfdist.compute_average_surface_distance(distances)[1]
    return float(nsd)


# ======================
# Lightning Model
# ======================
class SegmentationModel(pl.LightningModule):
    def __init__(self, lr=1e-4, run_dir="./runs"):
        super(SegmentationModel, self).__init__()
        self.save_hyperparameters()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            classes=1,
            in_channels=3,
            activation=None
        )
        self.criterion = BCEDiceLoss()
        self.lr = lr
        self.dice_metric = Dice(num_classes=1, average='macro')

    def forward(self, x):
        return self.model(x)

    def step(self, batch, stage):
        images, gt_masks = batch
        gt_masks = gt_masks.unsqueeze(1).float()
        preds = self(images)
        loss = self.criterion(preds, gt_masks)

        pred_binary = (torch.sigmoid(preds) > 0.5).int()
        dice_score = self.dice_metric(pred_binary, gt_masks.int())

        nsd_scores = []
        for i in range(len(pred_binary)):
            nsd = compute_nsd(pred_binary[i, 0], gt_masks[i, 0])
            nsd_scores.append(nsd)
        nsd_mean = np.mean(nsd_scores)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_dice", dice_score, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_nsd", nsd_mean, prog_bar=True, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.999))
        return optimizer


# ======================
# ENTRY POINT
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    # ======================
    # Data Loaders
    # ======================
    train_dataset = BinaryChestXRaySegmentationDataset(
        os.path.join(args.dataset_root, "train"),
        transform=image_transform,
        mask_transform=mask_transform
    )
    val_dataset = BinaryChestXRaySegmentationDataset(
        os.path.join(args.dataset_root, "val"),
        transform=image_transform,
        mask_transform=mask_transform
    )
    test_dataset = BinaryChestXRaySegmentationDataset(
        os.path.join(args.dataset_root, "test"),
        transform=image_transform,
        mask_transform=mask_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    # ======================
    # Run Directory
    # ======================
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("./runs", f"run_{timestamp}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "final_models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tensorboard_logs"), exist_ok=True)

    # ======================
    # Callbacks & Logger
    # ======================
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=True)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="best_model",
        save_top_k=1,
        mode="min"
    )
    logger = TensorBoardLogger(save_dir=os.path.join(run_dir, "tensorboard_logs"), name="")

    # ======================
    # Training
    # ======================
    model = SegmentationModel(lr=args.lr, run_dir=run_dir)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        strategy="auto",
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), os.path.join(run_dir, "final_models", "deeplabv3plus_final.pt"))
    trainer.test(model, test_loader)

    print(f"🎉 Training complete. Results saved in {run_dir}")


if __name__ == "__main__":
    main()