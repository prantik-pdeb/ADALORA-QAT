import os
from typing import Optional, Callable
from PIL import Image
import segmentation_models_pytorch as smp  # type: ignore
from torchvision.transforms import Resize, ToTensor, Normalize, Compose  # type: ignore
from torchmetrics.segmentation import DiceScore, MeanIoU
from torch import Tensor, device, no_grad, save, load
from torch.nn.functional import sigmoid
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss, DataParallel
from torch.optim import AdamW
from torch.cuda import is_available, device_count
from surface_distance import compute_average_surface_distance, compute_surface_distances  # type: ignore
from numpy import mean
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ================== CONSTANTS ==================
CHECKPOINT_DIR = "/home/prantik/Srimanth/DeeplabV3Plus/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DATASET_ROOT = "/home/prantik/datasets_Srimanth/Final_dataset_split_Resized"
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
IMG_SIZE = (512, 512)

# ================== MODEL ==================
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=1,
    in_channels=3,
    activation=None,
)

device_t = device("cuda" if is_available() else "cpu")
if device_count() > 1:
    model = DataParallel(model).to(device_t)
else:
    model.to(device_t)

# ================== DATASET ==================
class BinaryChestXRaySegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        root_dir: str,
        data_transform: Callable[[Image.Image], Tensor],
        mask_transform_fn: Callable[[Image.Image], Tensor],
    ):
        self.root_dir = root_dir
        self.image_transform = data_transform
        self.mask_transform = mask_transform_fn
        self.image_paths, self.mask_paths = [], []

        image_folder = os.path.join(root_dir, "images")
        mask_folder = os.path.join(root_dir, "masks")
        if not os.path.exists(image_folder) or not os.path.exists(mask_folder):
            raise FileNotFoundError("Missing 'images' or 'masks' folder in dataset.")

        image_files = set(os.listdir(image_folder))
        mask_files = set(os.listdir(mask_folder))
        common_files = image_files.intersection(mask_files)

        for file_name in sorted(common_files):
            self.image_paths.append(os.path.join(image_folder, file_name))
            self.mask_paths.append(os.path.join(mask_folder, file_name))

        unmatched_images = image_files - common_files
        unmatched_masks = mask_files - common_files
        if unmatched_images or unmatched_masks:
            print(f"⚠️ Warning: Skipped {len(unmatched_images)} unmatched images "
                  f"and {len(unmatched_masks)} unmatched masks.")

    def __len__(self) -> int:
        assert len(self.image_paths) == len(self.mask_paths)
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_path, mask_path = self.image_paths[idx], self.mask_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask).squeeze(0).long()
        mask = (mask > 0).float()

        return image, mask

# ================== TRANSFORMS ==================
image_transform = Compose([
    Resize(IMG_SIZE),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
mask_transform = Compose([Resize(IMG_SIZE), ToTensor()])

# ================== DATALOADERS ==================
train_dataset = BinaryChestXRaySegmentationDataset(
    os.path.join(DATASET_ROOT, "train"), image_transform, mask_transform)
val_dataset = BinaryChestXRaySegmentationDataset(
    os.path.join(DATASET_ROOT, "val"), image_transform, mask_transform)
test_dataset = BinaryChestXRaySegmentationDataset(
    os.path.join(DATASET_ROOT, "test"), image_transform, mask_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)

# ================== LOSS & OPTIMIZER ==================
bce_loss = BCEWithLogitsLoss()
dice_loss = smp.losses.DiceLoss(mode="binary", eps=1e-7)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# ================== METRICS ==================
dice_metric = DiceScore(num_classes=2, average="macro").to(device=device_t)
iou_metric = MeanIoU(num_classes=2).to(device=device_t)

def compute_nsd(pred: Tensor, target: Tensor) -> float:
    if pred.dim() == 4:
        pred = pred.squeeze(1)
        target = target.squeeze(1)
    nsd_scores = []
    for i in range(pred.shape[0]):
        pred_np = pred[i].cpu().detach().numpy().astype(bool)
        target_np = target[i].cpu().detach().numpy().astype(bool)
        if pred_np.sum() == 0 and target_np.sum() == 0:
            nsd_scores.append(1.0)
            continue
        distances = compute_surface_distances(target_np, pred_np, spacing_mm=(1.0, 1.0))
        nsd_scores.append(compute_average_surface_distance(distances)[1])
    return mean(nsd_scores)

# ================== TENSORBOARD ==================
writer = SummaryWriter(log_dir="/home/prantik/Srimanth/DeeplabV3Plus/runs")

# ================== RESUME TRAINING ==================
last_checkpoint = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
start_epoch = 0
global_step = 0
if os.path.exists(last_checkpoint):
    checkpoint = load(last_checkpoint, map_location=device_t)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint.get("global_step", 0)
    print(f"Resuming training from epoch {start_epoch}")

# ================== TRAINING LOOP ==================
for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    train_loss, train_nsd = 0.0, 0.0
    dice_metric.reset()
    iou_metric.reset()

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")
    log_interval = max(1, len(train_loader) // 20)

    for step, (images, gt_masks) in enumerate(loop):
        images, gt_masks = images.to(device_t), gt_masks.to(device_t).unsqueeze(1)
        optimizer.zero_grad()

        pred_masks = model(images)
        pred_sigmoid = sigmoid(pred_masks)

        loss = bce_loss(pred_sigmoid, gt_masks.float()) + dice_loss(pred_sigmoid, gt_masks.float())
        loss.backward()
        optimizer.step()

        nsd_score = compute_nsd(pred_sigmoid, gt_masks.float())
        dice_metric.update(pred_sigmoid.int(), gt_masks.int())
        iou_metric.update(pred_sigmoid.int(), gt_masks.int())

        train_loss += loss.item()
        train_nsd += nsd_score
        global_step += 1

        # --- Log approximately 20 times per epoch ---
        if (step + 1) % log_interval == 0 or (step + 1) == len(train_loader):
            avg_loss = train_loss / (step + 1)
            avg_dice = dice_metric.compute().item()
            avg_iou = iou_metric.compute().item()
            avg_nsd = train_nsd / (step + 1)

            writer.add_scalar("Train/Loss", avg_loss, global_step)
            writer.add_scalar("Train/Dice", avg_dice, global_step)
            writer.add_scalar("Train/IoU", avg_iou, global_step)
            writer.add_scalar("Train/NSD", avg_nsd, global_step)

            loop.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "dice": f"{avg_dice:.4f}",
                "iou": f"{avg_iou:.4f}",
                "nsd": f"{avg_nsd:.4f}"
            })

    # --- Log example predictions once per epoch ---
    sample_images, sample_masks = next(iter(val_loader))
    sample_images, sample_masks = sample_images.to(device_t), sample_masks.to(device_t).unsqueeze(1)
    with no_grad():
        sample_preds = sigmoid(model(sample_images)) > 0.5
    writer.add_images("Validation/Images", sample_images, epoch)
    writer.add_images("Validation/GroundTruth", sample_masks, epoch)
    writer.add_images("Validation/Predictions", sample_preds.float(), epoch)

    # ================== VALIDATION ==================
    model.eval()
    val_loss, val_nsd = 0.0, 0.0
    dice_metric.reset()
    iou_metric.reset()

    with no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation")
        for images, gt_masks in val_loop:
            images, gt_masks = images.to(device_t), gt_masks.to(device_t).unsqueeze(1)
            pred_masks = sigmoid(model(images))
            loss = bce_loss(pred_masks, gt_masks.float()) + dice_loss(pred_masks, gt_masks.float())
            nsd_score = compute_nsd(pred_masks, gt_masks.float())

            dice_metric.update(pred_masks.int(), gt_masks.int())
            iou_metric.update(pred_masks.int(), gt_masks.int())

            val_loss += loss.item()
            val_nsd += nsd_score

    val_loss /= len(val_loader)
    val_nsd /= len(val_loader)
    val_dice = dice_metric.compute().item()
    val_iou = iou_metric.compute().item()

    writer.add_scalar("Val/Loss", val_loss, epoch)
    writer.add_scalar("Val/Dice", val_dice, epoch)
    writer.add_scalar("Val/IoU", val_iou, epoch)
    writer.add_scalar("Val/NSD", val_nsd, epoch)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Val Loss: {val_loss:.4f} Val Dice: {val_dice:.4f} Val IoU: {val_iou:.4f} Val NSD: {val_nsd:.4f}")

    # ================== CHECKPOINTING ==================
    checkpoint_data = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    save(checkpoint_data, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt"))
    save(checkpoint_data, last_checkpoint)  # latest for resume

print("✅ Training Complete.")
writer.close()
