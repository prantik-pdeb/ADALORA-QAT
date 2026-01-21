import os
import time
import argparse
from typing import Callable, Optional
from PIL import Image
import numpy as np
import torch
from torch import Tensor, device, no_grad
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.nn.functional import sigmoid, interpolate
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance  # Google DeepMind official


# ================== DATASET ==================
class BinarySegmentationDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 image_transform: Optional[Callable] = None,
                 mask_transform: Optional[Callable] = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        image_files = sorted(os.listdir(images_dir))
        mask_files = sorted(os.listdir(masks_dir))
        self.common_files = sorted(set(image_files).intersection(mask_files))
        if not self.common_files:
            raise ValueError(f"No matching image-mask pairs found in {images_dir} and {masks_dir}.")

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, idx):
        fname = self.common_files[idx]
        img_path = os.path.join(self.images_dir, fname)
        mask_path = os.path.join(self.masks_dir, fname)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = mask.squeeze(0)
        mask = (mask > 0).float()
        return image, mask, fname


# ================== TRANSFORMS ==================
image_transform = Compose([
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

mask_transform = Compose([
    Resize((512, 512)),
    ToTensor(),
])


# ================== METRICS ==================
def compute_dice(pred: Tensor, target: Tensor) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-7)


def compute_iou(pred: Tensor, target: Tensor) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-7)


def compute_nsd(pred: np.ndarray, target: np.ndarray, tolerance: float = 1.0) -> float:
    if not pred.any() and not target.any():
        return 1.0
    if not pred.any() or not target.any():
        return 0.0

    surface_distances = compute_surface_distances(
        target.astype(bool),
        pred.astype(bool),
        spacing_mm=(1.0, 1.0)
    )
    nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance_mm=tolerance)
    return float(nsd)


# ================== LOAD MODEL ==================
def load_model(model_path: str, device: torch.device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    ckpt = torch.load(model_path, map_location=device)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    state_dict = {k.replace("model.", "").replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded Segformer-B5 model from checkpoint: {model_path}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    return model


# ================== EVALUATION ==================
def evaluate_model(model, dataloader, device, save_dir: str, tolerance_mm: float):
    model.eval()
    dice_scores, iou_scores, nsd_scores = [], [], []

    total_images = 0
    start_time = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with no_grad():
        for images, masks, fnames in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            outputs = model(images)
            preds = outputs.logits
            preds = interpolate(preds, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            preds = sigmoid(preds)
            preds_bin = (preds > 0.5).float()

            for i in range(preds_bin.shape[0]):
                pred_tensor = preds_bin[i, 0].cpu()
                gt_tensor = masks[i, 0].cpu()

                dice_scores.append(compute_dice(pred_tensor, gt_tensor).item())
                iou_scores.append(compute_iou(pred_tensor, gt_tensor).item())
                nsd_scores.append(
                    compute_nsd(
                        pred_tensor.numpy().astype(bool),
                        gt_tensor.numpy().astype(bool),
                        tolerance=tolerance_mm
                    )
                )

                pred_img = (pred_tensor.numpy() * 255).astype(np.uint8)
                Image.fromarray(pred_img).save(os.path.join(save_dir, fnames[i]))

            total_images += len(fnames)

    total_time = time.time() - start_time
    avg_time_per_image = total_time / total_images if total_images > 0 else 0.0

    max_mem_used_mb = None
    if device.type == "cuda":
        max_mem_used_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return dice_scores, iou_scores, nsd_scores, total_time, avg_time_per_image, max_mem_used_mb


# ================== UTILITIES ==================
def fmt(mean, std):
    return f"{mean:.4f} ± {std:.3f}"


# ================== ENTRY POINT ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--val_images_dir", required=True)
    parser.add_argument("--val_masks_dir", required=True)
    parser.add_argument("--save_preds_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--tolerance_mm", type=float, default=2.0)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_preds_dir, exist_ok=True)

    val_dataset = BinarySegmentationDataset(
        args.val_images_dir,
        args.val_masks_dir,
        image_transform=image_transform,
        mask_transform=mask_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    model = load_model(args.model_path, device)

    dice_scores, iou_scores, nsd_scores, total_time, avg_time_per_image, max_mem_used_mb = evaluate_model(
        model,
        val_loader,
        device,
        args.save_preds_dir,
        args.tolerance_mm
    )

    print("\n==================== RESULTS ====================")
    print(f"Dice Score Coefficient (DSC): {fmt(np.mean(dice_scores), np.std(dice_scores))}")
    print(f"Intersection over Union (IoU): {fmt(np.mean(iou_scores), np.std(iou_scores))}")
    print(f"Normalized Surface Dice (NSD): {fmt(np.mean(nsd_scores), np.std(nsd_scores))}")
    print("-------------------------------------------------")
    print(f"Total images evaluated: {len(val_dataset)}")
    print(f"Total inference time: {total_time:.3f} sec")
    print(f"Average inference time per image: {avg_time_per_image:.4f} sec")
    if max_mem_used_mb is not None:
        print(f"Peak GPU memory used: {max_mem_used_mb:.2f} MB")
    print(f"Predicted masks saved in: {args.save_preds_dir}")
    print("=================================================")


if __name__ == "__main__":
    main()
