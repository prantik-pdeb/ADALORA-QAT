### How to use:
### python -u "/home/prantik/Srimanth/test_folder/adalora-qat/inference.py" \
# --image_path /home/prantik/Srimanth/data/Final_dataset_split_Resized/test/images/C19RD_COVID-29.png \
# --checkpoint_path "/home/prantik/Srimanth/model_weights/ISBI_final_quantization_weights_15th(FINAL)/\
# checkpoints_stage2_int8_full/best_model_stage2_int8.pth" \
# --bbox 0 0 511 511 --save_mask --visualize\
# --output_mask_path /home/prantik/Srimanth/test_folder/adalora-qat/inf_res.png \
# --save_overlay /home/prantik/Srimanth/test_folder/adalora-qat/overlay


import argparse
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from sam_Ada_LoRA_QAT_two_stage import AdaLoRA_Sam, apply_full_quantization_to_sam


def load_model(checkpoint_path, bit_width, skip_qkv, max_rank, target_rank, alpha, device):

    sam_base = SamModel.from_pretrained("facebook/sam-vit-base")

    apply_full_quantization_to_sam(
        sam_base,
        bit_width=bit_width,
        skip_qkv=skip_qkv
    )

    model = AdaLoRA_Sam(
        sam_base,
        max_rank=max_rank,
        target_rank=target_rank,
        alpha=alpha
    )

    SUPPORTED_DEVICES = {"cpu", "cuda", "mps"}

    if device not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported device '{device}'. Supported: {SUPPORTED_DEVICES}")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this system.")

    device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model


def predict_mask(model, processor, image_path, bbox, device):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        image,
        input_boxes=[[bbox]],
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    if 'original_sizes' in inputs:
        inputs.pop('original_sizes')
    if 'reshaped_input_sizes' in inputs: # Often comes with original_sizes
        inputs.pop('reshaped_input_sizes')

    with torch.no_grad():
        outputs = model(**inputs)

    mask = torch.sigmoid(outputs.pred_masks)
    mask = (mask > 0.5).cpu().numpy().astype(np.uint8)

    return image, mask


def visualize_mask(image, mask, bbox, save_path):
    # 1. Ensure mask is 2D and same size as image (512, 512)
    if not isinstance(image, np.ndarray):
        image_np = np.array(image)
    else:
        image_np = image
    mask_np = mask.squeeze()
    if mask_np.shape != image_np.shape[:2]:
        mask_np = cv2.resize(mask_np, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 2. Create a figure that matches the image aspect ratio perfectly
    fig = plt.figure(frameon=False)
    fig.set_size_inches(image_np.shape[1]/100, image_np.shape[0]/100) # Maintain aspect ratio
    ax = plt.Axes(fig, [0., 0., 1., 1.]) # Remove all margins/padding
    ax.set_axis_off()
    fig.add_axes(ax)

    # 3. Overlay the image and the mask
    ax.imshow(image)
    
    # Create a colored version of the mask (e.g., Green)
    
    color_mask = np.zeros((*mask_np.shape, 4)) # RGBA
    color_mask[mask_np > 0.5] = [0, 1, 0, 0.5] # Green with 50% transparency
    ax.imshow(color_mask)

    # 4. Save without white borders
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_mask(mask, output_path):

    mask = mask[0][0] * 255
    mask_img = Image.fromarray(mask.squeeze().astype(np.uint8))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_img.save(output_path)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--bbox", type=int, nargs=4, required=True,
                        help="Bounding box as x1 y1 x2 y2")

    parser.add_argument("--bit_width", type=int, default=8)
    parser.add_argument("--skip_qkv", default=True)

    parser.add_argument("--max_rank", type=int, default=48)
    parser.add_argument("--target_rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--visualize", action="store_true",
                        help="Display mask overlay")

    parser.add_argument("--save_mask", action="store_true",
                        help="Save predicted mask as PNG")

    parser.add_argument("--output_mask_path", type=str,
                        default="outputs/pred_mask.png")

    parser.add_argument("--save_overlay", type=str,
                        default=None,
                        help="Path to save overlay visualization")

    return parser.parse_args()


def main():

    args = parse_args()

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    model = load_model(
        args.checkpoint_path,
        args.bit_width,
        args.skip_qkv,
        args.max_rank,
        args.target_rank,
        args.alpha,
        args.device
    )

    image, mask = predict_mask(
        model,
        processor,
        args.image_path,
        args.bbox,
        args.device
    )

    if args.visualize:
        visualize_mask(image, mask, args.bbox, args.save_overlay)

    if args.save_mask:
        save_mask(mask, args.output_mask_path)
        print(f"Mask saved to {args.output_mask_path}")


if __name__ == "__main__":
    main()