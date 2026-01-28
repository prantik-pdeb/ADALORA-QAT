import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
from PIL import Image
from tqdm import tqdm
import time
import json
from datetime import datetime
import gc
import warnings
import random
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.cuda.amp import autocast, GradScaler

# MONAI metrics
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.losses import DiceLoss

# Brevitas for quantization
try:
    import brevitas.nn as qnn
    from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
    BREVITAS_AVAILABLE = True
    print("✓ Brevitas available - Full QAT enabled")
except ImportError:
    BREVITAS_AVAILABLE = False
    print("  Brevitas not installed. QAT features will be unavailable.")
    print("   Install with: pip install brevitas")

warnings.filterwarnings('ignore')

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# Memory efficient attention
try:
    import torch.nn.functional as F
    if hasattr(F, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        print("✓ Memory-efficient attention enabled")
except:
    pass

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================================
# QUANTIZATION UTILITIES
# ============================================================================

def quantize_linear_layer(layer: nn.Linear, bit_width: int = 8):
    """Convert nn.Linear to QuantLinear (Brevitas)"""
    if not BREVITAS_AVAILABLE:
        return layer
    
    quant_layer = qnn.QuantLinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        weight_bit_width=bit_width,
        weight_quant=Int8WeightPerTensorFloat,
        bias_quant=None
    )
    
    quant_layer.weight.data = layer.weight.data.clone()
    if layer.bias is not None:
        quant_layer.bias.data = layer.bias.data.clone()
    
    return quant_layer


def quantize_module_recursive(module, bit_width: int = 8, skip_names: set = None):
    """Recursively replace Linear with quantized versions"""
    if not BREVITAS_AVAILABLE:
        return
    
    if skip_names is None:
        skip_names = set()
    
    for name, child in list(module.named_children()):
        if name in skip_names:
            continue
        
        if isinstance(child, nn.Linear):
            setattr(module, name, quantize_linear_layer(child, bit_width))
        else:
            quantize_module_recursive(child, bit_width, skip_names)


def apply_full_quantization_to_sam(sam_model, bit_width=8, skip_qkv=True):
    """
    Apply FULL quantization to SAM model
    
    Args:
        sam_model: SamModel instance
        bit_width: Quantization bit width (8 for INT8)
        skip_qkv: If True, skip qkv layers (they will have AdaLoRA)
    
    Returns:
        Number of quantized parameters
    """
    if not BREVITAS_AVAILABLE:
        print("  Brevitas not available, skipping quantization")
        return 0
    
    print("\n" + "="*80)
    print("APPLYING FULL MODEL QUANTIZATION")
    print("="*80)
    
    quantized_count = 0
    
    # 1. Quantize Vision Encoder (except qkv layers if skip_qkv=True)
    print(f"\n Quantizing Vision Encoder to {bit_width}-bit...")
    for layer_idx, layer in enumerate(sam_model.vision_encoder.layers):
        layer_quant_count = 0
        
        for name, module in layer.named_modules():
            # Skip qkv if requested 
            if skip_qkv and 'qkv' in name:
                continue
            
            if isinstance(module, nn.Linear):
                # Get parent module and child name
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[-1]
                
                # Navigate to parent
                parent = layer
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                # Replace with quantized version
                quant_module = quantize_linear_layer(module, bit_width)
                setattr(parent, child_name, quant_module)
                layer_quant_count += 1
                quantized_count += 1
        
        status = "QUANTIZED" if layer_quant_count > 0 else "SKIPPED"
        print(f"   Layer {layer_idx:2d}: {status} ({layer_quant_count} linear layers)")
    
    # Quantize patch embedding
    if hasattr(sam_model.vision_encoder, 'patch_embed'):
        embed_count = 0
        for name, module in sam_model.vision_encoder.patch_embed.named_modules():
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[-1]
                
                parent = sam_model.vision_encoder.patch_embed
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                quant_module = quantize_linear_layer(module, bit_width)
                setattr(parent, child_name, quant_module)
                embed_count += 1
                quantized_count += 1
        
        if embed_count > 0:
            print(f"   ✓ Patch Embedding: QUANTIZED ({embed_count} layers)")
    
    # 2. Quantize Mask Decoder (ALL layers)
    print(f"\n Quantizing Mask Decoder to {bit_width}-bit...")
    decoder_count = 0
    for name, module in sam_model.mask_decoder.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            
            parent = sam_model.mask_decoder
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            quant_module = quantize_linear_layer(module, bit_width)
            setattr(parent, child_name, quant_module)
            decoder_count += 1
            quantized_count += 1
    
    print(f"   ✓ All decoder layers QUANTIZED ({decoder_count} linear layers)")
    
    # 3. Quantize Prompt Encoder (ALL layers)
    print(f"\n Quantizing Prompt Encoder to {bit_width}-bit...")
    prompt_count = 0
    for name, module in sam_model.prompt_encoder.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            
            parent = sam_model.prompt_encoder
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            quant_module = quantize_linear_layer(module, bit_width)
            setattr(parent, child_name, quant_module)
            prompt_count += 1
            quantized_count += 1
    
    if prompt_count > 0:
        print(f"   ✓ Prompt encoder QUANTIZED ({prompt_count} linear layers)")
    
    # Calculate quantization coverage
    total_linear = sum(1 for m in sam_model.modules() if isinstance(m, (nn.Linear, qnn.QuantLinear)))
    quant_pct = (quantized_count / total_linear) * 100 if total_linear > 0 else 0
    
    print("\n" + "-"*80)
    print("QUANTIZATION SUMMARY")
    print("-"*80)
    print(f"Total linear layers:     {total_linear}")
    print(f"Quantized to INT8:       {quantized_count}")
    print(f"Quantization coverage:   {quant_pct:.1f}%")
    print("="*80 + "\n")
    
    return quantized_count


def enable_qat_mode(model):
    """Enable QAT mode for Brevitas quantized layers"""
    if not BREVITAS_AVAILABLE:
        return
    
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, (qnn.QuantLinear, qnn.QuantConv2d)):
            if hasattr(module, 'weight_quant'):
                module.weight_quant.training = True
            if hasattr(module, 'act_quant'):
                module.act_quant.training = True
            count += 1
    
    if count > 0:
        print(f"✓ QAT mode enabled for {count} quantized layers")


# ============================================================================
# ADAPTIVE LoRA IMPLEMENTATION 
# ============================================================================

class AdaLoRA_QKV(nn.Module):
    """Adaptive LoRA for SAM attention QKV projection with SVD parameterization"""
    
    def __init__(self, qkv: nn.Module, max_rank: int = 32, target_rank: int = 16, 
                 alpha: float = 16.0, epsilon: float = 1e-6):
        super(AdaLoRA_QKV, self).__init__()
        
        self.qkv = qkv
        self.dim = qkv.in_features if hasattr(qkv, 'in_features') else qkv.weight.shape[1]
        self.max_rank = max_rank
        self.target_rank = target_rank
        self.current_rank = max_rank
        self.alpha = alpha
        self.epsilon = epsilon
        
        # SVD parameterization
        self.P_q = nn.Parameter(torch.empty(self.dim, max_rank))
        self.Lambda_q = nn.Parameter(torch.ones(max_rank) * 0.1)
        self.Q_q = nn.Parameter(torch.empty(max_rank, self.dim))
        
        self.P_k = nn.Parameter(torch.empty(self.dim, max_rank))
        self.Lambda_k = nn.Parameter(torch.ones(max_rank) * 0.1)
        self.Q_k = nn.Parameter(torch.empty(max_rank, self.dim))
        
        self.P_v = nn.Parameter(torch.empty(self.dim, max_rank))
        self.Lambda_v = nn.Parameter(torch.ones(max_rank) * 0.1)
        self.Q_v = nn.Parameter(torch.empty(max_rank, self.dim))
        
        self._init_svd_params()
        
        self.register_buffer('importance_q', torch.zeros(max_rank))
        self.register_buffer('importance_k', torch.zeros(max_rank))
        self.register_buffer('importance_v', torch.zeros(max_rank))
        self.register_buffer('importance_ema', torch.tensor(0.9))
        
        self.register_buffer('mask_q', torch.ones(max_rank, dtype=torch.bool))
        self.register_buffer('mask_k', torch.ones(max_rank, dtype=torch.bool))
        self.register_buffer('mask_v', torch.ones(max_rank, dtype=torch.bool))
        
        self.register_buffer('qat_mode', torch.tensor(False))
        
        for param in self.qkv.parameters():
            param.requires_grad = False
    
    def _init_svd_params(self):
        nn.init.orthogonal_(self.P_q)
        nn.init.orthogonal_(self.Q_q)
        nn.init.orthogonal_(self.P_k)
        nn.init.orthogonal_(self.Q_k)
        nn.init.orthogonal_(self.P_v)
        nn.init.orthogonal_(self.Q_v)
    
    @property
    def scaling(self):
        return self.alpha / self.target_rank
    
    def enable_qat_mode(self):
        self.qat_mode = torch.tensor(True)
        self.importance_ema = torch.tensor(1.0)
    
    def freeze_singular_values(self):
        self.Lambda_q.requires_grad = False
        self.Lambda_k.requires_grad = False
        self.Lambda_v.requires_grad = False

    def get_orthogonal_regularization(self):
        """
        Compute orthogonal regularization loss
        Fixed: No in-place operations to avoid gradient computation errors
        """
        device = self.P_q.device
        I = torch.eye(self.max_rank, device=device)
    
        # Collect all regularization terms 
        reg_terms = [
            torch.norm(self.P_q.T @ self.P_q - I, p='fro'),
            torch.norm(self.P_k.T @ self.P_k - I, p='fro'),
            torch.norm(self.P_v.T @ self.P_v - I, p='fro'),
            torch.norm(self.Q_q @ self.Q_q.T - I, p='fro'),
            torch.norm(self.Q_k @ self.Q_k.T - I, p='fro'),
            torch.norm(self.Q_v @ self.Q_v.T - I, p='fro')
        ]
    
        # Stack and sum 
        reg_loss = torch.stack(reg_terms).sum()
    
        return reg_loss
    
    def compute_importance(self):
        if self.qat_mode:
            return
        
        if self.Lambda_q.grad is not None:
            raw_importance_q = torch.abs(self.Lambda_q.data) * torch.abs(self.Lambda_q.grad.data)
            self.importance_q = self.importance_ema * self.importance_q + (1 - self.importance_ema) * raw_importance_q
        
        if self.Lambda_k.grad is not None:
            raw_importance_k = torch.abs(self.Lambda_k.data) * torch.abs(self.Lambda_k.grad.data)
            self.importance_k = self.importance_ema * self.importance_k + (1 - self.importance_ema) * raw_importance_k
        
        if self.Lambda_v.grad is not None:
            raw_importance_v = torch.abs(self.Lambda_v.data) * torch.abs(self.Lambda_v.grad.data)
            self.importance_v = self.importance_ema * self.importance_v + (1 - self.importance_ema) * raw_importance_v
    
    def stabilize_singular_values(self, clip_value=0.02):
        if self.Lambda_q.grad is not None:
            self.Lambda_q.grad.clamp_(-clip_value, clip_value)
        if self.Lambda_k.grad is not None:
            self.Lambda_k.grad.clamp_(-clip_value, clip_value)
        if self.Lambda_v.grad is not None:
            self.Lambda_v.grad.clamp_(-clip_value, clip_value)
    
    def prune_to_target_rank(self):
        if self.current_rank <= self.target_rank:
            return
        
        combined_importance = (self.importance_q + self.importance_k + self.importance_v) / 3.0
        _, top_indices = torch.topk(combined_importance, self.target_rank)
        
        new_mask = torch.zeros(self.max_rank, dtype=torch.bool, device=self.mask_q.device)
        new_mask[top_indices] = True
        
        self.mask_q = new_mask
        self.mask_k = new_mask
        self.mask_v = new_mask
        self.current_rank = self.target_rank
    
    def get_rank_distribution(self):
        return {
            'current_rank': self.current_rank,
            'qat_mode': bool(self.qat_mode.item()),
        }
    
    def forward(self, x):
        original_shape = x.shape
        if x.ndim == 4:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
            needs_reshape_4d = True
        elif x.ndim == 3:
            needs_reshape_4d = False
        elif x.ndim == 2:
            needs_reshape_4d = False
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        qkv = self.qkv(x)
        
        masked_Lambda_q = self.Lambda_q * self.mask_q.float()
        masked_Lambda_k = self.Lambda_k * self.mask_k.float()
        masked_Lambda_v = self.Lambda_v * self.mask_v.float()
        
        delta_W_q = self.P_q @ torch.diag(masked_Lambda_q) @ self.Q_q
        delta_W_k = self.P_k @ torch.diag(masked_Lambda_k) @ self.Q_k
        delta_W_v = self.P_v @ torch.diag(masked_Lambda_v) @ self.Q_v
        
        delta_q = (x @ delta_W_q.T) * self.scaling
        delta_k = (x @ delta_W_k.T) * self.scaling
        delta_v = (x @ delta_W_v.T) * self.scaling
        
        if x.ndim == 2:
            BN = x.shape[0]
            qkv = qkv.reshape(BN, 3, self.dim)
            qkv[:, 0, :] += delta_q
            qkv[:, 1, :] += delta_k
            qkv[:, 2, :] += delta_v
            qkv = qkv.reshape(BN, 3 * self.dim)
            
        elif x.ndim == 3:
            B, N = x.shape[0], x.shape[1]
            qkv = qkv.reshape(B, N, 3, self.dim)
            qkv[:, :, 0, :] += delta_q
            qkv[:, :, 1, :] += delta_k
            qkv[:, :, 2, :] += delta_v
            qkv = qkv.reshape(B, N, 3 * self.dim)
            
            if needs_reshape_4d:
                H = int(np.sqrt(N))
                W = N // H
                qkv = qkv.reshape(B, H, W, 3 * self.dim)
        
        return qkv

# ============================================================================
# ADAPTIVE LoRA SAM MODEL 
# ============================================================================

class AdaLoRA_Sam(nn.Module):
    """SAM model with Adaptive LoRA adaptation"""
    
    def __init__(self, sam_model: SamModel, max_rank: int = 32, target_rank: int = 16, 
                 alpha: float = 16.0, lora_layers: list = None):
        super(AdaLoRA_Sam, self).__init__()
        
        self.sam = sam_model
        self.max_rank = max_rank
        self.target_rank = target_rank
        self.alpha = alpha
        
        # Freeze all SAM parameters
        for param in sam_model.parameters():
            param.requires_grad = False
        
        # Inject AdaLoRA into attention layers
        num_blocks = len(sam_model.vision_encoder.layers)
        if lora_layers is None:
            self.lora_layers = list(range(num_blocks))
        else:
            self.lora_layers = [l for l in lora_layers if 0 <= l < num_blocks]
        
        print(f"\nAdaptive LoRA Configuration:")
        print(f"  Max Rank: {max_rank}")
        print(f"  Target Rank: {target_rank}")
        print(f"  Alpha: {alpha}")
        print(f"  Target layers: {len(self.lora_layers)}/{num_blocks}")
        
        self.adalora_modules = nn.ModuleList()
        for layer_idx in self.lora_layers:
            layer = sam_model.vision_encoder.layers[layer_idx]
            
            original_qkv = layer.attn.qkv
            adalora_qkv = AdaLoRA_QKV(
                original_qkv,
                max_rank=max_rank,
                target_rank=target_rank,
                alpha=alpha
            )
            layer.attn.qkv = adalora_qkv
            self.adalora_modules.append(adalora_qkv)
        
        total_params = sum(p.numel() for p in sam_model.parameters())
        trainable_params = sum(p.numel() for p in sam_model.parameters() if p.requires_grad)
        
        print(f"\nParameter Statistics (Encoder LoRA only):")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  Trainable %: {100*trainable_params/total_params:.4f}%")
        
        self.pruning_history = []
    
    def forward(self, pixel_values, input_boxes, multimask_output=False):
        return self.sam(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=multimask_output
        )
    
    def enable_qat_mode(self):
        print("\n" + "="*80)
        print("Enabling QAT Mode for AdaLoRA")
        print("="*80)
        for module in self.adalora_modules:
            module.enable_qat_mode()
        print("="*80 + "\n")
    
    def freeze_singular_values(self):
        print("\nFreezing singular values for QAT phase...")
        for module in self.adalora_modules:
            module.freeze_singular_values()
    
    def compute_importance(self):
        for module in self.adalora_modules:
            module.compute_importance()
    
    def stabilize_singular_values(self, clip_value=0.02):
        for module in self.adalora_modules:
            module.stabilize_singular_values(clip_value)
    
    def get_orthogonal_regularization(self):
        total_reg = 0.0
        for module in self.adalora_modules:
            total_reg += module.get_orthogonal_regularization()
        return total_reg
    
    def prune_to_target_rank(self):
        print(f"\n{'='*80}")
        print("ADAPTIVE RANK PRUNING")
        print(f"{'='*80}")
        
        for idx, module in enumerate(self.adalora_modules):
            layer_idx = self.lora_layers[idx]
            module.prune_to_target_rank()
            print(f"Layer {layer_idx}: rank → {module.current_rank}")
        
        print(f"{'='*80}\n")
    
    def get_rank_statistics(self):
        stats = []
        for idx, module in enumerate(self.adalora_modules):
            layer_idx = self.lora_layers[idx]
            layer_stats = module.get_rank_distribution()
            layer_stats['layer_idx'] = layer_idx
            stats.append(layer_stats)
        return stats
    
    def save_adalora_weights(self, path: str):
        adalora_state = {}
        
        for i, module in enumerate(self.adalora_modules):
            layer_idx = self.lora_layers[i]
            prefix = f'layer_{layer_idx}'
            
            adalora_state[f'{prefix}_P_q'] = module.P_q.cpu()
            adalora_state[f'{prefix}_Lambda_q'] = module.Lambda_q.cpu()
            adalora_state[f'{prefix}_Q_q'] = module.Q_q.cpu()
            adalora_state[f'{prefix}_P_k'] = module.P_k.cpu()
            adalora_state[f'{prefix}_Lambda_k'] = module.Lambda_k.cpu()
            adalora_state[f'{prefix}_Q_k'] = module.Q_k.cpu()
            adalora_state[f'{prefix}_P_v'] = module.P_v.cpu()
            adalora_state[f'{prefix}_Lambda_v'] = module.Lambda_v.cpu()
            adalora_state[f'{prefix}_Q_v'] = module.Q_v.cpu()
            adalora_state[f'{prefix}_importance_q'] = module.importance_q.cpu()
            adalora_state[f'{prefix}_importance_k'] = module.importance_k.cpu()
            adalora_state[f'{prefix}_importance_v'] = module.importance_v.cpu()
            adalora_state[f'{prefix}_mask_q'] = module.mask_q.cpu()
            adalora_state[f'{prefix}_mask_k'] = module.mask_k.cpu()
            adalora_state[f'{prefix}_mask_v'] = module.mask_v.cpu()
            adalora_state[f'{prefix}_current_rank'] = module.current_rank
        
        checkpoint = {
            'adalora_state_dict': adalora_state,
            'max_rank': self.max_rank,
            'target_rank': self.target_rank,
            'alpha': self.alpha,
            'lora_layers': self.lora_layers,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, path)
        print(f"✓ AdaLoRA weights saved: {path}")
    
    def load_adalora_weights(self, path: str, strict: bool = True):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        adalora_state = checkpoint['adalora_state_dict']
        
        if strict:
            assert checkpoint['max_rank'] == self.max_rank
            assert checkpoint['target_rank'] == self.target_rank
        
        for i, module in enumerate(self.adalora_modules):
            layer_idx = self.lora_layers[i]
            prefix = f'layer_{layer_idx}'
            device = module.P_q.device
            
            module.P_q.data = adalora_state[f'{prefix}_P_q'].to(device)
            module.Lambda_q.data = adalora_state[f'{prefix}_Lambda_q'].to(device)
            module.Q_q.data = adalora_state[f'{prefix}_Q_q'].to(device)
            module.P_k.data = adalora_state[f'{prefix}_P_k'].to(device)
            module.Lambda_k.data = adalora_state[f'{prefix}_Lambda_k'].to(device)
            module.Q_k.data = adalora_state[f'{prefix}_Q_k'].to(device)
            module.P_v.data = adalora_state[f'{prefix}_P_v'].to(device)
            module.Lambda_v.data = adalora_state[f'{prefix}_Lambda_v'].to(device)
            module.Q_v.data = adalora_state[f'{prefix}_Q_v'].to(device)
            module.importance_q = adalora_state[f'{prefix}_importance_q'].to(device)
            module.importance_k = adalora_state[f'{prefix}_importance_k'].to(device)
            module.importance_v = adalora_state[f'{prefix}_importance_v'].to(device)
            module.mask_q = adalora_state[f'{prefix}_mask_q'].to(device)
            module.mask_k = adalora_state[f'{prefix}_mask_k'].to(device)
            module.mask_v = adalora_state[f'{prefix}_mask_v'].to(device)
            module.current_rank = adalora_state[f'{prefix}_current_rank']
        
        print(f"✓ AdaLoRA weights loaded from: {path}")

# ============================================================================
# HYBRID TRAINING UTILITIES
# ============================================================================

def enable_decoder_training(model, config):
    """Enable full fine-tuning for decoder and prompt encoder"""
    print("\n" + "="*80)
    print(" ENABLING HYBRID TRAINING MODE ")
    print("="*80)
    
    if config['stage1'].get('train_decoder', False):
        for param in model.sam.mask_decoder.parameters():
            param.requires_grad = True
        print("\n✓ Mask Decoder UNFROZEN")
    
    if config['stage1'].get('train_prompt_encoder', False):
        for param in model.sam.prompt_encoder.parameters():
            param.requires_grad = True
        print("✓ Prompt Encoder UNFROZEN")
    
    total_params = sum(p.numel() for p in model.sam.parameters())
    trainable_params = sum(p.numel() for p in model.sam.parameters() if p.requires_grad)
    
    print(f"\n Total Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print("="*80 + "\n")
    
    return model


def create_differential_optimizer(model, config):
    """Create optimizer with differential learning rates"""
    is_hybrid = config['stage1'].get('train_decoder', False) or config['stage1'].get('train_prompt_encoder', False)
    
    if not is_hybrid:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config['stage1']['learning_rate'],
            weight_decay=config['stage1']['weight_decay']
        )
    
    print("\n" + "="*80)
    print("CREATING DIFFERENTIAL LEARNING RATE OPTIMIZER")
    print("="*80)
    
    lora_params = []
    decoder_params = []
    prompt_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(key in name for key in ['Lambda_', 'P_q', 'P_k', 'P_v', 'Q_q', 'Q_k', 'Q_v']):
            lora_params.append(param)
        elif 'mask_decoder' in name:
            decoder_params.append(param)
        elif 'prompt_encoder' in name:
            prompt_params.append(param)
    
    param_groups = []
    
    if len(lora_params) > 0:
        param_groups.append({
            'params': lora_params,
            'lr': config['stage1']['learning_rate'],
            'weight_decay': config['stage1']['weight_decay'],
            'name': 'lora_encoder'
        })
        print(f"\n✓ LoRA Encoder: {len(lora_params)} tensors, LR={config['stage1']['learning_rate']}")
    
    if len(decoder_params) > 0:
        lr_decoder = config['stage1'].get('lr_decoder', config['stage1']['learning_rate'] * 0.4)
        param_groups.append({
            'params': decoder_params,
            'lr': lr_decoder,
            'weight_decay': config['stage1']['weight_decay'],
            'name': 'decoder'
        })
        print(f"✓ Decoder: {len(decoder_params)} tensors, LR={lr_decoder}")
    
    if len(prompt_params) > 0:
        lr_prompt = config['stage1'].get('lr_prompt', config['stage1']['learning_rate'] * 0.4)
        param_groups.append({
            'params': prompt_params,
            'lr': lr_prompt,
            'weight_decay': config['stage1']['weight_decay'],
            'name': 'prompt_encoder'
        })
        print(f"✓ Prompt Encoder: {len(prompt_params)} tensors, LR={lr_prompt}")
    
    optimizer = torch.optim.AdamW(param_groups)
    print("="*80 + "\n")
    
    return optimizer


# ============================================================================
# METRICS AND LOSS
# ============================================================================

class StandardSegmentationMetrics:
    """Standard medical image segmentation metrics"""
    def __init__(self, include_background=False):
        self.dice_metric = DiceMetric(include_background=include_background, reduction="mean_batch", get_not_nans=False)
        self.iou_metric = MeanIoU(include_background=include_background, reduction="mean_batch", get_not_nans=False)
        self.hd_metric = HausdorffDistanceMetric(include_background=include_background, percentile=95.0, reduction="mean_batch", get_not_nans=False)
        self.surface_metric = SurfaceDistanceMetric(include_background=include_background, symmetric=True, reduction="mean_batch", get_not_nans=False)
    
    def compute_batch_metrics(self, pred, target):
        pred_tensor = torch.from_numpy(pred).unsqueeze(1).float()
        target_tensor = torch.from_numpy(target).unsqueeze(1).float()
        dice_scores = self.dice_metric(pred_tensor, target_tensor)
        iou_scores = self.iou_metric(pred_tensor, target_tensor)
        dice_list = dice_scores.squeeze().cpu().numpy().flatten().tolist()
        iou_list = iou_scores.squeeze().cpu().numpy().flatten().tolist()
        if not isinstance(dice_list, list): dice_list = [dice_list]
        if not isinstance(iou_list, list): iou_list = [iou_list]
        
        hd_list = []
        assd_list = []
        for i in range(pred_tensor.shape[0]):
            try:
                hd_score = self.hd_metric(pred_tensor[i:i+1], target_tensor[i:i+1])
                hd_val = float(hd_score.item()) if not torch.isnan(hd_score) else 0.0
            except:
                hd_val = 0.0
            hd_list.append(hd_val)
            try:
                surface_dist = self.surface_metric(pred_tensor[i:i+1], target_tensor[i:i+1])
                assd_val = float(surface_dist.item()) if not torch.isnan(surface_dist) else 0.0
            except:
                assd_val = 0.0
            assd_list.append(assd_val)
        
        return {'dice': dice_list, 'iou': iou_list, 'hd95': hd_list, 'assd': assd_list}
    
    def compute_nsd(self, pred, target, tolerance=2.0, spacing=1.0):
        pred_bool = pred.astype(bool)
        target_bool = target.astype(bool)
        if not np.any(pred_bool) and not np.any(target_bool): return 1.0
        if not np.any(pred_bool) or not np.any(target_bool): return 0.0
        try:
            pred_surface = pred_bool ^ binary_erosion(pred_bool)
            target_surface = target_bool ^ binary_erosion(target_bool)
            if not np.any(pred_surface) or not np.any(target_surface):
                return 1.0 if np.array_equal(pred_bool, target_bool) else 0.0
            dt_pred = distance_transform_edt(~pred_bool, sampling=spacing)
            dt_target = distance_transform_edt(~target_bool, sampling=spacing)
            dist_pred_to_target = dt_target[pred_surface]
            dist_target_to_pred = dt_pred[target_surface]
            all_surface_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
            nsd = np.mean(all_surface_distances <= tolerance)
            return float(nsd)
        except:
            return 0.0


class MedSAMLoss(nn.Module):
    """MedSAM Loss: BCE + Dice"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)


class EarlyStopping:
    """Early stopping"""
    def __init__(self, patience=5, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False
        if self.mode == 'max':
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def perturb_bbox(bbox, max_perturbation=20, img_shape=(512, 512)):
    x_min, y_min, x_max, y_max = bbox
    perturb = np.random.randint(0, max_perturbation + 1, size=4)
    x_min_new = max(0, x_min - perturb[0])
    y_min_new = max(0, y_min - perturb[1])
    x_max_new = min(img_shape[1], x_max + perturb[2])
    y_max_new = min(img_shape[0], y_max + perturb[3])
    return [x_min_new, y_min_new, x_max_new, y_max_new], perturb


class MedicalSegmentationDataset(Dataset):
    """Medical segmentation dataset"""
    def __init__(self, image_paths, mask_paths, processor, target_size=512):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.target_size = target_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        image = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        mask_np = (np.array(mask) > 0).astype(np.uint8)
        
        coords = np.where(mask_np > 0)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            original_bbox = [x_min, y_min, x_max, y_max]
            perturbed_bbox, _ = perturb_bbox(original_bbox, max_perturbation=20, img_shape=mask_np.shape)
            prompt_box = perturbed_bbox
        else:
            prompt_box = [0, 0, 1, 1]
        
        inputs = self.processor(image, input_boxes=[[prompt_box]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np, dtype=torch.float32)
        return inputs


# ============================================================================
# TRAINING FUNCTIONS 
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, 
                metrics_computer, accumulation_steps=1, ortho_reg_weight=0.01):
    """Train for one epoch"""
    model.train()
    scaler = GradScaler()
    
    total_loss = 0
    total_main_loss = 0
    total_ortho_loss = 0
    all_metrics = {'dice': [], 'iou': []}
    
    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            ground_truth_mask = batch["ground_truth_mask"].to(device)
            
            with autocast():
                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                predicted_masks = outputs.pred_masks.squeeze(1).squeeze(1)
                ground_truth_resized = nn.functional.interpolate(
                    ground_truth_mask.unsqueeze(1), 
                    size=predicted_masks.shape[-2:], 
                    mode='nearest'
                ).squeeze(1)
                
                main_loss = criterion(predicted_masks, ground_truth_resized)
                
                if isinstance(model, nn.DataParallel):
                    ortho_loss = model.module.get_orthogonal_regularization()
                else:
                    ortho_loss = model.get_orthogonal_regularization()
                
                loss = main_loss + ortho_reg_weight * ortho_loss
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if isinstance(model, nn.DataParallel):
                model.module.stabilize_singular_values(clip_value=0.02)
                model.module.compute_importance()
            else:
                model.stabilize_singular_values(clip_value=0.02)
                model.compute_importance()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            total_loss += loss.item() * accumulation_steps
            total_main_loss += main_loss.item()
            total_ortho_loss += ortho_loss.item()
            
            if batch_idx % 20 == 0:
                with torch.no_grad():
                    pred_binary = (torch.sigmoid(predicted_masks) > 0.5).float()
                    gt_binary = ground_truth_resized.float()
                    
                    intersection = (pred_binary * gt_binary).sum(dim=(1, 2))
                    union = pred_binary.sum(dim=(1, 2)) + gt_binary.sum(dim=(1, 2))
                    dice_batch = (2. * intersection / (union + 1e-8)).mean().item()
                    all_metrics['dice'].append(dice_batch)
                    
                    iou_batch = (intersection / (union - intersection + 1e-8)).mean().item()
                    all_metrics['iou'].append(iou_batch)
                    
                    del pred_binary, gt_binary, intersection, union
            
            current_dice = np.mean(all_metrics["dice"]) if len(all_metrics["dice"]) > 0 else 0.0
            pbar.set_postfix({
                'loss': f'{main_loss.item():.4f}', 
                'dice': f'{current_dice:.4f}'
            })
            
            del outputs, predicted_masks, ground_truth_resized, main_loss, ortho_loss, loss
            del pixel_values, input_boxes, ground_truth_mask
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n  OOM at batch {batch_idx}. Clearing cache...")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                continue
            else:
                raise
    
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    torch.cuda.empty_cache()
    
    return {
        'loss': total_loss / len(dataloader),
        'main_loss': total_main_loss / len(dataloader),
        'ortho_loss': total_ortho_loss / len(dataloader),
        'dice': np.mean(all_metrics['dice']) if len(all_metrics['dice']) > 0 else 0.0,
        'iou': np.mean(all_metrics['iou']) if len(all_metrics['iou']) > 0 else 0.0,
    }


def validate(model, dataloader, criterion, device, metrics_computer):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_metrics = {'dice': [], 'iou': [], 'nsd': [], 'hd95': [], 'assd': []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            try:
                pixel_values = batch["pixel_values"].to(device)
                input_boxes = batch["input_boxes"].to(device)
                ground_truth_mask = batch["ground_truth_mask"].to(device)
                
                outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                predicted_masks = outputs.pred_masks.squeeze(1).squeeze(1)
                ground_truth_resized = nn.functional.interpolate(
                    ground_truth_mask.unsqueeze(1), size=predicted_masks.shape[-2:], mode='nearest'
                ).squeeze(1)
                
                loss = criterion(predicted_masks, ground_truth_resized)
                total_loss += loss.item()
                
                pred_binary = (torch.sigmoid(predicted_masks) > 0.5).cpu().numpy()
                gt_binary = ground_truth_resized.cpu().numpy()
                
                batch_metrics = metrics_computer.compute_batch_metrics(pred_binary, gt_binary)
                all_metrics['dice'].extend(batch_metrics['dice'])
                all_metrics['iou'].extend(batch_metrics['iou'])
                all_metrics['hd95'].extend(batch_metrics['hd95'])
                all_metrics['assd'].extend(batch_metrics['assd'])
                
                for i in range(len(pred_binary)):
                    nsd = metrics_computer.compute_nsd(pred_binary[i], gt_binary[i], tolerance=2.0)
                    all_metrics['nsd'].append(nsd)
                
                del outputs, predicted_masks, ground_truth_resized, pred_binary, gt_binary
                del pixel_values, input_boxes, ground_truth_mask, loss
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise
    
    torch.cuda.empty_cache()
    
    return {
        'loss': total_loss / len(dataloader),
        'dice': np.mean(all_metrics['dice']),
        'dice_std': np.std(all_metrics['dice']),
        'iou': np.mean(all_metrics['iou']),
        'iou_std': np.std(all_metrics['iou']),
        'nsd': np.mean(all_metrics['nsd']),
        'nsd_std': np.std(all_metrics['nsd']),
        'hd95': np.mean(all_metrics['hd95']),
        'hd95_std': np.std(all_metrics['hd95']),
        'assd': np.mean(all_metrics['assd']),
        'assd_std': np.std(all_metrics['assd'])
    }


# ============================================================================
# TWO-STAGE TRAINING WITH FULL QUANTIZATION
# ============================================================================

def two_stage_training_full_quant(config, splits, processor, device):
    """
    Two-Stage Training with FULL Quantization
    
    Stage 1: HYBRID (AdaLoRA encoder + Full decoder) - FP32
    Stage 2: FULL QAT (Encoder + Decoder + Prompt) - INT8
    """
    
    # ========================================================================
    # STAGE 1: Hybrid Training (FP32)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STAGE 1: HYBRID TRAINING (FP32)")
    print("="*80)
    
    sam_model_stage1 = SamModel.from_pretrained(config['model_name'])
    model_stage1 = AdaLoRA_Sam(
        sam_model=sam_model_stage1,
        max_rank=config['stage1']['max_rank'],
        target_rank=config['stage1']['target_rank'],
        alpha=config['stage1']['lora_alpha'],
        lora_layers=None
    )
    
    # Enable gradient checkpointing
    try:
        if hasattr(model_stage1.sam.vision_encoder, 'gradient_checkpointing_enable'):
            model_stage1.sam.vision_encoder.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
    except:
        pass
    
    # Enable hybrid training
    model_stage1 = enable_decoder_training(model_stage1, config)
    
    if config['multi_gpu']:
        model_stage1 = nn.DataParallel(model_stage1)
    model_stage1.to(device)
    
    #  Datasets
    train_dataset = MedicalSegmentationDataset(
        splits['train']['images'], splits['train']['masks'], processor, target_size=config['image_size']
    )
    val_dataset = MedicalSegmentationDataset(
        splits['val']['images'], splits['val']['masks'], processor, target_size=config['image_size']
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['stage1']['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=config['pin_memory'], persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['stage1']['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=config['pin_memory'], persistent_workers=True
    )
    
    criterion = MedSAMLoss()
    optimizer_stage1 = create_differential_optimizer(model_stage1, config)
    metrics_computer = StandardSegmentationMetrics(include_background=False)
    early_stopping = EarlyStopping(patience=config['stage1']['patience'], min_delta=0.0001, mode='max')
    
    # Training Stage 1
    best_val_dice_stage1 = 0
    history_stage1 = {'train': [], 'val': []}
    
    for epoch in range(1, config['stage1']['num_epochs'] + 1):
        print(f"\n[Stage 1] Epoch {epoch}/{config['stage1']['num_epochs']}")
        print("-" * 80)
        
        # Pruning
        if epoch in config['stage1']['prune_at_epochs']:
            model_to_prune = model_stage1.module if config['multi_gpu'] else model_stage1
            model_to_prune.prune_to_target_rank()
        
        # Train
        train_metrics = train_epoch(
            model_stage1, train_loader, optimizer_stage1, criterion, device, epoch,
            metrics_computer, config['stage1']['accumulation_steps'], config['stage1']['ortho_reg_weight']
        )
        history_stage1['train'].append({'epoch': epoch, **train_metrics})
        
        print(f"Train | Loss: {train_metrics['main_loss']:.4f} | DSC: {train_metrics['dice']:.4f}")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Validate
        val_metrics = validate(model_stage1, val_loader, criterion, device, metrics_computer)
        history_stage1['val'].append({'epoch': epoch, **val_metrics})
        
        print(f"Val   | Loss: {val_metrics['loss']:.4f} | DSC: {val_metrics['dice']:.4f}±{val_metrics['dice_std']:.4f}")
        
        # Save best model
        if val_metrics['dice'] > best_val_dice_stage1:
            best_val_dice_stage1 = val_metrics['dice']
            model_to_save = model_stage1.module if config['multi_gpu'] else model_stage1
            
            stage1_weights_path = os.path.join(config['stage1']['save_dir'], 'best_adalora_stage1_fp32.pth')
            model_to_save.save_adalora_weights(stage1_weights_path)
            
            model_state = model_stage1.module.state_dict() if config['multi_gpu'] else model_stage1.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'val_dice': float(best_val_dice_stage1),
                'val_metrics': {k: float(v) for k, v in val_metrics.items() if not k.endswith('_std')},
            }
            torch.save(checkpoint, os.path.join(config['stage1']['save_dir'], 'best_model_stage1_fp32.pth'))
            print(f"✓ Stage 1 best model saved (DSC: {best_val_dice_stage1:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['dice']):
            print(f"\n[Stage 1] Early stopping at epoch {epoch}")
            break
    
    print("\n" + "="*80)
    print(f"STAGE 1 COMPLETE - Best Val DSC: {best_val_dice_stage1:.4f}")
    print("="*80 + "\n")
    
    # Clean up Stage 1
    del model_stage1, optimizer_stage1
    torch.cuda.empty_cache()
    gc.collect()
    
    # ========================================================================
    # STAGE 2: FULL QAT (INT8)
    # ========================================================================
    
    print("\n" + "="*80)
    print("STAGE 2: FULL QUANTIZATION AWARE TRAINING (INT8)")
    print("="*80)
    print("\n Quantizing: Encoder + Decoder + Prompt Encoder")
    print("="*80 + "\n")
    
    # Initialize Stage 2 model with quantization
    sam_model_stage2 = SamModel.from_pretrained(config['model_name'])
    
    #  APPLY FULL QUANTIZATION (Encoder + Decoder + Prompt)
    quant_count = apply_full_quantization_to_sam(
        sam_model_stage2, 
        bit_width=config['stage2']['bit_width'],
        skip_qkv=True  # Skip qkv, AdaLoRA will handle it
    )
    
    model_stage2 = AdaLoRA_Sam(
        sam_model=sam_model_stage2,
        max_rank=config['stage1']['max_rank'],
        target_rank=config['stage1']['target_rank'],
        alpha=config['stage1']['lora_alpha'],
        lora_layers=None
    )
    
    # Enable hybrid mode for Stage 2
    model_stage2 = enable_decoder_training(model_stage2, config)
    
    # Load Stage 1 weights
    print("\nLoading Stage 1 trained weights...")
    stage1_weights_path = os.path.join(config['stage1']['save_dir'], 'best_adalora_stage1_fp32.pth')
    model_stage2.load_adalora_weights(stage1_weights_path, strict=True)
    
    # Load full checkpoint
    stage1_checkpoint = torch.load(
        os.path.join(config['stage1']['save_dir'], 'best_model_stage1_fp32.pth'),
        map_location='cpu',
        weights_only=False
    )
    
    model_stage2.load_state_dict(stage1_checkpoint['model_state_dict'], strict=False)
    print("✓ Full model weights loaded from Stage 1")
    
    # Enable QAT mode
    model_stage2.enable_qat_mode()
    
    # Freeze singular values
    if config['stage2']['freeze_singular_values']:
        model_stage2.freeze_singular_values()
    
    # Enable QAT for all quantized layers
    if BREVITAS_AVAILABLE:
        enable_qat_mode(model_stage2)
    
    # Gradient checkpointing
    try:
        if hasattr(model_stage2.sam.vision_encoder, 'gradient_checkpointing_enable'):
            model_stage2.sam.vision_encoder.gradient_checkpointing_enable()
    except:
        pass
    
    if config['multi_gpu']:
        model_stage2 = nn.DataParallel(model_stage2)
    model_stage2.to(device)
    
    # Create optimizer for Stage 2
    optimizer_stage2 = create_differential_optimizer(model_stage2, config)
    
    # Use very low learning rates for QAT
    for param_group in optimizer_stage2.param_groups:
        param_group['lr'] = config['stage2']['learning_rate']
    
    # Training Stage 2
    best_val_dice_stage2 = 0
    history_stage2 = {'train': [], 'val': []}
    
    for epoch in range(1, config['stage2']['num_epochs'] + 1):
        print(f"\n[Stage 2 - QAT] Epoch {epoch}/{config['stage2']['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(
            model_stage2, train_loader, optimizer_stage2, criterion, device, epoch,
            metrics_computer, config['stage2']['accumulation_steps'], config['stage2']['ortho_reg_weight']
        )
        history_stage2['train'].append({'epoch': epoch, **train_metrics})
        
        print(f"Train | Loss: {train_metrics['main_loss']:.4f} | DSC: {train_metrics['dice']:.4f}")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Validate
        val_metrics = validate(model_stage2, val_loader, criterion, device, metrics_computer)
        history_stage2['val'].append({'epoch': epoch, **val_metrics})
        
        print(f"Val   | Loss: {val_metrics['loss']:.4f} | DSC: {val_metrics['dice']:.4f}±{val_metrics['dice_std']:.4f}")
        
        # Save best model
        if val_metrics['dice'] > best_val_dice_stage2:
            best_val_dice_stage2 = val_metrics['dice']
            model_to_save = model_stage2.module if config['multi_gpu'] else model_stage2
            
            stage2_weights_path = os.path.join(config['stage2']['save_dir'], 'best_adalora_stage2_int8.pth')
            model_to_save.save_adalora_weights(stage2_weights_path)
            
            model_state = model_stage2.module.state_dict() if config['multi_gpu'] else model_stage2.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'val_dice': float(best_val_dice_stage2),
                'val_metrics': {k: float(v) for k, v in val_metrics.items() if not k.endswith('_std')},
                'stage1_dice': float(best_val_dice_stage1),
                'quantized_layers': quant_count
            }
            torch.save(checkpoint, os.path.join(config['stage2']['save_dir'], 'best_model_stage2_int8.pth'))
            print(f"✓ Stage 2 best model saved (DSC: {best_val_dice_stage2:.4f})")
    
    print("\n" + "="*80)
    print(f"STAGE 2 COMPLETE - Best Val DSC: {best_val_dice_stage2:.4f}")
    print(f"Performance: Stage 1 ({best_val_dice_stage1:.4f}) → Stage 2 ({best_val_dice_stage2:.4f})")
    degradation = (best_val_dice_stage1 - best_val_dice_stage2) / best_val_dice_stage1 * 100
    print(f"QAT Degradation: {degradation:.2f}%")
    
    if degradation < 3:
        print(" EXCELLENT! Full quantization degradation < 3%")
    elif degradation < 5:
        print("✓ GOOD! Full quantization degradation < 5%")
    else:
        print("  Higher degradation than target")
    
    print("="*80 + "\n")
    
    return model_stage2, best_val_dice_stage1, best_val_dice_stage2, history_stage1, history_stage2


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function for HYBRID training with FULL quantization"""
    
    config = {
        'data_dir': '/home/prantik.d/Srimanth/data/Final_dataset_split_Resized',
        'model_name': 'facebook/sam-vit-base',
        'image_size': 512,
        'use_amp': True,
        
        # ====================================================================
        # STAGE 1: HYBRID (AdaLoRA Encoder + Full Decoder) - FP32
        # ====================================================================
        'stage1': {
            'max_rank': 48,
            'target_rank': 32,
            'lora_alpha': 32.0,
            'ortho_reg_weight': 0.003,
            'batch_size': 16,
            'accumulation_steps': 1,
            'num_epochs': 25,
            
            'learning_rate': 5e-5,       
            'lr_decoder': 2e-5,          
            'lr_prompt': 2e-5,          
            
            'weight_decay': 1e-4,
            'prune_at_epochs': [3, 7, 12],
            'patience': 8,
            
            'train_decoder': True,
            'train_prompt_encoder': True,
            
            'save_dir': './checkpoints_stage1_fp32',
        },
        
        # ====================================================================
        # STAGE 2: FULL QAT (Encoder + Decoder + Prompt) - INT8
        # ====================================================================
        'stage2': {
            'bit_width': 8,
            'ortho_reg_weight': 0.0,
            'batch_size': 16,
            'accumulation_steps': 1,
            'num_epochs': 10,
            'learning_rate': 5e-7,       
            'weight_decay': 1e-5,
            'freeze_singular_values': True,
            'save_dir': './checkpoints_stage2_int8_full',
        },
        
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
    }
    
    # Device setup
    if torch.cuda.device_count() > 1:
        config['device'] = 'cuda'
        config['multi_gpu'] = True
        print(f"Using {torch.cuda.device_count()} GPUs")
    elif torch.cuda.is_available():
        config['device'] = 'cuda'
        config['multi_gpu'] = False
    else:
        config['device'] = 'cpu'
        config['multi_gpu'] = False
    
    os.makedirs(config['stage1']['save_dir'], exist_ok=True)
    os.makedirs(config['stage2']['save_dir'], exist_ok=True)
    
    
    # Load data
    print(f"Loading data from: {config['data_dir']}")
    splits = {}
    for split_name in ['train', 'val', 'test']:
        img_dir = os.path.join(config['data_dir'], split_name, 'images')
        mask_dir = os.path.join(config['data_dir'], split_name, 'masks')
        
        valid_ext = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_ext)])
        
        image_paths = [os.path.join(img_dir, f) for f in image_files]
        mask_paths = [os.path.join(mask_dir, f) for f in image_files]
        
        splits[split_name] = {
            'images': image_paths,
            'masks': mask_paths
        }
        print(f"  {split_name.upper()}: {len(image_paths)} samples")
    
    # Initialize processor
    processor = SamProcessor.from_pretrained(config['model_name'])
    
    # Run two-stage training
    model_final, best_dice_stage1, best_dice_stage2, history_s1, history_s2 = two_stage_training_full_quant(
        config, splits, processor, config['device']
    )
    
    # ========================================================================
    # Final Test Evaluation
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(" FINAL TEST EVALUATION")
    print("=" * 80)
    # Load checkpoint
    checkpoint_path = os.path.join(config['stage2']['save_dir'], 'best_model_stage2_int8.pth')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"\nLoading best Stage 2 model (INT8) from epoch {checkpoint['epoch']}")
    # Handle DataParallel properly
    if config['multi_gpu']:
        # Unwrap, load, move to device, rewrap
        model_unwrapped = model_final.module
        model_unwrapped.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model_unwrapped = model_unwrapped.to(config['device'])
        model_final = nn.DataParallel(model_unwrapped)
    else:
        model_final.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model_final = model_final.to(config['device'])

    # Ensure device sync
    if config['device'] != 'cpu':
        torch.cuda.synchronize()

    print(f"Model loaded and moved to {config['device']}")
    
    # Create test dataset
    test_dataset = MedicalSegmentationDataset(
        splits['test']['images'], splits['test']['masks'], 
        processor, target_size=config['image_size']
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Evaluate
    criterion = MedSAMLoss()
    metrics_computer = StandardSegmentationMetrics(include_background=False)
    test_metrics = validate(model_final, test_loader, criterion, config['device'], metrics_computer)
    
    print("\n Test Results (mean ± std):")
    print(f"  DSC:   {test_metrics['dice']:.4f} ± {test_metrics['dice_std']:.4f}")
    print(f"  IoU:   {test_metrics['iou']:.4f} ± {test_metrics['iou_std']:.4f}")
    print(f"  NSD:   {test_metrics['nsd']:.4f} ± {test_metrics['nsd_std']:.4f}")
    print(f"  HD95:  {test_metrics['hd95']:.2f} ± {test_metrics['hd95_std']:.2f} px")
    print(f"  ASSD:  {test_metrics['assd']:.2f} ± {test_metrics['assd_std']:.2f} px")
    
    # ========================================================================
    # Quantization Analysis
    # ========================================================================
    
    model_for_stats = model_final.module if config['multi_gpu'] else model_final
    
    # Count parameters by type
    total_params = 0
    fp32_params = 0
    int8_params = 0
    
    for name, module in model_for_stats.sam.named_modules():
        if isinstance(module, qnn.QuantLinear):
            params = sum(p.numel() for p in module.parameters())
            int8_params += params
            total_params += params
        elif isinstance(module, nn.Linear):
            params = sum(p.numel() for p in module.parameters())
            fp32_params += params
            total_params += params
    
    # Add AdaLoRA parameters (FP32)
    adalora_params = sum(p.numel() for p in model_for_stats.adalora_modules.parameters())
    fp32_params += adalora_params
    total_params += adalora_params
    
    # Calculate sizes
    fp32_size_mb = (fp32_params * 4) / (1024 ** 2)
    int8_size_mb = (int8_params * 1) / (1024 ** 2)
    total_quantized_mb = fp32_size_mb + int8_size_mb
    original_size_mb = (total_params * 4) / (1024 ** 2)
    compression_ratio = original_size_mb / total_quantized_mb
    
    print("\n" + "=" * 80)
    print(" MODEL COMPRESSION ANALYSIS")
    print("=" * 80)
    print(f"\nParameter Distribution:")
    print(f"  Total Parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  FP32 Parameters:       {fp32_params:,} ({fp32_params/1e6:.2f}M, {100*fp32_params/total_params:.1f}%)")
    print(f"  INT8 Parameters:       {int8_params:,} ({int8_params/1e6:.2f}M, {100*int8_params/total_params:.1f}%)")
    
    print(f"\nModel Size:")
    print(f"  Original (FP32):       {original_size_mb:.2f} MB")
    print(f"  Quantized (Mixed):     {total_quantized_mb:.2f} MB")
    print(f"    ├─ FP32 components:  {fp32_size_mb:.2f} MB")
    print(f"    └─ INT8 components:  {int8_size_mb:.2f} MB")
    
    print(f"\nCompression:")
    print(f"  Compression Ratio:     {compression_ratio:.2f}x")
    print(f"  Size Reduction:        {(1 - 1/compression_ratio)*100:.1f}%")
    print(f"  Quantized Coverage:    {100*int8_params/total_params:.1f}%")
    
    # Target comparison
    target_compression = 2.7
    if compression_ratio >= target_compression:
        print(f"\n   EXCELLENT! Achieved {compression_ratio:.2f}x ≥ {target_compression}x target!")
    elif compression_ratio >= target_compression * 0.9:
        print(f"\n  ✓ GOOD! Close to {target_compression}x target")
    else:
        print(f"\n    Below {target_compression}x target compression")
    
    # ========================================================================
    # Performance Analysis
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(" FULL QUANTIZATION PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    print(f"\n Training Results:")
    print(f"  Stage 1 (FP32):        {best_dice_stage1:.4f} DSC")
    print(f"  Stage 2 (INT8 Full):   {best_dice_stage2:.4f} DSC")
    print(f"  Final Test (INT8):     {test_metrics['dice']:.4f} DSC")
    
    val_degradation = (best_dice_stage1 - best_dice_stage2) / best_dice_stage1 * 100
    test_degradation = (best_dice_stage1 - test_metrics['dice']) / best_dice_stage1 * 100
    
    print(f"\n Performance Degradation:")
    print(f"  Val:  {best_dice_stage1:.4f} → {best_dice_stage2:.4f} ({val_degradation:+.2f}%)")
    print(f"  Test: {best_dice_stage1:.4f} → {test_metrics['dice']:.4f} ({test_degradation:+.2f}%)")
    
    if test_degradation < 3:
        print(f"   EXCELLENT! Full quantization degradation < 3%")
    elif test_degradation < 5:
        print(f"  ✓ GOOD! Full quantization degradation < 5%")
    else:
        print(f"    Higher degradation than ideal")
    
    # Comparison to baseline
    baseline_dice = 0.95
    print(f"\n Comparison to Baseline:")
    print(f"  Baseline (Decoder FT, FP32):  {baseline_dice:.4f} DSC, 360 MB")
    print(f"  Our Method (INT8 Full):       {test_metrics['dice']:.4f} DSC, {total_quantized_mb:.0f} MB")
    
    gap_from_baseline = (baseline_dice - test_metrics['dice']) / baseline_dice * 100
    print(f"\n  Performance Gap:              {gap_from_baseline:.2f}%")
    print(f"  Model Size Reduction:         {compression_ratio:.2f}x smaller")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    
    results = {
        'metadata': {
            'framework': 'PyTorch + Hugging Face + Brevitas',
            'model': 'SAM ViT-Base',
            'method': 'HYBRID: AdaLoRA + Full Decoder + FULL QAT',
            'timestamp': datetime.now().isoformat()
        },
        'config': {
            'stage1': config['stage1'],
            'stage2': config['stage2'],
            'data_dir': config['data_dir'],
        },
        'training': {
            'stage1_best_val_dice': float(best_dice_stage1),
            'stage2_best_val_dice': float(best_dice_stage2),
            'val_degradation_percent': float(val_degradation),
            'test_degradation_percent': float(test_degradation),
        },
        'model': {
            'total_parameters': int(total_params),
            'fp32_parameters': int(fp32_params),
            'int8_parameters': int(int8_params),
            'quantized_percentage': float(100 * int8_params / total_params),
        },
        'compression': {
            'original_mb': float(original_size_mb),
            'quantized_mb': float(total_quantized_mb),
            'fp32_mb': float(fp32_size_mb),
            'int8_mb': float(int8_size_mb),
            'compression_ratio': float(compression_ratio),
            'size_reduction_pct': float((1 - 1/compression_ratio) * 100),
        },
        'test_metrics': {
            'dice': {'mean': float(test_metrics['dice']), 'std': float(test_metrics['dice_std'])},
            'iou': {'mean': float(test_metrics['iou']), 'std': float(test_metrics['iou_std'])},
            'nsd': {'mean': float(test_metrics['nsd']), 'std': float(test_metrics['nsd_std'])},
            'hd95': {'mean': float(test_metrics['hd95']), 'std': float(test_metrics['hd95_std'])},
            'assd': {'mean': float(test_metrics['assd']), 'std': float(test_metrics['assd_std'])}
        },
        'comparison': {
            'baseline_dice': float(baseline_dice),
            'gap_from_baseline_percent': float(gap_from_baseline)
        },
        'dataset': {
            'train_samples': len(splits['train']['images']),
            'val_samples': len(splits['val']['images']),
            'test_samples': len(splits['test']['images'])
        }
    }
    
    results_path = os.path.join(config['stage2']['save_dir'], 'results_full_quantization.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n Results saved to: {results_path}")
    print(f" Stage 1 model (FP32): {config['stage1']['save_dir']}/best_model_stage1_fp32.pth")
    print(f" Stage 2 model (INT8): {config['stage2']['save_dir']}/best_model_stage2_int8.pth")
    
    print("\n" + "=" * 80)
    print("=" * 80)
    print(f"\n Method: Hybrid AdaLoRA + Full Decoder + FULL QAT")
    print(f"  Stage 1 (FP32):        {best_dice_stage1:.4f} DSC")
    print(f"  Stage 2 (INT8 Full):   {best_dice_stage2:.4f} DSC")
    print(f"  Test Performance:      {test_metrics['dice']:.4f} ± {test_metrics['dice_std']:.4f} DSC")
    
    print(f"\n  Quantization:")
    print(f"  Encoder: INT8 ({100*int8_params/total_params:.0f}% of model)")
    print(f"  Decoder: INT8")
    print(f"  Prompt:  INT8")
    print(f"  AdaLoRA: FP32 (rank {config['stage1']['target_rank']})")
    
    print(f"\n Efficiency:")
    print(f"  Compression:  {compression_ratio:.2f}x")
    print(f"  Model Size:   {original_size_mb:.0f} MB → {total_quantized_mb:.0f} MB")
    print(f"  Quantized:    {100*int8_params/total_params:.0f}% of parameters")
    
    print(f"\n Performance vs Baseline:")
    print(f"  Baseline:     {baseline_dice:.4f} DSC, 360 MB (FP32)")
    print(f"  Our Method:   {test_metrics['dice']:.4f} DSC, {total_quantized_mb:.0f} MB (INT8)")
    print(f"  Gap:          {gap_from_baseline:.2f}%")
    print(f"  Compression:  {compression_ratio:.2f}x smaller")
    


if __name__ == "__main__":
    main()
