#!/usr/bin/env python3
"""
Training script for FMRIFlamingo model.

Usage:
    python scripts/train.py [--resume CHECKPOINT_PATH] [--device DEVICE]
    python scripts/train.py --a100   # Optimize for A100 80GB (batch=8, TF32, 4 workers, no grad checkpointing)

Features:
    - Early stopping based on validation loss
    - Checkpoint saving/loading
    - Gradient accumulation
    - Mixed precision training (BF16/FP16)
    - Learning rate scheduling with warmup
    - Memory monitoring and logging
    - Safety stop (opt-in)
    - Loss history tracking
"""

import sys
import os
# Disable tokenizer parallelism to prevent deadlocks/slowness in DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from bitsandbytes.optim import AdamW8bit as AdamW
except ImportError:
    from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
from datetime import datetime
import psutil  # For memory monitoring

# Metrics imports
from jiwer import wer
try:
    from evaluate import load as load_metric
    HAS_EVALUATE = True
except ImportError:
    try:
        from datasets import load_metric  # legacy
        HAS_EVALUATE = True
    except ImportError:
        HAS_EVALUATE = False
        print("⚠️  Warning: 'evaluate' or 'datasets' library not found. BLEU/METEOR will be skipped.")

# Add paths
SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

# Import project modules
from src.datasets.huth_fmri_dataset import HuthFMRIDataset
from src.models.fmri_flamingo import FMRIFlamingo

# Import config
import importlib.util
config_path = FMRI_FLAMINGO_DIR / "config.py"
spec = importlib.util.spec_from_file_location("fmri_config", config_path)
fmri_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmri_config)

# ============================================================================
# Configuration
# ============================================================================

# Training parameters (from config)
BATCH_SIZE = getattr(fmri_config, "BATCH_SIZE", 1)
GRADIENT_ACCUMULATION_STEPS = getattr(fmri_config, "GRADIENT_ACCUMULATION_STEPS", 4)
NUM_EPOCHS = getattr(fmri_config, "NUM_EPOCHS", 20)
LR_ENCODER = getattr(fmri_config, "LR_ENCODER", 2e-4)
LR_PROJECTOR = getattr(fmri_config, "LR_PROJECTOR", 1e-4)
LR_BASE = getattr(fmri_config, "LR_BASE", 2e-4)
LR_LLM = getattr(fmri_config, "LR_LLM", 0.0)
WARMUP_FRAC = getattr(fmri_config, "WARMUP_FRAC", 0.03)
WEIGHT_DECAY = getattr(fmri_config, "WEIGHT_DECAY", 1e-2)
GRAD_CLIP_NORM = getattr(fmri_config, "GRAD_CLIP_NORM", 1.0)
EARLY_STOP_PATIENCE = getattr(fmri_config, "EARLY_STOP_PATIENCE", 5)
SAVE_EVERY_N_EPOCHS = getattr(fmri_config, "SAVE_EVERY_N_EPOCHS", 1)
KEEP_N_CHECKPOINTS = getattr(fmri_config, "KEEP_N_CHECKPOINTS", 3)
TEXT_MASKING_PROB = getattr(fmri_config, "TEXT_MASKING_PROB", 0.0)
RANKING_LOSS_WEIGHT = getattr(fmri_config, "RANKING_LOSS_WEIGHT", 0.5)
RANKING_LOSS_FRAC = getattr(fmri_config, "RANKING_LOSS_FRAC", 0.5)
NUM_RANKING_DISTRACTORS = getattr(fmri_config, "NUM_RANKING_DISTRACTORS", 7)
RANKING_CHUNK_SIZE = getattr(fmri_config, "RANKING_CHUNK_SIZE", 2)
TELEPATHY_RANKING_FRAC = getattr(fmri_config, "TELEPATHY_RANKING_FRAC", 1.0)
RESTRICT_GENERATION_TO_WORD_VOCAB = getattr(fmri_config, "RESTRICT_GENERATION_TO_WORD_VOCAB", False)

# Model parameters
LLM_ID = getattr(fmri_config, "LLM_ID", "meta-llama/Llama-3.2-1B")
NUM_ROIS = getattr(fmri_config, "NUM_ROIS", 200)
CROSS_ATTN_EVERY_N_LAYERS = getattr(fmri_config, "CROSS_ATTN_EVERY_N_LAYERS", 4)
USE_GRADIENT_CHECKPOINTING = getattr(fmri_config, "USE_GRADIENT_CHECKPOINTING", True)
GRADIENT_CHECKPOINTING = getattr(fmri_config, "GRADIENT_CHECKPOINTING", USE_GRADIENT_CHECKPOINTING)

# Dataset parameters
WINDOW_SIZE = getattr(fmri_config, "DEFAULT_WINDOW_SIZE", 10)
STRIDE = getattr(fmri_config, "DEFAULT_STRIDE", 10)

# Memory optimization
USE_MIXED_PRECISION = getattr(fmri_config, "USE_MIXED_PRECISION", True)
MIXED_PRECISION_DTYPE = getattr(fmri_config, "MIXED_PRECISION_DTYPE", "bf16")
DATALOADER_NUM_WORKERS = getattr(fmri_config, "DATALOADER_NUM_WORKERS", 0)

# Observability
ENABLE_MEMORY_LOGGING = getattr(fmri_config, "ENABLE_MEMORY_LOGGING", True)
MEMORY_LOG_INTERVAL = getattr(fmri_config, "MEMORY_LOG_INTERVAL", 10)
ENABLE_SAFETY_STOP = getattr(fmri_config, "ENABLE_SAFETY_STOP", False)
SAFETY_STOP_RSS_GB = getattr(fmri_config, "SAFETY_STOP_RSS_GB", 12.0)
# Log grad norms for encoder and perceiver every N optimizer steps (0 = disable). Use to confirm gradients flow when loss won't go down.
LOG_GRAD_NORMS_EVERY = getattr(fmri_config, "LOG_GRAD_NORMS_EVERY", 0)

# Paths
CHECKPOINT_DIR = getattr(fmri_config, "CHECKPOINT_DIR", FMRI_FLAMINGO_DIR / "checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOSS_HISTORY_PATH = CHECKPOINT_DIR / "loss_history.json"
LOGS_DIR = getattr(fmri_config, "LOGS_DIR", FMRI_FLAMINGO_DIR / "logs")
LOGS_DIR.mkdir(exist_ok=True)
METRICS_LOG_PATH = LOGS_DIR / "metrics.jsonl"

# ============================================================================
# Helper Functions
# ============================================================================

def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def log_memory_metrics(step: int, epoch: int, device: torch.device):
    """Log memory metrics to JSONL file."""
    if not ENABLE_MEMORY_LOGGING:
        return
    
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024**3)
    
    system_mem = psutil.virtual_memory()
    available_gb = system_mem.available / (1024**3)
    
    metrics = {
        "step": step,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "rss_gb": round(rss_gb, 3),
        "system_available_ram_gb": round(available_gb, 3),
    }
    
    if device.type == "cuda":
        metrics["cuda_allocated_gb"] = round(torch.cuda.memory_allocated(device) / 1024**3, 3)
        metrics["cuda_reserved_gb"] = round(torch.cuda.memory_reserved(device) / 1024**3, 3)
    
    with open(METRICS_LOG_PATH, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def check_safety_stop(epoch: int, step: int) -> bool:
    """Check if we should stop due to high memory usage."""
    if not ENABLE_SAFETY_STOP:
        return False
    
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024**3)
    
    if rss_gb > SAFETY_STOP_RSS_GB:
        return True
    
    return False


def save_checkpoint(
    model: FMRIFlamingo,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> Path:
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": {
            "llm_id": LLM_ID,
            "num_rois": NUM_ROIS,
            "batch_size": BATCH_SIZE,
        },
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best_checkpoint.pt"
        torch.save(checkpoint, best_path)
        print(f"✅ Saved best checkpoint to {best_path}")
    
    # Clean up old checkpoints (keep only last N)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) > KEEP_N_CHECKPOINTS:
        for old_checkpoint in checkpoints[:-KEEP_N_CHECKPOINTS]:
            old_checkpoint.unlink()
            print(f"🗑️  Removed old checkpoint: {old_checkpoint.name}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: FMRIFlamingo,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
) -> Dict[str, Any]:
    """Load model checkpoint."""
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    try:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    except (ValueError, KeyError) as e:
        print(f"⚠️  Optimizer/scheduler state mismatch (param groups changed), starting fresh: {e}")
        print(f"   Model weights loaded successfully — only optimizer state is reset.")
    
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("val_loss", float("inf"))
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Train loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"   Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return {
        "start_epoch": start_epoch,
        "best_val_loss": best_val_loss,
    }


def load_loss_history() -> list:
    """Load loss history from file."""
    if LOSS_HISTORY_PATH.exists():
        with open(LOSS_HISTORY_PATH, "r") as f:
            return json.load(f)
    return []


def save_loss_history(history: list):
    """Save loss history to file."""
    with open(LOSS_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def print_memory_stats(device: torch.device):
    """Print memory statistics."""
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"   GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    elif device.type == "mps":
        print(f"   Using MPS device")


def _log_grad_norms(model: FMRIFlamingo, step: int, epoch: int):
    """Log L2 grad norms for encoder and perceiver (diagnostic when loss won't go down)."""
    enc_sq, perc_sq = 0.0, 0.0
    enc_parts = {}  # per-param breakdown
    for name, p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        g = p.grad.float().norm().item() ** 2
        if "vision_encoder" in name or "tokenizer" in name.lower():
            enc_sq += g
            # Track top contributors
            short_name = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
            enc_parts[short_name] = g ** 0.5
        elif "perceiver" in name.lower():
            perc_sq += g
    enc_norm = enc_sq ** 0.5
    perc_norm = perc_sq ** 0.5
    tqdm.write(f"  [epoch {epoch} step {step}] grad_norm: encoder={enc_norm:.4f} perceiver={perc_norm:.4f}")
    if enc_parts and enc_norm > 10:
        top = sorted(enc_parts.items(), key=lambda x: -x[1])[:5]
        tqdm.write(f"    encoder breakdown: {', '.join(f'{n}={v:.1f}' for n, v in top)}")


# ============================================================================
# Training Functions
# ============================================================================

def create_optimizer(model: FMRIFlamingo) -> torch.optim.Optimizer:
    """Create optimizer with different learning rates for different components."""
    # Separate parameters by component
    encoder_params = []
    projector_params = []
    base_params = []
    llm_params = []
    
    # Freeze LLM backbone if not training it (saves ~11GB of unused gradients)
    if LR_LLM == 0.0:
        for name, param in model.named_parameters():
            if "lang_encoder" in name and "gated_cross_attn_layer" not in name:
                param.requires_grad = False

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "vision_encoder" in name or "tokenizer" in name.lower():
            encoder_params.append(param)
        elif "projector" in name.lower() or "perceiver" in name.lower():
            projector_params.append(param)
        elif "gated_cross_attn_layer" in name:
            # Cross-attention layers are inside lang_encoder but need to be trained
            base_params.append(param)
        elif "lang_encoder" in name:
            llm_params.append(param)
        else:
            base_params.append(param)
    
    # Create parameter groups
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": LR_ENCODER})
    if projector_params:
        param_groups.append({"params": projector_params, "lr": LR_PROJECTOR})
    if base_params:
        param_groups.append({"params": base_params, "lr": LR_BASE})
    if llm_params and LR_LLM > 0:
        param_groups.append({"params": llm_params, "lr": LR_LLM})
    
    optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    
    print(f"📊 Optimizer parameter groups:")
    for i, group in enumerate(param_groups):
        print(f"   Group {i+1}: {len(group['params'])} params, LR={group['lr']:.2e}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler with warmup."""
    num_warmup_steps = int(num_training_steps * WARMUP_FRAC)
    num_cosine_steps = num_training_steps - num_warmup_steps
    
    # Warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )
    
    # Cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_cosine_steps,
        eta_min=1e-6,
    )
    
    # Sequential scheduler (warmup then cosine)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps],
    )
    
    print(f"📈 Learning rate schedule:")
    print(f"   Warmup steps: {num_warmup_steps} ({WARMUP_FRAC*100:.1f}% of training)")
    print(f"   Cosine steps: {num_cosine_steps}")
    
    return scheduler


def _compute_ranking_loss(
    model: FMRIFlamingo,
    batch: Dict[str, Any],
    device: torch.device,
    tokenizer: Any,
    distractor_queue: deque,
    pad_token_id: int,
    scaler=None,
    telepathy: bool = False,
) -> Optional[float]:
    """
    Contrastive ranking loss: score correct > distractors given (prompt + fMRI) or fMRI only.
    When telepathy=True: no prompt, only BOS + fMRI (trains for Telepathy eval).
    Backwards PER SAMPLE to avoid OOM.
    Returns mean loss for logging or None if skipped.
    """
    B = len(batch["input_ids"])
    prompt_lens = batch["prompt_len"]
    if isinstance(prompt_lens, torch.Tensor):
        prompt_lens = prompt_lens.tolist()
    prompts = []
    targets = []
    for i in range(B):
        ids = batch["input_ids"][i]
        if isinstance(ids, torch.Tensor):
            ids = ids.to(device)
        else:
            ids = torch.tensor(ids, device=device)
        pl = prompt_lens[i]
        prompts.append(ids[:pl])
        targets.append(ids[pl:])
    if B >= 2:
        pass
    else:
        if len(distractor_queue) < 1:
            return None
        k = min(NUM_RANKING_DISTRACTORS, len(distractor_queue))
        sampled = random.sample(list(distractor_queue), k)
        correct_list = targets[0].tolist()
        distractors = [torch.tensor(s, device=device, dtype=targets[0].dtype) for s in sampled if s != correct_list][:k]
        if len(distractors) < 1:
            return None
        targets = [targets[0]] + distractors

    # Telepathy: no prompt, only BOS + fMRI (matches evaluate_ranking_telepathy.py)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else pad_token_id
    if telepathy:
        prompts = [torch.tensor([bos_id], device=device, dtype=targets[0].dtype) for _ in range(B)]
        prompt_lens = [1] * B

    chunk_size = max(1, RANKING_CHUNK_SIZE)
    mean_loss = 0.0
    count = 0
    for i in range(B):
        if B >= 2:
            candidates_i = [targets[k] for k in range(B)]
            correct_idx = i
        else:
            candidates_i = targets
            correct_idx = 0
        prompt_i = prompts[i]
        prompt_len = prompt_lens[i]
        n_cand = len(candidates_i)
        mini_batch = {
            "time_series": [batch["time_series"][i]],
            "input_ids": [batch["input_ids"][i]],
            "prompt_len": [batch["prompt_len"][i]],
        }
        _, images, _, _ = model.pad_and_apply_batch(mini_batch, include_labels=False)
        seqs = [torch.cat([prompt_i, c], dim=0) for c in candidates_i]
        max_len = max(s.size(0) for s in seqs)
        all_per_seq_loss = []
        for start in range(0, n_cand, chunk_size):
            end = min(start + chunk_size, n_cand)
            chunk_candidates = candidates_i[start:end]
            cs = len(chunk_candidates)
            vision_x = images.repeat(cs, 1, 1, 1)
            padded_list = []
            attn_list = []
            for s in [torch.cat([prompt_i, c], dim=0) for c in chunk_candidates]:
                L = s.size(0)
                if L < max_len:
                    pad = torch.full((max_len - L,), pad_token_id, device=device, dtype=s.dtype)
                    padded_list.append(torch.cat([s, pad], dim=0))
                    attn_list.append(torch.cat([torch.ones(L, device=device), torch.zeros(max_len - L, device=device)]))
                else:
                    padded_list.append(s)
                    attn_list.append(torch.ones(L, device=device))
            lang_x = torch.stack(padded_list)
            attention_mask = torch.stack(attn_list)
            labels = lang_x.clone()
            labels[:, :prompt_len] = -100
            labels[attention_mask == 0] = -100
            outputs = model.model(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            flat_loss = loss_fct(flat_logits, flat_labels)
            per_token = flat_loss.view(shift_labels.shape)
            mask_ce = (shift_labels != -100).float()
            per_seq_chunk = (per_token * mask_ce).sum(dim=1) / (mask_ce.sum(dim=1) + 1e-8)
            all_per_seq_loss.append(per_seq_chunk)
            del outputs, logits, vision_x, lang_x
        per_seq_loss = torch.cat(all_per_seq_loss, dim=0)
        log_sum_exp = torch.logsumexp(-per_seq_loss, dim=0).clamp(max=50.0)
        l_correct = per_seq_loss[correct_idx]
        rank_loss_i = (l_correct + log_sum_exp) * RANKING_LOSS_WEIGHT
        mean_loss += rank_loss_i.item()
        count += 1
        # Backward this sample immediately so we free its graph (avoids OOM from B samples)
        if scaler is not None:
            scaler.scale(rank_loss_i).backward()
        else:
            rank_loss_i.backward()
    if count == 0:
        return None
    return mean_loss / count


def train_epoch(
    model: FMRIFlamingo,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    scaler=None,  # For mixed precision
    tokenizer=None,  # For masking and ranking
    distractor_queue: Optional[deque] = None,
) -> float:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    prog = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]",
        leave=False,
    )
    
    optimizer.zero_grad(set_to_none=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
    if distractor_queue is None:
        distractor_queue = deque(maxlen=500)

    for batch_idx, batch in enumerate(prog):
        # --- Text Masking for Posterior Collapse Fix ---
        # If enabled, mask a percentage of input tokens to force fMRI usage
        if TEXT_MASKING_PROB > 0.0:
            # 1. Get standard processed batch (tensors on device)
            input_ids_orig, images, attention_mask, labels = model.pad_and_apply_batch(batch, include_labels=True)
            
            # 2. Apply masking to input_ids_orig
            masked_input_ids = input_ids_orig.clone()
            
            # Generate random mask
            rand_mask = torch.rand(masked_input_ids.shape, device=masked_input_ids.device) < TEXT_MASKING_PROB
            
            # Don't mask padding (0 or pad_token_id)
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            rand_mask = rand_mask & (masked_input_ids != pad_token_id)
            
            # Apply mask (use pad_token_id as mask if no mask token)
            masked_input_ids[rand_mask] = pad_token_id
            
            # Zero out attention at masked positions so the model does not attend to them
            # (otherwise we train on "attend to PAD" instead of "position missing, use fMRI")
            masked_attention_mask = attention_mask.clone()
            masked_attention_mask[rand_mask] = 0
            
            # 3. Create a "pre-processed" batch dict that compute_loss will recognize
            batch_masked = {
                "lang_x": masked_input_ids,
                "labels": labels,  # Original labels (model predicts masked tokens from fMRI + context)
                "attention_mask": masked_attention_mask,
            }
            # Copy original batch keys for vision processing (time_series is needed)
            if isinstance(batch, dict):
                batch_masked.update(batch)
            
            # Call compute_loss with our special batch
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16 if MIXED_PRECISION_DTYPE == "bf16" else torch.float16):
                    loss = model.compute_loss(batch_masked)
            else:
                loss = model.compute_loss(batch_masked)
        else:
            # Forward pass (with mixed precision if enabled)
            if scaler is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16 if MIXED_PRECISION_DTYPE == "bf16" else torch.float16):
                    loss = model.compute_loss(batch)
            else:
                loss = model.compute_loss(batch)

        # --- Contrastive ranking loss: run in a SEPARATE step to avoid OOM (CE graph freed first) ---
        do_ranking_this_batch = (
            RANKING_LOSS_WEIGHT > 0
            and RANKING_LOSS_FRAC > 0
            and tokenizer is not None
            and random.random() < RANKING_LOSS_FRAC
            and (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0  # only when we're about to step
        )
        if do_ranking_this_batch:
            loss = loss  # CE only for first backward
        # else: loss stays CE-only

        # Scale loss for gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS

        # Backward pass (CE only)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights (every GRADIENT_ACCUMULATION_STEPS batches)
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            # Gradient clipping — per param group so encoder can't starve perceiver
            if scaler is not None:
                scaler.unscale_(optimizer)
            for group in optimizer.param_groups:
                clip_grad_norm_(group["params"], GRAD_CLIP_NORM)

            # Optional: log grad norms to confirm encoder/perceiver are updating (set LOG_GRAD_NORMS_EVERY in config)
            step_count = (batch_idx + 1) // GRADIENT_ACCUMULATION_STEPS
            if LOG_GRAD_NORMS_EVERY > 0 and step_count > 0 and step_count % LOG_GRAD_NORMS_EVERY == 0:
                _log_grad_norms(model, step_count, epoch)

            # Optimizer step (CE)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Ranking step: Telepathy (no prompt) or full-prompt ranking, per TELEPATHY_RANKING_FRAC
            if do_ranking_this_batch:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                use_telepathy = random.random() < TELEPATHY_RANKING_FRAC
                if scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16 if MIXED_PRECISION_DTYPE == "bf16" else torch.float16):
                        rank_mean = _compute_ranking_loss(
                            model, batch, device, tokenizer, distractor_queue, pad_token_id,
                            scaler=scaler, telepathy=use_telepathy,
                        )
                else:
                    rank_mean = _compute_ranking_loss(
                        model, batch, device, tokenizer, distractor_queue, pad_token_id,
                        scaler=None, telepathy=use_telepathy,
                    )
                # Step only if we actually ran ranking (did backward)
                if rank_mean is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    for group in optimizer.param_groups:
                        clip_grad_norm_(group["params"], GRAD_CLIP_NORM)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        
        # Track loss
        running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        num_batches += 1
        
        # Memory logging (every N batches)
        if (batch_idx + 1) % MEMORY_LOG_INTERVAL == 0:
            log_memory_metrics(
                step=batch_idx + 1,
                epoch=epoch,
                device=device
            )
        
        # Safety stop check (if enabled)
        if check_safety_stop(epoch=epoch, step=batch_idx + 1):
            print(f"\n⚠️  SAFETY STOP TRIGGERED: RSS exceeded {SAFETY_STOP_RSS_GB} GB")
            print(f"   Saving checkpoint before exit...")
            # Note: We'll save checkpoint in main training loop
            return running_loss / num_batches if num_batches > 0 else 0.0
        
        # Periodic cache clearing only when needed (skip on A100 - hurts throughput)
        if (batch_idx + 1) % 100 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
        
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0
        prog.set_postfix(
            loss=f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}",
            lr=f"{current_lr:.2e}",
        )

        # Push current batch targets to distractor queue (for ranking loss in future batches)
        if RANKING_LOSS_WEIGHT > 0 and distractor_queue is not None and tokenizer is not None:
            B = len(batch["input_ids"])
            for i in range(B):
                ids = batch["input_ids"][i]
                pl = batch["prompt_len"][i]
                if isinstance(pl, torch.Tensor):
                    pl = pl.item()
                if isinstance(ids, torch.Tensor):
                    target_ids = ids[pl:].tolist()
                else:
                    target_ids = list(ids)[pl:]
                if target_ids:
                    distractor_queue.append(target_ids)

    # Handle remaining gradients if batch count doesn't divide evenly
    if num_batches % GRADIENT_ACCUMULATION_STEPS != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        for group in optimizer.param_groups:
            clip_grad_norm_(group["params"], GRAD_CLIP_NORM)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate_generation(
    model: FMRIFlamingo,
    val_loader: DataLoader,
    tokenizer: Any,
    num_samples: int = 4,
    allowed_token_ids: Optional[set] = None,
) -> Dict[str, float]:
    """
    Run generation on a small subset of validation data and compute metrics.

    Args:
        model: FMRIFlamingo model
        val_loader: Validation DataLoader
        tokenizer: Tokenizer
        num_samples: Number of samples to generate
        allowed_token_ids: If set, restrict generation to these token IDs (e.g. English words only).

    Returns:
        Dict with metrics (WER, BLEU, etc.)
    """
    model.eval()
    metrics = {}
    
    # Collect samples
    samples = []
    count = 0
    
    # We need to reconstruct the original batch structure for generation
    # The val_loader yields collated batches where fields are lists
    # We'll take the first batch and extract num_samples
    
    for batch in val_loader:
        batch_size = len(batch['subject_id'])
        
        # Unpack batch into individual samples
        for i in range(batch_size):
            if count >= num_samples:
                break
                
            # Reconstruct sample for generation
            # We need: time_series, input_ids (prompt only), prompt_len
            
            # Get full input_ids and prompt_len
            full_ids = batch['input_ids'][i]
            p_len = batch['prompt_len'][i]
            
            # Slice to prompt only for input
            prompt_ids = full_ids[:p_len]
            
            # Get target (answer) for evaluation
            # Answer is everything after prompt
            answer_ids = full_ids[p_len:]
            # Remove EOS token if present at the end
            if answer_ids[-1] == tokenizer.eos_token_id or answer_ids[-1] == tokenizer.convert_tokens_to_ids("<|endofchunk|>"):
                answer_ids = answer_ids[:-1]
                
            target_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            
            sample = {
                'input': {
                    'input_ids': prompt_ids,
                    'prompt_len': p_len,
                    'time_series': batch['time_series'][i]
                },
                'target': target_text
            }
            samples.append(sample)
            count += 1
            
        if count >= num_samples:
            break
            
    if not samples:
        return {}
        
    # Generate
    print(f"   Generating text for {len(samples)} validation samples...")
    predictions = []
    references = []
    
    # Load BLEU metric if available
    bleu_metric = None
    if HAS_EVALUATE:
        try:
            bleu_metric = load_metric("bleu")
        except Exception as e:
            print(f"   ⚠️  Failed to load BLEU metric: {e}")
    
    with torch.no_grad():
        for sample in samples:
            # Prepare batch for generate (list of 1 dict)
            gen_batch = [sample['input']]
            
            try:
                # Generate (optionally restrict to English word tokens to avoid code/special tokens)
                gen_kwargs = dict(
                    max_new_tokens=1,
                    num_beams=3,
                    do_sample=False,
                    temperature=1.0,
                    no_repeat_ngram_size=2,
                )
                if allowed_token_ids is not None:
                    gen_kwargs["prefix_allowed_tokens_fn"] = (
                        lambda batch_id, input_ids: allowed_token_ids
                    )
                outputs = model.generate(gen_batch, **gen_kwargs)
                
                pred_text = outputs[0].strip()
                ref_text = sample['target']
                
                predictions.append(pred_text)
                references.append(ref_text)
                
                # Print sample for visual inspection
                print(f"   📝 Sample {len(predictions)}:")
                print(f"      Ref:  '{ref_text}'")
                print(f"      Pred: '{pred_text}'")
                    
            except Exception as e:
                print(f"   ⚠️  Generation failed for sample: {e}")
                
    # Compute metrics
    if predictions:
        # WER
        try:
            # Handle empty strings to avoid WER errors
            clean_preds = [p.strip().upper() if p.strip() else " " for p in predictions]
            clean_refs = [r.strip().upper() if r.strip() else " " for r in references]
            wer_score = wer(clean_refs, clean_preds)
            metrics["WER"] = wer_score
        except Exception as e:
            print(f"   ⚠️  WER calculation failed: {e}")
            metrics["WER"] = 1.0 # Worst case
            
        # BLEU
        if bleu_metric:
            try:
                # BLEU expects list of lists for references
                bleu_refs = [[r] for r in references]
                # Tokenize by space for simple BLEU
                bleu_preds_tok = [p.split() for p in predictions]
                bleu_refs_tok = [[r.split()] for r in references]
                
                # Use evaluate's compute
                results = bleu_metric.compute(predictions=predictions, references=bleu_refs)
                metrics["BLEU"] = results["bleu"]
            except Exception as e:
                print(f"   ⚠️  BLEU calculation failed: {e}")
                metrics["BLEU"] = 0.0
    
    return metrics


def validate(
    model: FMRIFlamingo,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    tokenizer: Any = None, # Added tokenizer
) -> float:
    """Validate model."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        prog = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]",
            leave=False,
        )
        
        for batch in prog:
            loss = model.compute_loss(batch)
            running_loss += loss.item()
            num_batches += 1
            
            prog.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    try:
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
    except OverflowError:
        perplexity = float("inf")
    print(f"   Val Perplexity: {perplexity:.2f}")

    # --- Zero-fMRI ablation: re-run val with zeroed fMRI to check if brain signal matters ---
    zero_running_loss = 0.0
    zero_num_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            # Zero out fMRI data
            if isinstance(batch, dict) and "time_series" in batch:
                batch["time_series"] = [torch.zeros_like(ts) for ts in batch["time_series"]]
            loss_zero = model.compute_loss(batch)
            zero_running_loss += loss_zero.item()
            zero_num_batches += 1
    zero_avg_loss = zero_running_loss / zero_num_batches if zero_num_batches > 0 else 0.0
    try:
        zero_perplexity = torch.exp(torch.tensor(zero_avg_loss)).item()
    except OverflowError:
        zero_perplexity = float("inf")
    print(f"   Zero-fMRI Val Loss: {zero_avg_loss:.4f} | Perplexity: {zero_perplexity:.2f}")
    print(f"   fMRI contribution: {zero_avg_loss - avg_loss:.4f} loss difference ({zero_perplexity - perplexity:.2f} perplexity)")

    # Run generation validation if tokenizer is provided
    if tokenizer is not None:
        allowed_token_ids = None
        if RESTRICT_GENERATION_TO_WORD_VOCAB:
            try:
                from src.utils.word_vocab import get_english_word_token_ids
                allowed_token_ids = get_english_word_token_ids(tokenizer)
            except Exception as e:
                print(f"   ⚠️  Word vocab restriction skipped: {e}")
        gen_metrics = validate_generation(
            model, val_loader, tokenizer, allowed_token_ids=allowed_token_ids
        )
        if gen_metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in gen_metrics.items()])
            print(f"   Val Generation: {metrics_str}")
    
    return avg_loss


def train(
    resume_from: Optional[Path] = None,
    device: Optional[torch.device] = None,
    a100: bool = False,
):
    """Main training loop."""
    global BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, USE_GRADIENT_CHECKPOINTING, GRADIENT_CHECKPOINTING, DATALOADER_NUM_WORKERS
    # A100 80GB: maximize throughput (larger batch, no grad checkpointing, faster data loading)
    if a100:
        BATCH_SIZE = 8
        GRADIENT_ACCUMULATION_STEPS = 2
        USE_GRADIENT_CHECKPOINTING = False
        GRADIENT_CHECKPOINTING = False
        DATALOADER_NUM_WORKERS = 8  # More workers to keep GPU fed (data loading is the bottleneck)
        if device is None:
            device = get_device()
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print("🚀 A100 80GB mode: batch_size=8, grad_accum=2, no checkpointing, 8 workers, TF32 enabled")

    # Setup device
    if device is None:
        device = get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"🖥️  Using device: {device}")
    print_memory_stats(device)
    pin_memory = a100 and device.type == "cuda"
    num_workers = DATALOADER_NUM_WORKERS
    
    # Create datasets
    print("\n📦 Loading datasets...")
    
    # Initialize tokenizer for pre-tokenization in workers
    print("   Loading tokenizer for data workers...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_ID,
        local_files_only=False,
        trust_remote_code=True,
    )
    # Add special tokens to match FMRIFlamingo
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = "<PAD>"
    
    # Get story splits to prevent leakage
    from src.datasets.huth_fmri_dataset import create_splits
    train_stories, val_stories, test_stories = create_splits()
    print(f"   Stories split: {len(train_stories)} train, {len(val_stories)} val, {len(test_stories)} test")
    
    train_dataset = HuthFMRIDataset(
        split="train",
        story_names=train_stories,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        tokenizer=tokenizer,
    )
    val_dataset = HuthFMRIDataset(
        split="validation",
        story_names=val_stories,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        tokenizer=tokenizer,
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        raise ValueError("No training samples found! Check data paths and TextGrid files.")
    if len(val_dataset) == 0:
        print("⚠️  Warning: No validation samples found!")
    
    # Custom collate: time_series is list of (voxels, TRs) tensors; don't stack (variable voxel counts)
    def collate_fn(batch):
        """Collate batch; time_series stays as list of tensors for pad_fmri_data."""
        collated = {}
        for key in batch[0].keys():
            if key in ["time_series", "input_ids"]:
                # Keep as list of tensors (variable length)
                collated[key] = [item[key] for item in batch]
            else:
                # For other fields, use default collation (list of values)
                collated[key] = [item[key] for item in batch]
        return collated
    
    # Create data loaders (pin_memory + workers when --a100 for A100 80GB)
    prefetch = 8 if num_workers > 0 else None
    
    # Ensure validation batch size is 1 for generation validation to work correctly
    # (Generation logic assumes batch_size=1 or needs complex padding handling)
    val_batch_size = 1
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size, # Force batch size 1 for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
    )
    
    # Create model
    print("\n🤖 Creating model...")
    try:
        model = FMRIFlamingo(
            device=device,
            llm_id=LLM_ID,
            num_rois=NUM_ROIS,
            cross_attn_every_n_layers=CROSS_ATTN_EVERY_N_LAYERS,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
        )
        print(f"✅ Model created successfully")
        if GRADIENT_CHECKPOINTING:
            print(f"   ✅ Gradient checkpointing enabled (memory efficient)")
        print_memory_stats(device)  # Print again after model is loaded
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            print(f"\n❌ HuggingFace authentication required!")
            print(f"   To fix this:")
            print(f"   1. Run: huggingface-cli login")
            print(f"   2. Or request access to {LLM_ID} on HuggingFace")
            print(f"   3. Then run this script again")
            sys.exit(1)
        else:
            raise
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model)
    num_training_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = create_scheduler(optimizer, num_training_steps)
    
    # Mixed precision scaler
    scaler = None
    if USE_MIXED_PRECISION and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print(f"✅ Mixed precision training enabled ({MIXED_PRECISION_DTYPE.upper()})")
    
    # Load checkpoint if resuming
    start_epoch = 1
    best_val_loss = float("inf")
    loss_history = load_loss_history()
    
    if resume_from is not None:
        resume_info = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_epoch = resume_info["start_epoch"]
        best_val_loss = resume_info["best_val_loss"]
        print(f"📊 Previous loss history: {len(loss_history)} entries")
        
        # Override optimizer parameters with current config values
        # This ensures that changes to WEIGHT_DECAY in config.py take effect even when resuming
        print(f"🔄 Overriding optimizer parameters from config:")
        print(f"   Weight Decay: {WEIGHT_DECAY} (was likely different in checkpoint)")
        for group in optimizer.param_groups:
            group['weight_decay'] = WEIGHT_DECAY
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print(f"   Total epochs: {NUM_EPOCHS}")
    print(f"   Starting from epoch: {start_epoch}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"   Early stopping patience: {EARLY_STOP_PATIENCE}")
    print()
    
    epochs_no_improve = 0
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        # Train
        distractor_queue = deque(maxlen=500) if RANKING_LOSS_WEIGHT > 0 else None
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch,
            scaler=scaler, tokenizer=tokenizer, distractor_queue=distractor_queue,
        )
        
        # Validate
        val_loss = validate(model, val_loader, device, epoch, tokenizer=tokenizer)
        
        # Release H5 file handle buffers accumulated during this epoch
        train_dataset.close_handles()
        val_dataset.close_handles()

        # Log
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print(f"   Train loss: {train_loss:.4f}")
        print(f"   Val loss:   {val_loss:.4f}")
        print_memory_stats(device)
        
        # Save loss history
        loss_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "timestamp": datetime.now().isoformat(),
        })
        save_loss_history(loss_history)
        
        # Check for improvement
        is_best = val_loss < best_val_loss - 1e-4  # Small threshold to avoid noise
        
        if is_best:
            best_val_loss = val_loss
            epochs_no_improve = 0
            print(f"   ✅ New best validation loss!")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
        
        # Save checkpoint
        if epoch % SAVE_EVERY_N_EPOCHS == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, train_loss, val_loss,
                CHECKPOINT_DIR, is_best=is_best
            )
        
        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epochs_no_improve} epochs without improvement")
            print(f"   Best validation loss: {best_val_loss:.4f} at epoch {epoch - epochs_no_improve}")
            break
    
    print(f"\n✅ Training completed!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Total epochs: {epoch}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train FMRIFlamingo model")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--a100",
        action="store_true",
        help="Optimize for A100 80GB: batch=8, grad_accum=2, no grad checkpointing, 4 workers, TF32, pin_memory",
    )
    
    args = parser.parse_args()
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    resume_from = args.resume
    
    train(resume_from=resume_from, device=device, a100=args.a100)


if __name__ == "__main__":
    main()
