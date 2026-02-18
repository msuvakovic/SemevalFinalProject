#!/usr/bin/env python3
"""
Train FMRIFlamingo for Telepathy using ORPO (Odds Ratio Preference Optimization).

Telepathy = fMRI-only: no text context, just BOS + <image> + answer.
ORPO trains chosen (correct next word) > rejected (distractors) via preference loss.
Uses a distractor queue (and optional hard negatives) to force the model to use fMRI.

Usage:
    python scripts/train_telepathy_orpo.py [--resume CHECKPOINT] [--device cuda] [--a100]

Isolated from the main train.py; reads same config and dataset.
"""

import sys
import random
import argparse
import json
from pathlib import Path
from collections import deque
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

from src.datasets.huth_fmri_dataset import HuthFMRIDataset
from src.models.fmri_flamingo import FMRIFlamingo

import importlib.util
config_path = FMRI_FLAMINGO_DIR / "config.py"
spec = importlib.util.spec_from_file_location("fmri_config", config_path)
fmri_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmri_config)

# ---------------------------------------------------------------------------
# Config (from config.py, overridable)
# ---------------------------------------------------------------------------
BATCH_SIZE = getattr(fmri_config, "BATCH_SIZE", 2)
GRADIENT_ACCUMULATION_STEPS = getattr(fmri_config, "GRADIENT_ACCUMULATION_STEPS", 4)
NUM_EPOCHS = getattr(fmri_config, "NUM_EPOCHS", 20)
# ORPO: single-sample steps → high variance. Keep LR low; if avg loss creeps up over epoch, use 1e-5.
LR = getattr(fmri_config, "LR_ORPO", 2e-5)
WEIGHT_DECAY = getattr(fmri_config, "WEIGHT_DECAY", 1e-2)
GRAD_CLIP_NORM = getattr(fmri_config, "GRAD_CLIP_NORM", 1.0)
NUM_ORPO_REJECTS = getattr(fmri_config, "NUM_ORPO_REJECTS", 7)  # 1 chosen + 7 rejected per sample
ORPO_BETA = getattr(fmri_config, "ORPO_BETA", 0.1)  # temperature for sigmoid in ORPO
LLM_ID = getattr(fmri_config, "LLM_ID", "meta-llama/Llama-3.2-1B")
NUM_ROIS = getattr(fmri_config, "NUM_ROIS", 200)
CROSS_ATTN_EVERY_N_LAYERS = getattr(fmri_config, "CROSS_ATTN_EVERY_N_LAYERS", 1)
USE_GRADIENT_CHECKPOINTING = getattr(fmri_config, "USE_GRADIENT_CHECKPOINTING", True)
CHECKPOINT_DIR = getattr(fmri_config, "CHECKPOINT_DIR", FMRI_FLAMINGO_DIR / "checkpoints")
ORPO_CHECKPOINT_DIR = CHECKPOINT_DIR / "telepathy_orpo"
ORPO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ORPO_LOSS_HISTORY_PATH = ORPO_CHECKPOINT_DIR / "loss_history.json"
ORPO_METRICS_JSONL_PATH = ORPO_CHECKPOINT_DIR / "metrics.jsonl"  # per-step (optional)
RANDOM_SEED = getattr(fmri_config, "RANDOM_SEED", 42)
LOG_STEP_EVERY = 5  # print/write every N optimizer steps (with grad accum, steps are rarer so log more often)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_ID, trust_remote_code=True
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = "<PAD>"
    return tokenizer


def build_telepathy_prompt_and_answer_ids(
    tokenizer,
    answer_text: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (full_ids, answer_ids, prompt_len) for Telepathy: BOS + <image> + answer."""
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id
    media_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
    # Prompt = BOS + <image> (so model conditions on fMRI at one position)
    prompt_ids = [bos_id, media_id]
    answer_ids = tokenizer(
        answer_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()
    full_ids = prompt_ids + answer_ids
    prompt_len = len(prompt_ids)
    return (
        torch.tensor(full_ids, dtype=torch.long, device=device),
        torch.tensor(answer_ids, dtype=torch.long, device=device),
        prompt_len,
    )


def sequence_log_prob_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Sum of log probs of the answer tokens (positions after prompt). Causal: logits[i] predicts labels[i+1]."""
    # logits: (B, L, V), labels: (B, L); labels have -100 for prompt/pad
    shift_logits = logits[:, prompt_len - 1 : -1, :]  # (B, L_ans, V)
    shift_labels = labels[:, prompt_len:]  # (B, L_ans)
    mask = (shift_labels != -100)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    idx = shift_labels.clone()
    idx[idx == -100] = 0
    token_log_probs = log_probs.gather(dim=2, index=idx.unsqueeze(2)).squeeze(2)
    return (token_log_probs * mask.float()).sum() / mask.sum().float().clamp(min=1e-8)


def orpo_loss_chosen_vs_rejected(
    model: FMRIFlamingo,
    time_series: List,
    chosen_ids: torch.Tensor,
    rejected_ids_list: List[torch.Tensor],
    tokenizer,
    device: torch.device,
    pad_token_id: int,
    beta: float = ORPO_BETA,
) -> Tuple[torch.Tensor, float]:
    """ORPO loss and accuracy. Returns (loss, acc) where acc = fraction of rejections where chosen had higher log P."""
    prompt_len = 2  # BOS + <image>
    batch = {
        "time_series": time_series,
        "input_ids": [chosen_ids],
        "prompt_len": [prompt_len],
    }
    _, images, _, _ = model.pad_and_apply_batch(batch, include_labels=False)
    images = images.to(device)

    def log_prob_for_seq(seq_ids: torch.Tensor) -> torch.Tensor:
        seq = seq_ids.unsqueeze(0)
        labels = seq.clone()
        labels[:, :prompt_len] = -100
        vision_x = images
        lang_x = seq
        attention_mask = (seq != pad_token_id).long()
        outputs = model.model(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits
        return sequence_log_prob_from_logits(logits, labels, prompt_len)

    log_p_chosen = log_prob_for_seq(chosen_ids)
    losses = []
    correct = []
    for rej in rejected_ids_list:
        if rej.shape[0] < 1:
            continue
        log_p_rej = log_prob_for_seq(rej)
        # ORPO: -log σ(β * (log P(chosen) - log P(rejected))) = softplus(-β * (log_p_chosen - log_p_rej))
        log_odds = beta * (log_p_chosen - log_p_rej)
        losses.append(F.softplus(-log_odds))
        correct.append((log_p_chosen > log_p_rej).float().item())
    if not losses:
        return torch.tensor(0.0, device=device), 0.0
    acc = sum(correct) / len(correct) if correct else 0.0
    return torch.stack(losses).mean(), acc


def collate_telepathy(batch: List[Dict]) -> Dict[str, Any]:
    out = {}
    for key in batch[0].keys():
        if key == "time_series":
            out[key] = [b[key] for b in batch]
        else:
            out[key] = [b[key] for b in batch]
    return out


def train_epoch(
    model: FMRIFlamingo,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    distractor_queue: deque,
    device: torch.device,
    epoch: int,
) -> float:
    """
    One epoch: loop over batches, for each sample compute ORPO loss (chosen vs rejects)
    and run an optimizer step. Loss is computed and weights updated per sample (not per batch).
    Progress: tqdm bar shows running avg loss and step count; every LOG_STEP_EVERY steps
    we also print and append to metrics.jsonl.
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_steps = 0
    accumulated_loss = 0.0
    samples_since_step = 0
    pad_token_id = tokenizer.pad_token_id

    bos_id = tokenizer.bos_token_id or tokenizer.pad_token_id
    media_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=f"Epoch {epoch} [ORPO Telepathy]", leave=True, unit="batch")
    for batch_idx, batch in enumerate(pbar):
        time_series_list = batch["time_series"]
        input_ids_list = batch["input_ids"]
        prompt_lens = batch["prompt_len"]

        for i in range(len(time_series_list)):
            full_ids = input_ids_list[i]
            if isinstance(full_ids, torch.Tensor):
                full_ids = full_ids.to(device)
            else:
                full_ids = torch.tensor(full_ids, dtype=torch.long, device=device)
            pl = prompt_lens[i]
            if isinstance(pl, torch.Tensor):
                pl = pl.item()
            answer_part = full_ids[pl:].tolist()
            if len(answer_part) == 0:
                continue
            # Telepathy: prompt is only BOS + <image>, then answer
            chosen_ids = torch.tensor(
                [bos_id, media_id] + answer_part,
                dtype=torch.long,
                device=device,
            )

            # Sample rejections from queue (from other samples)
            k = min(NUM_ORPO_REJECTS, len(distractor_queue))
            if k == 0:
                continue
            rejected = random.sample(list(distractor_queue), k)
            rejected_tensors = [
                torch.tensor([bos_id, media_id] + r, dtype=torch.long, device=device)
                for r in rejected
                if r != answer_part and len(r) > 0
            ]
            if not rejected_tensors:
                continue

            # Same fMRI for chosen vs rejected; ORPO pushes log P(chosen) > log P(rejected).
            loss, step_acc = orpo_loss_chosen_vs_rejected(
                model=model,
                time_series=[time_series_list[i]],
                chosen_ids=chosen_ids,
                rejected_ids_list=rejected_tensors,
                tokenizer=tokenizer,
                device=device,
                pad_token_id=pad_token_id,
                beta=ORPO_BETA,
            )
            
            if loss.requires_grad:
                # Scale loss for gradient accumulation
                # Note: We divide by GRADIENT_ACCUMULATION_STEPS * BATCH_SIZE if we want to normalize per batch
                # But here we loop over samples. Let's normalize by accumulation steps.
                loss_scaled = loss / GRADIENT_ACCUMULATION_STEPS
                loss_scaled.backward()
                
                step_loss = loss.item()
                accumulated_loss += step_loss
                total_acc += step_acc
                samples_since_step += 1
                
                # Step optimizer if we've accumulated enough
                if samples_since_step >= GRADIENT_ACCUMULATION_STEPS:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Update stats only on step (smoother)
                    total_loss += accumulated_loss / samples_since_step # Avg loss over this accumulation
                    accumulated_loss = 0.0
                    samples_since_step = 0
                    n_steps += 1
                    
                    avg_so_far = total_loss / n_steps
                    avg_acc_so_far = total_acc / (n_steps * GRADIENT_ACCUMULATION_STEPS) # Approx
                    
                    pbar.set_postfix(loss=f"{avg_so_far:.4f}", acc=f"{step_acc:.2%}", steps=n_steps)
                    
                    # Log every N steps (same interval as loss)
                    if n_steps % LOG_STEP_EVERY == 0:
                        tqdm.write(f"  step {n_steps}  loss={step_loss:.4f}  avg_loss={avg_so_far:.4f}  acc={step_acc:.2%}  avg_acc={avg_acc_so_far:.2%}")
                        try:
                            with open(ORPO_METRICS_JSONL_PATH, "a") as f:
                                f.write(json.dumps({"epoch": epoch, "step": n_steps, "loss": step_loss, "avg_loss": avg_so_far, "acc": step_acc, "avg_acc": avg_acc_so_far}) + "\n")
                        except Exception:
                            pass

            distractor_queue.append(answer_part)
    
    # Final step if any gradients remaining
    if samples_since_step > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(n_steps, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Telepathy with ORPO")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--orpo-rejects", type=int, default=NUM_ORPO_REJECTS)
    parser.add_argument("--orpo-beta", type=float, default=ORPO_BETA)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap train samples (for debugging)")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = get_device() if args.device is None else torch.device(args.device)
    lr = args.lr
    print(f"Telepathy ORPO training on {device}  lr={lr}  (if avg loss creeps up over epoch, try --lr 1e-5)")

    tokenizer = get_tokenizer()
    train_dataset = HuthFMRIDataset(
        split="train",
        window_size=getattr(fmri_config, "DEFAULT_WINDOW_SIZE", 10),
        stride=getattr(fmri_config, "DEFAULT_STRIDE", 10),
        tokenizer=tokenizer,
        max_samples=args.max_samples,
    )
    if len(train_dataset) == 0:
        print("No training samples. Check data paths and TextGrids.")
        sys.exit(1)
    print(f"Train samples: {len(train_dataset)}")

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_telepathy,
    )

    model = FMRIFlamingo(
        device=device,
        llm_id=LLM_ID,
        num_rois=NUM_ROIS,
        cross_attn_every_n_layers=CROSS_ATTN_EVERY_N_LAYERS,
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    start_epoch = 1
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        print(f"Resumed from epoch {ckpt.get('epoch', 0)}")

    distractor_queue = deque(maxlen=500)
    # Prime queue with some answers from dataset
    for idx in range(min(100, len(train_dataset))):
        s = train_dataset[idx]
        if "input_ids" in s and "prompt_len" in s:
            ids = s["input_ids"]
            pl = s["prompt_len"]
            if hasattr(pl, "item"):
                pl = pl.item()
            distractor_queue.append(ids[pl:].tolist() if hasattr(ids, "tolist") else ids[pl:])

    epochs_range = range(start_epoch, args.epochs + 1)
    for epoch in tqdm(epochs_range, desc="Epochs", unit="epoch", position=0):
        train_loss = train_epoch(
            model, loader, optimizer, tokenizer, distractor_queue, device, epoch
        )
        tqdm.write(f"Epoch {epoch}  train_orpo_loss={train_loss:.4f}")

        # Append to loss history (per epoch) so you can plot or verify trends
        try:
            history = []
            if ORPO_LOSS_HISTORY_PATH.exists():
                with open(ORPO_LOSS_HISTORY_PATH) as f:
                    history = json.load(f)
            history.append({"epoch": epoch, "train_loss": train_loss})
            with open(ORPO_LOSS_HISTORY_PATH, "w") as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass

        if epoch % 1 == 0:
            path = ORPO_CHECKPOINT_DIR / f"telepathy_orpo_epoch_{epoch}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
            }, path)
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
