#!/usr/bin/env python3
"""
Evaluate FMRIFlamingo using RANKING (1-in-25).

Same as evaluate_ranking.py but with 24 distractors (Chance = 4%).
"""

import sys
import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

# Import project modules
from src.datasets.huth_fmri_dataset import HuthFMRIDataset
from src.models.fmri_flamingo import FMRIFlamingo
from transformers import AutoTokenizer

# Import config
import importlib.util
config_path = FMRI_FLAMINGO_DIR / "config.py"
spec = importlib.util.spec_from_file_location("fmri_config", config_path)
fmri_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmri_config)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def evaluate_ranking(
    checkpoint_path: Path,
    subject_id: str,
    story_name: str,
    device: torch.device,
    num_distractors: int = 24, # 1 correct + 24 distractors = 25 choices
):
    print(f"🤖 Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        fmri_config.LLM_ID,
        local_files_only=False,
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = "<PAD>"
    
    # Initialize model
    model = FMRIFlamingo(
        device=device,
        llm_id=fmri_config.LLM_ID,
        num_rois=fmri_config.NUM_ROIS,
        cross_attn_every_n_layers=fmri_config.CROSS_ATTN_EVERY_N_LAYERS,
        gradient_checkpointing=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    # Load dataset
    print(f"📦 Loading data for {subject_id}/{story_name}...")
    dataset = HuthFMRIDataset(
        split="test",
        subject_ids=[subject_id],
        story_names=[story_name],
        window_size=fmri_config.DEFAULT_WINDOW_SIZE,
        stride=fmri_config.DEFAULT_STRIDE,
        tokenizer=tokenizer,
    )
    
    if len(dataset) == 0:
        raise ValueError(f"No samples found for {subject_id}/{story_name}")
    
    # Custom collate
    def collate_fn(batch):
        collated = {}
        for key in batch[0].keys():
            if key in ["time_series", "input_ids"]:
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = [item[key] for item in batch]
        return collated
        
    loader = DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Get a pool of valid targets from the dataset itself (In-Story Distractors)
    all_targets = []
    print("   Collecting distractor pool from dataset...")
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        if dataset.tokenizer:
            ans = sample['answer']
            ids = dataset.tokenizer(ans, add_special_tokens=False)["input_ids"]
            if ids:
                all_targets.append(ids)
    
    print(f"   Collected {len(all_targets)} valid target sequences for distractors.")
    
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_samples = 0
    
    print(f"🚀 Running Ranking Evaluation (1 correct vs {num_distractors} distractors)...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # 1. Prepare Inputs
            input_ids_full = batch['input_ids'][0] # (seq_len)
            prompt_len = batch['prompt_len'][0]
            
            # Split prompt and target
            prompt_ids = input_ids_full[:prompt_len]
            target_ids = input_ids_full[prompt_len:]
            
            if len(target_ids) == 0:
                continue
                
            # 2. Generate Distractors (In-Story)
            distractor_ids_list = []
            attempts = 0
            while len(distractor_ids_list) < num_distractors and attempts < 1000:
                rand_idx = np.random.randint(0, len(all_targets))
                dist = torch.tensor(all_targets[rand_idx], device=device)
                if dist.shape == target_ids.shape and torch.equal(dist, target_ids.to(device)):
                    attempts += 1
                    continue
                distractor_ids_list.append(dist)
                
            while len(distractor_ids_list) < num_distractors:
                 rand_tokens = np.random.choice(list(range(1000, 30000)), size=len(target_ids))
                 distractor_ids_list.append(torch.tensor(rand_tokens, device=device))
            
            # Prepare lang_x
            correct_seq = torch.cat([prompt_ids.to(device), target_ids.to(device)])
            
            # 3. Construct Batch and Run Inference in Chunks
            fmri_data = batch['time_series'][0].to(device) # (voxels, TRs)
            all_candidates = [correct_seq] + [torch.cat([prompt_ids.to(device), dist]) for dist in distractor_ids_list]
            all_losses = []
            
            chunk_size = 10
            for i in range(0, len(all_candidates), chunk_size):
                chunk_candidates = all_candidates[i:i+chunk_size]
                current_batch_size = len(chunk_candidates)
                
                voxels, trs = fmri_data.shape
                fmri_flat = fmri_data.reshape(-1)
                vision_x_one = fmri_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                vision_x = vision_x_one.repeat(current_batch_size, 1, 1, 1)
                
                # Pad candidates
                max_len = max([c.size(0) for c in chunk_candidates])
                padded_candidates = []
                attention_masks = []
                
                for c in chunk_candidates:
                    len_c = c.size(0)
                    if len_c < max_len:
                        padding = torch.full((max_len - len_c,), tokenizer.pad_token_id, device=device, dtype=c.dtype)
                        padded_c = torch.cat([c, padding])
                        mask = torch.cat([torch.ones(len_c, device=device), torch.zeros(max_len - len_c, device=device)])
                    else:
                        padded_c = c
                        mask = torch.ones(len_c, device=device)
                    padded_candidates.append(padded_c)
                    attention_masks.append(mask)
                
                lang_x = torch.stack(padded_candidates)
                attention_mask = torch.stack(attention_masks)
                
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
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                flat_loss = loss_fct(flat_logits, flat_labels)
                
                per_token_loss = flat_loss.view(shift_labels.shape)
                mask = (shift_labels != -100).float()
                per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)
                
                all_losses.extend(per_sample_loss.tolist())
                
                del outputs, logits, vision_x, lang_x
                torch.cuda.empty_cache()
            
            # 5. Rank
            correct_loss = all_losses[0]
            better_distractors = sum(1 for l in all_losses[1:] if l < correct_loss)
            rank = 1 + better_distractors
            
            if rank == 1:
                top1_correct += 1
            if rank <= 5:
                top5_correct += 1
            if rank <= 10:
                top10_correct += 1
            
            total_samples += 1
            
            if total_samples % 10 == 0:
                print(f"   Step {total_samples}: Top-1: {top1_correct/total_samples:.2%} | Top-5: {top5_correct/total_samples:.2%}")

    print("\n🏆 Final Ranking Results (1-in-25):")
    print(f"Total Samples: {total_samples}")
    print(f"Top-1 Accuracy:  {top1_correct/total_samples:.2%}")
    print(f"Top-5 Accuracy:  {top5_correct/total_samples:.2%}")
    print(f"Chance Level (Top-1): {1/(num_distractors+1):.2%}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=FMRI_FLAMINGO_DIR / "checkpoints" / "best_checkpoint.pt")
    parser.add_argument("--subject", type=str, default="UTS03")
    parser.add_argument("--story", type=str, default="avatar")
    parser.add_argument("--distractors", type=int, default=24, help="Number of distractor words")
    args = parser.parse_args()
    
    device = get_device()
    evaluate_ranking(args.checkpoint, args.subject, args.story, device, args.distractors)

if __name__ == "__main__":
    main()
