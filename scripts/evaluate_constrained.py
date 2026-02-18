#!/usr/bin/env python3
"""
Evaluate FMRIFlamingo predictions with CONSTRAINED generation (first word only).

This script:
1. Loads the trained FMRIFlamingo model.
2. Generates predictions for the test story.
3. STRICTLY truncates predictions to the first word (or first whitespace).
4. Evaluates using Huth metrics.
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

# Import Huth evaluation utils (from decoding/ folder)
DECODING_DIR = FMRI_FLAMINGO_DIR / "decoding"
if str(DECODING_DIR) not in sys.path:
    sys.path.insert(0, str(DECODING_DIR))

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def generate_predictions(
    checkpoint_path: Path,
    subject_id: str,
    story_name: str,
    output_path: Path,
    device: torch.device,
):
    """Generate predictions and save as .npz."""
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
    
    # Load dataset for this story
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
    
    # Generate
    print("🚀 Generating predictions (CONSTRAINED: First word only)...")
    generated_words = []
    generated_times = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # Create a clean batch for generation
            batch_size = len(batch['subject_id'])
            input_ids_list = batch['input_ids']
            prompt_lens = batch['prompt_len']
            
            clean_batch = []
            for i in range(batch_size):
                full_ids = input_ids_list[i]
                p_len = prompt_lens[i]
                prompt_ids = full_ids[:p_len]
                
                clean_batch.append({
                    'input_ids': prompt_ids,
                    'prompt_len': p_len,
                    'time_series': batch['time_series'][i]
                })
            
            # Generate
            outputs = model.generate(
                clean_batch, 
                max_new_tokens=10, # Generate a few tokens to be safe
                do_sample=False, # Greedy
                temperature=1.0,
                top_p=1.0,
            )
            
            # Post-processing: STRICTLY take first word
            for output_text in outputs:
                text = output_text.strip()
                # Split by whitespace and take first part
                first_word = text.split()[0] if text else ""
                # Clean up punctuation if needed (optional, but Huth usually compares raw words)
                # Lowercase to match reference
                generated_words.append(first_word.lower())
                
            # Print sample for inspection (first one in batch)
            if len(generated_words) % 10 == 0:
                print(f"   Sample Pred: '{generated_words[-1]}'")

            
            # Collect times
            tr_ends = batch['tr_end']
            for t_end in tr_ends:
                generated_times.append(float(t_end) * fmri_config.TR)
    
    # Save to .npz
    print(f"💾 Saving {len(generated_words)} predictions to {output_path}...")
    np.savez(
        output_path,
        words=np.array(generated_words),
        times=np.array(generated_times)
    )
    print("✅ Done.")

def evaluate_metrics(subject_id: str, story_name: str):
    """Run Huth evaluation script."""
    print("\n📊 Running Huth evaluation metrics...")
    
    cmd = [
        sys.executable,
        "decoding/evaluate_predictions.py",
        "--subject", subject_id,
        "--experiment", "fmri_flamingo_constrained", # Use different experiment name
        "--task", story_name,
        "--metrics", "WER", "BLEU", "METEOR", "BERT",
        "--null", "0"
    ]
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
        
    # Load and print scores
    score_path = FMRI_FLAMINGO_DIR / "scores" / subject_id / "fmri_flamingo_constrained" / f"{story_name}.npz"
    if score_path.exists():
        print(f"\n📈 Scores loaded from {score_path}")
        scores = np.load(score_path, allow_pickle=True)
        story_scores = scores['story_scores'].item()
        
        print("\n🏆 Results (Constrained):")
        print("-" * 40)
        print(f"{'Metric':<10} | {'Score':<10}")
        print("-" * 40)
        
        for (ref, metric), score in story_scores.items():
            if isinstance(score, (np.ndarray, list)):
                score = np.mean(score)
            print(f"{metric:<10} | {score:.4f}")
        print("-" * 40)
    else:
        print(f"❌ Score file not found at {score_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=FMRI_FLAMINGO_DIR / "checkpoints" / "best_checkpoint.pt")
    parser.add_argument("--subject", type=str, default="UTS03")
    parser.add_argument("--story", type=str, default="avatar")
    args = parser.parse_args()
    
    # Check for leakage
    from src.datasets.huth_fmri_dataset import create_splits
    train_stories, _, _ = create_splits()
    if args.story in train_stories:
        print(f"\n⚠️  WARNING: Story '{args.story}' is in the TRAINING set! Evaluation will be biased (leakage).")
        print("   Consider using a held-out story like 'avatar'.\n")
    else:
        print(f"\n✅ Story '{args.story}' is held-out (not in training set).")
    
    device = get_device()
    
    # Output to separate experiment folder
    output_dir = FMRI_FLAMINGO_DIR / "results" / args.subject / "fmri_flamingo_constrained"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.story}.npz"
    
    generate_predictions(args.checkpoint, args.subject, args.story, output_path, device)
    evaluate_metrics(args.subject, args.story)

if __name__ == "__main__":
    main()
