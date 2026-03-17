#!/usr/bin/env python3
"""
Evaluate FMRIFlamingo predictions using the same metrics as the Ridge baseline.

This script:
1. Loads the trained FMRIFlamingo model (best_checkpoint.pt).
2. Generates predictions for the test story (UTS09).
3. Formats predictions into the .npz format expected by Huth's evaluation code.
4. Calls the Huth evaluation logic (WER, BLEU, METEOR, BERT) to get scores.
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

# We need to mock 'config' module for evaluate_predictions if it imports it
# But evaluate_predictions imports 'config' from local dir.
# We will just run the evaluation logic directly here or shell out to it.
# Let's shell out to it to ensure strict comparability.

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
    config_dict = checkpoint.get("config", {})
    
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
    
    # Read cross_attn setting from checkpoint config if available
    ckpt_cross_attn = config_dict.get("CROSS_ATTN_EVERY_N_LAYERS", fmri_config.CROSS_ATTN_EVERY_N_LAYERS)
    print(f"  cross_attn_every_n_layers: {ckpt_cross_attn} (from checkpoint)")

    # Initialize model
    model = FMRIFlamingo(
        device=device,
        llm_id=fmri_config.LLM_ID,
        num_rois=fmri_config.NUM_ROIS,
        cross_attn_every_n_layers=ckpt_cross_attn,
        gradient_checkpointing=False,
    )

    # Strip duplicate-prefix keys (checkpoint may have both "model.*" and "llm.*")
    state_dict = checkpoint["model_state_dict"]
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered, strict=False)
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
    
    # Custom collate (same as train.py)
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
        batch_size=1, # Batch size 1 to avoid padding issues during generation (right-padding vs left-padding)
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Generate
    print("🚀 Generating predictions...")
    generated_words = []
    generated_times = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # Generate text
            # We want to generate the answer.
            # The model sees: pre_prompt <image> post_prompt
            # We ask it to generate.
            
            # Note: The dataset provides 'input_ids' which includes the answer.
            # We need to strip the answer for generation if we want true generation.
            # However, FMRIFlamingo.generate() takes the full batch and handles masking internally?
            # Let's check FMRIFlamingo.generate logic.
            # It calls pad_and_apply_batch(include_labels=False).
            # pad_and_apply_batch constructs: pre <image> post answer <EOC>
            # Wait, if we provide 'answer', it includes it in input_ids.
            # We should probably clear the 'answer' field in the batch before passing to generate
            # if we want it to predict it.
            
            # Actually, pad_and_apply_batch constructs text from strings.
            # If we pass pre-tokenized input_ids, it uses those.
            # The dataset returns input_ids which HAVE the answer.
            # This is LEAKAGE if we just feed it to generate().
            
            # Fix: We must NOT use the pre-tokenized input_ids from the dataset for generation,
            # because they contain the ground truth answer.
            # We should rely on the raw text fields (pre_prompt, post_prompt) and empty answer.
            
            # Create a clean batch for generation
            gen_batch = []
            batch_size = len(batch['subject_id'])
            
            for i in range(batch_size):
                # Reconstruct item without answer
                item = {
                    'time_series': batch['time_series'][i],
                    'pre_prompt': dataset.samples[dataset.samples.index(dataset.samples[0])]['pre_prompt'], # Wait, we need correct index
                    # The batch doesn't have raw text if we used pre-tokenization in dataset?
                    # HuthFMRIDataset __getitem__ returns raw text fields ONLY if tokenizer is None.
                    # If tokenizer is set, it returns input_ids.
                    
                    # Problem: We need raw text to construct generation prompt without answer.
                    # Or we need to slice input_ids to remove answer.
                }
                
            # Let's look at HuthFMRIDataset again.
            # It returns input_ids AND prompt_len.
            # We can slice input_ids[:prompt_len] to get just the prompt!
            
            input_ids_list = batch['input_ids']
            prompt_lens = batch['prompt_len']
            
            # We need to manually construct the input for generate()
            # FMRIFlamingo.generate() calls pad_and_apply_batch.
            # If we pass a list of dicts with 'input_ids', it uses them.
            
            clean_batch = []
            for i in range(batch_size):
                full_ids = input_ids_list[i]
                p_len = prompt_lens[i]
                # Slice to just prompt (pre <image> post)
                prompt_ids = full_ids[:p_len]
                
                clean_batch.append({
                    'input_ids': prompt_ids,
                    'prompt_len': p_len,
                    'time_series': batch['time_series'][i]
                })
            
            # Debug: print decoded prompt
            if len(generated_words) == 0:
                print(f"DEBUG: Prompt IDs shape: {clean_batch[0]['input_ids'].shape}")
                print(f"DEBUG: Decoded prompt: {tokenizer.decode(clean_batch[0]['input_ids'])}")
            
            # Generate
            # max_new_tokens=1 (we just want the next word)
            outputs = model.generate(
                clean_batch, 
                max_new_tokens=5, # Allow a few tokens for a word
                do_sample=False, # Greedy decoding for reproducibility
                temperature=1.0,
                top_p=1.0,
            )
            
            # Outputs is list of strings
            # We need to extract the first word/meaningful content
            for output_text in outputs:
                # Output text contains the answer.
                # We need to clean it up (remove special tokens is done by batch_decode)
                # Just take the text.
                # Lowercase to match Huth's evaluation protocol (references are lowercased)
                generated_words.append(output_text.strip().lower())
            
            # Collect times (TR start time)
            # We use the middle of the window as the timestamp?
            # Huth eval expects a timestamp for each word.
            # Our samples are windows.
            # Let's use the end time of the window (prediction target).
            # tr_end is index. We need time.
            # TR = 2.0s.
            tr_ends = batch['tr_end']
            for t_end in tr_ends:
                generated_times.append(float(t_end) * fmri_config.TR)
    
    # Save to .npz
    # Huth format: 'words' (array of str), 'times' (array of float)
    print(f"💾 Saving {len(generated_words)} predictions to {output_path}...")
    np.savez(
        output_path,
        words=np.array(generated_words),
        times=np.array(generated_times)
    )

    # Free model from GPU before BERTScore needs it
    del model, checkpoint
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    print("✅ Done.")

def evaluate_per_window(pred_file: Path, show_preds: bool = False):
    """Per-window evaluation on direct generation output."""
    print("\n--- Per-Window Evaluation ---")
    data = np.load(pred_file, allow_pickle=True)
    preds = data["words"]
    refs = data["reference"]

    n = len(preds)
    assert len(refs) == n, f"Mismatch: {len(preds)} preds vs {len(refs)} refs"

    # Exact match
    exact = sum(1 for p, r in zip(preds, refs) if p == r)

    # Content words (skip "sp" = silence, common function words)
    function_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
                      "for", "of", "with", "is", "was", "are", "were", "be", "been",
                      "have", "has", "had", "do", "does", "did", "i", "you", "he",
                      "she", "it", "we", "they", "my", "his", "her", "its", "our",
                      "that", "this", "not", "so", "if", "up", "no", "just"}
    content_preds = [(p, r) for p, r in zip(preds, refs) if r not in function_words and r != "sp"]
    content_exact = sum(1 for p, r in content_preds if p == r)

    # Silence detection
    silence_refs = [(p, r) for p, r in zip(preds, refs) if r == "sp"]
    silence_correct = sum(1 for p, r in silence_refs if p == "sp")

    # Non-silence refs
    speech_refs = [(p, r) for p, r in zip(preds, refs) if r != "sp"]
    speech_exact = sum(1 for p, r in speech_refs if p == r)

    # Per-word WER (1 - exact_match_rate for single words)
    wer = 1.0 - exact / n

    # Unique predictions
    unique = len(set(preds))

    # BERTScore (per-word)
    try:
        from bert_score import score as bert_score
        valid = [(p if p else ".", r) for p, r in zip(preds, refs)]
        p_list = [v[0] for v in valid]
        r_list = [v[1] for v in valid]
        P, R, F1 = bert_score(p_list, r_list, lang="en", verbose=False,
                              device="cuda" if torch.cuda.is_available() else "cpu")
        bert_f1 = F1.mean().item()
        bert_available = True
    except Exception as e:
        print(f"  BERTScore skipped: {e}")
        bert_f1 = None
        bert_available = False

    # BLEU (corpus-level: treat each word as a 1-token sequence)
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        bleu_refs = [[r.split()] for r in refs]   # list of list of list of tokens
        bleu_hyps = [p.split() for p in preds]
        smoother = SmoothingFunction().method1
        bleu1 = corpus_bleu(bleu_refs, bleu_hyps, weights=(1, 0, 0, 0), smoothing_function=smoother)
        bleu4 = corpus_bleu(bleu_refs, bleu_hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
        bleu_available = True
    except Exception as e:
        print(f"  BLEU skipped: {e}")
        bleu1 = bleu4 = None
        bleu_available = False

    # METEOR (average over windows)
    try:
        import nltk
        try:
            nltk.data.find("wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        from nltk.translate.meteor_score import single_meteor_score
        meteor_scores = [single_meteor_score(r.split(), p.split()) for p, r in zip(preds, refs)]
        meteor = float(np.mean(meteor_scores))
        meteor_available = True
    except Exception as e:
        print(f"  METEOR skipped: {e}")
        meteor = None
        meteor_available = False

    # Semantic near-misses (BERTScore > 0.8 but not exact match)
    near_misses = []
    if bert_available:
        for i, (p, r) in enumerate(zip(preds, refs)):
            if p != r and r != "sp" and p != "sp" and F1[i].item() > 0.8:
                near_misses.append((i, r, p, F1[i].item()))

    # Print results
    print(f"\n  Total windows:           {n}")
    print(f"  Exact match (overall):   {exact}/{n} ({100*exact/n:.1f}%)")
    if len(speech_refs) > 0:
        print(f"  Exact match (speech):    {speech_exact}/{len(speech_refs)} ({100*speech_exact/len(speech_refs):.1f}%)")
    if len(content_preds) > 0:
        print(f"  Exact match (content):   {content_exact}/{len(content_preds)} ({100*content_exact/len(content_preds):.1f}%)")
    if len(silence_refs) > 0:
        print(f"  Silence detection:       {silence_correct}/{len(silence_refs)} ({100*silence_correct/len(silence_refs):.1f}%)")
    print(f"  Per-window WER:          {wer:.4f}")
    if bert_available:
        print(f"  BERTScore F1:            {bert_f1:.4f}")
    if bleu_available:
        print(f"  BLEU-1:                  {bleu1:.4f}")
        print(f"  BLEU-4:                  {bleu4:.4f}")
    if meteor_available:
        print(f"  METEOR:                  {meteor:.4f}")
    print(f"  Unique predictions:      {unique}/{n}")

    if near_misses:
        print(f"\n  Semantic near-misses (BERTScore > 0.8, not exact):")
        for idx, ref, pred, bs in near_misses[:15]:
            print(f"    win {idx:3d} | ref={ref:<15s} | pred={pred:<15s} | BERT={bs:.3f}")
        if len(near_misses) > 15:
            print(f"    ... and {len(near_misses)-15} more")

    print("-" * 40)

    if show_preds:
        print(f"\n{'WIN':>4}  {'REF':<20}  {'PRED':<20}  {'MATCH':<5}  {'BERT':>6}")
        print("-" * 60)
        for i, (p, r) in enumerate(zip(preds, refs)):
            match_str = "✓" if p == r else " "
            bert_str = f"{F1[i].item():.3f}" if bert_available else "    -"
            print(f"{i:>4}  {r:<20}  {p:<20}  {match_str:<5}  {bert_str:>6}")
        print("-" * 60)


def evaluate_metrics(subject_id: str, story_name: str):
    """Run Huth evaluation script."""
    print("\n📊 Running Huth evaluation metrics...")
    
    # We need to map our story name to Huth's task name
    # e.g. "wheretheressmoke" -> "wheretheressmoke"
    # Assuming they match.
    
    # Command:
    # python decoding/evaluate_predictions.py --subject UTS09 --experiment fmri_flamingo --task wheretheressmoke --metrics WER BLEU METEOR
    
    cmd = [
        sys.executable,
        "decoding/evaluate_predictions.py",
        "--subject", subject_id,
        "--experiment", "fmri_flamingo",
        "--task", story_name,
        "--metrics", "WER", "BLEU", "METEOR", "BERT",
        "--null", "0" # Disable null model generation for speed/simplicity first
    ]
    
    # Note: Huth's evaluate_predictions.py might be case-sensitive.
    # We should check if we need to lowercase predictions before saving.
    # Standard WER is case-sensitive.
    # Let's keep raw predictions for now to be rigorous.
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
        
    # Load and print scores
    score_path = FMRI_FLAMINGO_DIR / "scores" / subject_id / "fmri_flamingo" / f"{story_name}.npz"
    if score_path.exists():
        print(f"\n📈 Scores loaded from {score_path}")
        scores = np.load(score_path, allow_pickle=True)
        # keys: window_scores, window_zscores, story_scores, story_zscores
        # story_scores is a dict of (reference, metric) -> score
        # But np.savez saves dicts as object arrays wrapped in 0-d array
        
        story_scores = scores['story_scores'].item()
        
        print("\n🏆 Results:")
        print("-" * 40)
        print(f"{'Metric':<10} | {'Score':<10}")
        print("-" * 40)
        
        for (ref, metric), score in story_scores.items():
            # score might be a float or array
            if isinstance(score, (np.ndarray, list)):
                score = np.mean(score)
            print(f"{metric:<10} | {score:.4f}")
        print("-" * 40)
    else:
        print(f"❌ Score file not found at {score_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=FMRI_FLAMINGO_DIR / "checkpoints" / "best_checkpoint.pt")
    parser.add_argument("--subject", type=str, default="UTS03") # Use UTS03 for dev, UTS09 for test
    parser.add_argument("--story", type=str, default="avatar") # Use a held-out story (avatar is in test split)
    parser.add_argument("--per-window", type=Path, default=None,
                        help="Path to direct generation .npz for per-window eval (skips model loading)")
    parser.add_argument("--show-preds", action="store_true",
                        help="Print full side-by-side table of ref vs pred for each window (use with --per-window)")
    args = parser.parse_args()

    # Per-window only mode: skip model loading entirely
    if args.per_window:
        evaluate_per_window(args.per_window, show_preds=args.show_preds)
        return

    # Check for leakage
    from src.datasets.huth_fmri_dataset import create_splits
    train_stories, _, _ = create_splits()
    if args.story in train_stories:
        print(f"\n⚠️  WARNING: Story '{args.story}' is in the TRAINING set! Evaluation will be biased (leakage).")
        print("   Consider using a held-out story like 'avatar'.\n")
    else:
        print(f"\n✅ Story '{args.story}' is held-out (not in training set).")

    device = get_device()

    output_dir = FMRI_FLAMINGO_DIR / "results" / args.subject / "fmri_flamingo"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.story}.npz"

    generate_predictions(args.checkpoint, args.subject, args.story, output_path, device)

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    evaluate_metrics(args.subject, args.story)

if __name__ == "__main__":
    main()
