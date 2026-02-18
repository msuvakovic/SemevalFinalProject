#!/usr/bin/env python3
"""
Masked-prompt RANKING evaluation for FMRIFlamingo.

Same as evaluate_ranking.py, but we MASK the last N tokens of the prompt
(replace with PAD, attention=0) so the model cannot use that text and must
rely on fMRI to pick the correct continuation. Use this to check if the
model learned to use fMRI when context is missing (train/eval alignment).
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None
    print("⚠️  bert-score not installed. BERTScore metric will be skipped.")

SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

from src.datasets.huth_fmri_dataset import HuthFMRIDataset
from src.models.fmri_flamingo import FMRIFlamingo
from transformers import AutoTokenizer

import importlib.util
config_path = FMRI_FLAMINGO_DIR / "config.py"
spec = importlib.util.spec_from_file_location("fmri_config", config_path)
fmri_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmri_config)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_ranking_masked(
    checkpoint_path: Path,
    subject_id: str,
    story_name: str,
    device: torch.device,
    num_distractors: int = 99,
    mask_last_n: int = 5,
    blind: bool = False,
):
    print(f"🤖 Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    def collate_fn(batch):
        return {k: [b[k] for b in batch] for k in batch[0].keys()}

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    all_targets = []
    for i in range(len(dataset)):
        sample = dataset.samples[i]
        if dataset.tokenizer:
            ans = sample["answer"]
            ids = dataset.tokenizer(ans, add_special_tokens=False)["input_ids"]
            if ids:
                all_targets.append(ids)
    print(f"   Collected {len(all_targets)} valid target sequences for distractors.")

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_samples = 0
    pred_texts = []
    ref_texts = []

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    mode = "MASKED+BLIND" if blind else "MASKED"
    print(f"🎭 Running {mode} Ranking Evaluation (last {mask_last_n} prompt tokens masked{'; fMRI ZEROD' if blind else ''}; 1 correct vs {num_distractors} distractors)...")

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids_full = batch["input_ids"][0]
            prompt_len = batch["prompt_len"][0]
            prompt_ids = input_ids_full[:prompt_len]
            target_ids = input_ids_full[prompt_len:]
            if len(target_ids) == 0:
                continue

            fmri_data = batch["time_series"][0].to(device)
            if blind:
                fmri_data = torch.zeros_like(fmri_data)

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

            correct_seq = torch.cat([prompt_ids.to(device), target_ids.to(device)])
            all_candidates = [correct_seq] + [
                torch.cat([prompt_ids.to(device), dist]) for dist in distractor_ids_list
            ]
            all_losses = []

            n_mask = min(mask_last_n, prompt_len)

            chunk_size = 10
            for i in range(0, len(all_candidates), chunk_size):
                chunk_candidates = all_candidates[i : i + chunk_size]
                current_batch_size = len(chunk_candidates)
                voxels, trs = fmri_data.shape
                fmri_flat = fmri_data.reshape(-1)
                vision_x_one = fmri_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                vision_x = vision_x_one.repeat(current_batch_size, 1, 1, 1)

                max_len = max(c.size(0) for c in chunk_candidates)
                padded_candidates = []
                attention_masks = []

                for c in chunk_candidates:
                    len_c = c.size(0)
                    c_masked = c.clone()
                    if n_mask > 0:
                        c_masked[prompt_len - n_mask : prompt_len] = pad_token_id
                    if len_c < max_len:
                        padding = torch.full(
                            (max_len - len_c,), pad_token_id, device=device, dtype=c.dtype
                        )
                        padded_c = torch.cat([c_masked, padding])
                        attn = torch.cat([
                            torch.ones(prompt_len - n_mask, device=device),
                            torch.zeros(n_mask, device=device),
                            torch.ones(len_c - prompt_len, device=device),
                            torch.zeros(max_len - len_c, device=device),
                        ])
                    else:
                        padded_c = c_masked
                        attn = torch.cat([
                            torch.ones(prompt_len - n_mask, device=device),
                            torch.zeros(n_mask, device=device),
                            torch.ones(len_c - prompt_len, device=device),
                        ])
                    padded_candidates.append(padded_c)
                    attention_masks.append(attn)

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
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_labels = shift_labels.view(-1)
                flat_loss = loss_fct(flat_logits, flat_labels)
                per_token_loss = flat_loss.view(shift_labels.shape)
                mask_ce = (shift_labels != -100).float()
                per_sample_loss = (per_token_loss * mask_ce).sum(dim=1) / mask_ce.sum(dim=1)
                all_losses.extend(per_sample_loss.tolist())
                del outputs, logits, vision_x, lang_x
                torch.cuda.empty_cache()

            correct_loss = all_losses[0]
            better_distractors = sum(1 for l in all_losses[1:] if l < correct_loss)
            rank = 1 + better_distractors
            if rank == 1:
                top1_correct += 1
            if rank <= 5:
                top5_correct += 1
            if rank <= 10:
                top10_correct += 1

            best_candidate_idx = np.argmin(all_losses)
            best_seq = all_candidates[best_candidate_idx]
            best_new_ids = best_seq[prompt_len:]
            pred_text = tokenizer.decode(best_new_ids, skip_special_tokens=True).strip()
            ref_text = tokenizer.decode(target_ids, skip_special_tokens=True).strip()
            pred_texts.append(pred_text)
            ref_texts.append(ref_text)
            total_samples += 1
            if total_samples % 10 == 0:
                print(f"   Step {total_samples}: Top-1: {top1_correct/total_samples:.2%} | Top-5: {top5_correct/total_samples:.2%} | Top-10: {top10_correct/total_samples:.2%}")

    print("\n🏆 Final " + mode + " Ranking Results:")
    print(f"Total Samples: {total_samples}")
    print(f"Top-1 Accuracy:  {top1_correct/total_samples:.2%}")
    print(f"Top-5 Accuracy:  {top5_correct/total_samples:.2%}")
    print(f"Top-10 Accuracy: {top10_correct/total_samples:.2%}")
    print(f"Chance Level (Top-1): {1/(num_distractors+1):.2%}")

    if bert_score is not None and len(pred_texts) > 0:
        print("\n📊 Calculating BERTScore on Top-1 predictions...")
        try:
            clean_preds = [p if p else "." for p in pred_texts]
            clean_refs = [r if r else "." for r in ref_texts]
            P, R, F1 = bert_score(clean_preds, clean_refs, lang="en", verbose=True)
            print(f"BERTScore F1: {F1.mean().item():.4f}")
            print(f"BERTScore Precision: {P.mean().item():.4f}")
            print(f"BERTScore Recall: {R.mean().item():.4f}")
        except Exception as e:
            print(f"❌ Error calculating BERTScore: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Ranking eval with last N prompt tokens masked (model must use fMRI)."
    )
    parser.add_argument("--checkpoint", type=Path, default=FMRI_FLAMINGO_DIR / "checkpoints" / "best_checkpoint.pt")
    parser.add_argument("--subject", type=str, default="UTS03")
    parser.add_argument("--story", type=str, default="wheretheressmoke")
    parser.add_argument("--distractors", type=int, default=99)
    parser.add_argument("--mask-last-n", type=int, default=5, help="Mask last N tokens of prompt (default 5)")
    parser.add_argument("--blind", action="store_true", help="Zero out fMRI (masked + blind: LM prior only with reduced context)")
    args = parser.parse_args()
    device = get_device()
    evaluate_ranking_masked(
        args.checkpoint,
        args.subject,
        args.story,
        device,
        args.distractors,
        mask_last_n=args.mask_last_n,
        blind=args.blind,
    )


if __name__ == "__main__":
    main()
