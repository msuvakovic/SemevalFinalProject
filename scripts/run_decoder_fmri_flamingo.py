#!/usr/bin/env python3
"""
Run Huth-style word-level beam search decoding, but score candidates with FMRIFlamingo
instead of the ridge encoding model.

Same protocol as decoding/run_decoder.py: same data (fMRI, word times), same beam search,
same Decoder + LanguageModel. Only the scoring step changes: we use P(candidate | fMRI)
from our model (negative CE loss) instead of P(fMRI | candidate) from the encoding model.

Requirements:
  - decoding/ module and its config (for resp paths, word_rate_model, word times, Decoder, LM).
  - Our checkpoint, tokenizer, and config (for FMRIFlamingo and voxel alignment).

Usage:
  python scripts/run_decoder_fmri_flamingo.py --checkpoint checkpoints/best_checkpoint.pt --subject UTS09 --task <story> [--experiment perceived_speech]
"""

import os
import sys
import argparse
import gc
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Project root
SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

# Decoding module (Huth pipeline) - same repo, different config
DECODING_DIR = FMRI_FLAMINGO_DIR / "decoding"
if DECODING_DIR.exists() and str(DECODING_DIR) not in sys.path:
    sys.path.insert(0, str(DECODING_DIR))

# Our config and model
import importlib.util
config_path = FMRI_FLAMINGO_DIR / "config.py"
spec = importlib.util.spec_from_file_location("fmri_config", config_path)
fmri_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmri_config)

LLM_ID = getattr(fmri_config, "LLM_ID", "meta-llama/Llama-3.2-1B")
NUM_ROIS = getattr(fmri_config, "NUM_ROIS", 200)
ROI_SELECTION_METHOD = getattr(fmri_config, "ROI_SELECTION_METHOD", "all_voxels")
MAX_PATCHES = getattr(fmri_config, "MAX_PATCHES", 120000)  # tokenizer max_rois; do not exceed
DEFAULT_WINDOW_SIZE = getattr(fmri_config, "DEFAULT_WINDOW_SIZE", 10)

# Persistent chunk size — survives across timesteps so OOM recovery sticks
_chunk_size_limit = 32  # Start conservative (was 64 but OOMs with GPT + Llama + fMRI on 16GB)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def precompute_roi_grouping(num_voxels, num_rois, method):
    """Pre-compute ROI mapping on CPU so we can group voxels before sending to GPU."""
    from src.models.fmri_tokenizer import FMRITokenizer
    dummy = FMRITokenizer(num_rois=num_rois, roi_selection_method=method)
    voxel_to_roi = dummy._create_roi_mapping(num_voxels, num_rois, method)
    return voxel_to_roi


def apply_roi_grouping_cpu(fmri_slice_T, voxel_to_roi, num_rois):
    """Group voxels into ROIs on CPU. Input: (n_vox, n_trs) numpy. Output: (num_rois, n_trs) numpy."""
    mapping = voxel_to_roi.numpy()
    n_vox, n_trs = fmri_slice_T.shape
    # Use np.bincount for fast accumulation (no Python loop over 2000 ROIs)
    roi_data = np.zeros((num_rois, n_trs), dtype=np.float64)
    roi_counts = np.bincount(mapping, minlength=num_rois).astype(np.float64)
    roi_counts = np.maximum(roi_counts, 1.0)  # avoid div by zero
    for t in range(n_trs):
        roi_data[:, t] = np.bincount(mapping, weights=fmri_slice_T[:, t], minlength=num_rois)
    roi_data /= roi_counts[:, None]
    return roi_data.astype(fmri_slice_T.dtype)


def score_all_hypotheses_batched(model, tokenizer, device, fmri_segment, hyp_candidates):
    """
    Score ALL candidate continuations across ALL hypotheses in one batched call.

    fmri_segment: (num_rois, n_trs) tensor on device (already ROI-grouped)
    hyp_candidates: list of (context_words, candidate_words_list) tuples, one per hypothesis

    Returns: list of score-lists, one per hypothesis (same order as hyp_candidates)
    """
    global _chunk_size_limit

    # Tokenize everything across all hypotheses
    all_input_ids = []
    all_prompt_lens = []
    hyp_sizes = []  # how many candidates per hypothesis, for splitting results later

    MAX_CONTEXT_WORDS = 50  # Cap context to last 50 words to keep step time flat (O(n²) attention)

    for context_words, candidate_words in hyp_candidates:
        # openai-gpt BPE tokens use "</w>" as word boundary markers (e.g. "the</w>").
        # Strip them before feeding to the Llama tokenizer, which treats "</w>" as literal text.
        clean_context = [w.replace("</w>", "") for w in context_words[-MAX_CONTEXT_WORDS:]]
        context_str = " ".join(clean_context) if clean_context else ""
        prompt_str = f"{context_str} <image>" if context_str else "<image>"
        prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        prompt_len = len(prompt_ids)

        for cw in candidate_words:
            clean_cw = cw.replace("</w>", "")
            text = f"{prompt_str} {clean_cw}"
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            all_input_ids.append(input_ids)
            all_prompt_lens.append(prompt_len)
        hyp_sizes.append(len(candidate_words))

    if not all_input_ids:
        return [[] for _ in hyp_candidates]

    # Pad to same length
    max_len = max(len(ids) for ids in all_input_ids)
    pad_id = tokenizer.pad_token_id
    padded_ids = []
    attention_masks = []
    for ids in all_input_ids:
        pad_len = max_len - len(ids)
        if pad_len > 0:
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)]))
            attention_masks.append(torch.cat([torch.ones(len(ids), dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)]))
        else:
            padded_ids.append(ids)
            attention_masks.append(torch.ones(len(ids), dtype=torch.bool))

    # Keep batches on CPU, only move chunks to GPU
    input_ids_batch = torch.stack(padded_ids)
    attention_mask_batch = torch.stack(attention_masks)

    # Build labels: mask prompt tokens and padding with -100
    labels_batch = input_ids_batch.clone()
    for i, pl in enumerate(all_prompt_lens):
        labels_batch[i, :pl] = -100
    labels_batch[~attention_mask_batch] = -100

    # Score in mini-batches
    B = len(all_input_ids)
    voxels, trs = fmri_segment.shape
    fmri_single = fmri_segment.reshape(1, 1, 1, voxels * trs)

    all_scores = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    _layers = list(model.model.lang_encoder._get_decoder_layers())

    with torch.no_grad(), torch.amp.autocast('cuda'):
        # Encode fMRI ONCE — FMRITokenizer + Perceiver are the expensive parts
        # and the fMRI input is identical for all candidates at this timestep.
        model.model._encode_vision_x(fmri_single)
        cached_vis_x = _layers[0].vis_x.clone()  # (1, T, n, D)

        start = 0
        while start < B:
            end = min(start + _chunk_size_limit, B)
            chunk_size = end - start

            # Expand cached vision features to match text batch size (no copy)
            expanded_vis = cached_vis_x.expand(chunk_size, *cached_vis_x.shape[1:])
            for layer in _layers:
                layer.condition_vis_x(expanded_vis)

            ids_chunk = input_ids_batch[start:end].to(device)
            mask_chunk = attention_mask_batch[start:end].to(device)
            labels_chunk = labels_batch[start:end].to(device)

            try:
                output = model.model(
                    vision_x=None,
                    lang_x=ids_chunk,
                    attention_mask=mask_chunk,
                    labels=labels_chunk,
                    use_cached_vision_x=True,
                    clear_conditioned_layers=False,
                )
            except torch.cuda.OutOfMemoryError:
                del ids_chunk, mask_chunk, labels_chunk
                torch.cuda.empty_cache()
                _chunk_size_limit = max(1, _chunk_size_limit // 2)
                tqdm.write(f"⚠️  OOM — reducing chunk size to {_chunk_size_limit}")
                continue

            logits = output.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels_chunk[:, 1:].contiguous()
            per_token_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = per_token_loss.view(chunk_size, -1)
            valid_mask = (shift_labels != -100).float()
            per_candidate_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            all_scores.extend((-per_candidate_loss).cpu().tolist())
            del output, logits, shift_logits, per_token_loss
            start = end

    model.model.lang_encoder.clear_conditioned_layers()
    del cached_vis_x

    # Split flat scores back into per-hypothesis lists
    results = []
    offset = 0
    for size in hyp_sizes:
        results.append(all_scores[offset:offset + size])
        offset += size
    return results


def direct_generate(model, tokenizer, device, fmri_segment, context_words, max_context=50):
    """
    Direct generation: let the LLM generate the next word conditioned on fMRI + context.
    No beam search — just greedy/beam generation from the model itself.

    fmri_segment: (NUM_ROIS, n_trs) tensor on device (already ROI-grouped)
    context_words: list of previous words (strings)
    Returns: predicted word (string)
    """
    # Build prompt matching training format: "context words <image> Predict the next word:"
    clean_context = [w.replace("</w>", "") for w in context_words[-max_context:]]
    context_str = " ".join(clean_context) if clean_context else ""

    # Build the batch in the same format as training
    voxels, trs = fmri_segment.shape
    fmri_2d = fmri_segment  # (NUM_ROIS, window_size) — already on device

    # Construct prompt the same way HuthFMRIDataset does
    pre_prompt = context_str
    post_prompt = "Predict the next word:"

    # Tokenize prompt (no answer — model generates it)
    prompt_text = f"{pre_prompt} <image> {post_prompt}" if pre_prompt else f"<image> {post_prompt}"
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    batch = [{
        'input_ids': prompt_ids,
        'prompt_len': len(prompt_ids),
        'time_series': fmri_2d.cpu().float(),  # autocast handles fp16 conversion
    }]

    with torch.no_grad(), torch.amp.autocast('cuda'):
        outputs = model.generate(
            batch,
            max_new_tokens=1,
            num_beams=3,
            do_sample=False,
            temperature=1.0,
            no_repeat_ngram_size=2,
        )
    return outputs[0].strip() if outputs else ""


def main():
    parser = argparse.ArgumentParser(description="Huth-style decoding with FMRIFlamingo scoring.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FMRIFlamingo checkpoint")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g. UTS09)")
    parser.add_argument("--task", type=str, required=True, help="Story/task name (e.g. adollshouse)")
    parser.add_argument("--experiment", type=str, default="perceived_speech", help="Experiment type for word times")
    parser.add_argument("--ignore-overlap", action="store_true", help="Allow decoding on training stories")
    parser.add_argument("--beam-width", type=int, default=200, help="Beam width (default 200)")
    parser.add_argument("--extensions", type=int, default=5, help="Extensions per hypothesis")
    parser.add_argument("--direct", action="store_true", help="Skip beam search, use direct LLM generation per timestep")
    args = parser.parse_args()

    device = get_device()

    # Load our model and tokenizer
    from src.models.fmri_flamingo import FMRIFlamingo
    from transformers import AutoTokenizer

    print(f"Loading FMRIFlamingo from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = "<PAD>"

    # Read cross_attn setting from checkpoint config if available (avoids mismatch when config.py changes)
    ckpt_cross_attn = ckpt.get("config", {}).get("CROSS_ATTN_EVERY_N_LAYERS",
                      getattr(fmri_config, "CROSS_ATTN_EVERY_N_LAYERS", 4))
    print(f"  cross_attn_every_n_layers: {ckpt_cross_attn} (from checkpoint)")

    model = FMRIFlamingo(
        device=device,
        llm_id=LLM_ID,
        num_rois=NUM_ROIS,
        cross_attn_every_n_layers=ckpt_cross_attn,
        gradient_checkpointing=False,
    )

    # Strip duplicate-prefix keys (checkpoint may have both "model.*" and "llm.*" for same weights)
    state_dict = ckpt["model_state_dict"]
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered.keys())
    if missing:
        print(f"  ⚠️  {len(missing)} keys not found in checkpoint (will use random init)")
    model.load_state_dict(filtered, strict=False)
    del ckpt
    model.eval()
    model.half()  # fp16 for ~2x speedup
    model.to(device)

    # Load decoding config and data (same as run_decoder.py)
    try:
        import config as dec_config
    except ImportError:
        print("decoding/config.py not found. Set PYTHONPATH to include decoding/ or run from repo root.")
        sys.exit(1)

    # --- Train/test overlap guard ---
    try:
        from splits import load_split
        train_stories, _, _ = load_split(args.subject, getattr(dec_config, "SPLIT_FILE", None))
        if args.task in train_stories and not args.ignore_overlap:
            print(f"ERROR: '{args.task}' is a training story for {args.subject}. "
                  f"Decoding on training data produces inflated metrics. "
                  f"Use --ignore-overlap to force.")
            sys.exit(1)
        elif args.task in train_stories:
            print(f"WARNING: '{args.task}' is a training story — results will be optimistic.")
    except Exception as e:
        print(f"WARNING: Could not verify train/test split ({e}). Proceeding without check.")

    resp_path = None
    for p in [
        os.path.join(dec_config.DATA_TEST_DIR, "preprocessed_data", args.subject, args.task + ".hf5"),
        os.path.join(dec_config.DATA_TEST_DIR, "test_response", args.subject, args.experiment, args.task + ".hf5"),
    ]:
        if os.path.exists(p):
            resp_path = p
            break
    if resp_path is None:
        print(f"fMRI not found for {args.subject}/{args.task}. Tried DATA_TEST_DIR/preprocessed_data and test_response.")
        sys.exit(1)

    import h5py
    with h5py.File(resp_path, "r") as hf:
        resp = np.nan_to_num(hf["data"][:])
    # resp: (n_trs, n_voxels)

    # ROI grouping on CPU: match training by grouping all voxels into ROIs,
    # but do it on CPU to avoid GPU OOM (81k voxels on GPU + GPT + Llama = too much).
    # The model receives pre-grouped (NUM_ROIS, TRs) data — same as training output.
    n_vox = resp.shape[1]
    if ROI_SELECTION_METHOD != "all_voxels":
        voxel_to_roi = precompute_roi_grouping(n_vox, NUM_ROIS, ROI_SELECTION_METHOD)
        print(f"Pre-computed ROI mapping: {n_vox} voxels → {NUM_ROIS} ROIs via {ROI_SELECTION_METHOD} (CPU)")
    else:
        voxel_to_roi = None
        print(f"Using all {n_vox} voxels (no ROI grouping)")

    # ── Direct generation mode (no beam search) ──────────────────────────
    if args.direct:
        print("Direct generation mode — skipping beam search, using model.generate()")
        from src.datasets.huth_fmri_dataset import (
            load_textgrid, extract_words_from_textgrid,
            align_trs_to_words, TRFile,
        )
        HEMODYNAMIC_DELAY = getattr(fmri_config, "HEMODYNAMIC_DELAY", 4.0)
        TR = getattr(fmri_config, "TR", 2.0)
        TRIM_TRS = getattr(fmri_config, "TRIM_TRS", 5)

        total_trs = resp.shape[0]
        effective_start = TRIM_TRS
        effective_end = total_trs - TRIM_TRS
        num_trs = effective_end - effective_start

        # Load word alignments from TextGrid (same as training)
        textgrid = load_textgrid(args.task)
        words = extract_words_from_textgrid(textgrid) if textgrid else []
        if not words:
            print(f"No words found in TextGrid for {args.task}")
            sys.exit(1)

        tr_file = TRFile(tr=TR)
        tr_file.simulate(num_trs)
        tr_times = tr_file.get_reltriggertimes()
        alignments = align_trs_to_words(tr_times, words, HEMODYNAMIC_DELAY)

        # Slide windows exactly like training
        window_size = DEFAULT_WINDOW_SIZE
        stride = getattr(fmri_config, "DEFAULT_STRIDE", 5)
        # Use stride=1 for dense evaluation (every 2s); training stride is coarser
        stride = 1  # Every 2s for maximum evaluation density
        context_words_setting = 5  # matches HuthFMRIDataset default

        predicted_words = []
        predicted_times = []
        reference_words = []

        n_windows = (num_trs - window_size) // stride + 1
        for wi in tqdm(range(n_windows), desc="Direct generation"):
            start_idx = wi * stride
            end_idx = start_idx + window_size
            tr_start_abs = effective_start + start_idx
            tr_end_abs = effective_start + end_idx

            # Get words aligned to this window
            window_alignments = alignments[start_idx:end_idx]
            all_words_in_window = []
            for align in window_alignments:
                all_words_in_window.extend(align['words'])

            if not all_words_in_window:
                continue

            # Context = all words except last, target = last word (same as training)
            if len(all_words_in_window) > context_words_setting:
                context = all_words_in_window[:-1][-context_words_setting:]
            elif len(all_words_in_window) > 1:
                context = all_words_in_window[:-1]
            else:
                context = []
            target = all_words_in_window[-1]

            # Skip silence and TextGrid artifact tokens (consistent with training filter)
            if target == "sp" or target.startswith("{"):
                continue

            # Load fMRI window
            fmri_slice = resp[tr_start_abs:tr_end_abs, :]  # (window_size, n_vox)
            fmri_T = fmri_slice.T  # (n_vox, window_size)
            if voxel_to_roi is not None:
                fmri_T = apply_roi_grouping_cpu(fmri_T, voxel_to_roi, NUM_ROIS)
            fmri_segment = torch.from_numpy(fmri_T).half().to(device)

            pred = direct_generate(model, tokenizer, device, fmri_segment, context)

            predicted_words.append(pred.lower())
            reference_words.append(target.lower())
            predicted_times.append(float(tr_end_abs) * TR)

            del fmri_segment
            if wi % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                if wi > 0:
                    # Show progress samples
                    tqdm.write(f"  [{wi}/{n_windows}] ref='{target}' pred='{pred}'")

        # Save results
        save_dir = Path(dec_config.RESULT_DIR) / args.subject / args.experiment
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{args.task}_fmri_flamingo_direct.npz"
        np.savez(out_path,
                 words=np.array(predicted_words),
                 times=np.array(predicted_times),
                 reference=np.array(reference_words))
        print(f"\nSaved {len(predicted_words)} predictions to {out_path}")

        # Quick inline metrics
        from collections import Counter
        exact = sum(1 for p, r in zip(predicted_words, reference_words) if p == r)
        unique = len(set(predicted_words))
        print(f"Exact match: {exact}/{len(predicted_words)} ({100*exact/len(predicted_words):.1f}%)")
        print(f"Unique predictions: {unique}/{len(predicted_words)}")
        print(f"\nMost common predictions:")
        for w, c in Counter(predicted_words).most_common(10):
            print(f"  {c:3d}x  {w}")

        return

    # ── Beam search mode (original) ─────────────────────────────────────
    # Word times (Huth pipeline)
    word_rate_model_path = os.path.join(dec_config.MODEL_DIR, args.subject, f"word_rate_model_{'speech' if args.experiment in ['imagined_speech', 'perceived_movies'] else 'auditory'}.npz")
    if not os.path.exists(word_rate_model_path):
        print(f"Word rate model not found: {word_rate_model_path}. Train decoding pipeline first (train_EM, train_WR).")
        sys.exit(1)

    from utils_stim import predict_word_rate, predict_word_times
    wr = np.load(word_rate_model_path, allow_pickle=True)
    word_rate = predict_word_rate(resp, wr["weights"], wr["voxels"], wr["mean_rate"])
    starttime = -10 if args.experiment == "perceived_speech" else 0
    word_times, tr_times = predict_word_times(word_rate, resp, starttime=starttime)
    tr = 2.0
    from StimulusModel import get_lanczos_mat, affected_trs
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    # Decoder and LM (for beam proposals)
    from Decoder import Decoder, Hypothesis
    from LanguageModel import LanguageModel
    from GPT import GPT

    gpt = GPT(path="openai-gpt", vocab=None, device=dec_config.GPT_DEVICE)
    lm = LanguageModel(gpt, gpt.vocab, nuc_mass=dec_config.LM_MASS, nuc_ratio=dec_config.LM_RATIO)
    decoder = Decoder(word_times, args.beam_width, extensions=args.extensions)

    _DUMMY_EMB = None

    for sample_index in tqdm(range(len(word_times)), desc="Decoding"):
        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        ncontext = decoder.time_window(sample_index, dec_config.LM_TIME, floor=5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)

        fmri_slice = resp[trs, :]
        n_trs_actual = fmri_slice.shape[0]
        if n_trs_actual < DEFAULT_WINDOW_SIZE:
            pad = np.zeros((DEFAULT_WINDOW_SIZE - n_trs_actual, fmri_slice.shape[1]), dtype=fmri_slice.dtype)
            fmri_slice = np.concatenate([fmri_slice, pad], axis=0)
        elif n_trs_actual > DEFAULT_WINDOW_SIZE:
            fmri_slice = fmri_slice[-DEFAULT_WINDOW_SIZE:]
        fmri_T = fmri_slice.T
        if voxel_to_roi is not None:
            fmri_T = apply_roi_grouping_cpu(fmri_T, voxel_to_roi, NUM_ROIS)
        fmri_segment = torch.from_numpy(fmri_T).half().to(device)

        hyp_data = []
        hyp_candidates = []

        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1:
                continue
            hyp_data.append((hyp, nuc, logprobs, nextensions))
            hyp_candidates.append((hyp.words, list(nuc)))

        if hyp_candidates:
            all_scores = score_all_hypotheses_batched(
                model, tokenizer, device, fmri_segment, hyp_candidates,
            )

            for (hyp, nuc, logprobs, nextensions), scores in zip(hyp_data, all_scores):
                local_extensions = [Hypothesis(parent=hyp, extension=(nuc[i], logprobs[i], _DUMMY_EMB)) for i in range(len(nuc))]
                decoder.add_extensions(local_extensions, scores, nextensions)

        del fmri_segment, hyp_data, hyp_candidates
        decoder.extend(verbose=False)
        torch.cuda.empty_cache()
        if sample_index % 20 == 0:
            gc.collect()

    if args.experiment in ["perceived_movie", "perceived_multispeaker"]:
        decoder.word_times += 10

    save_dir = Path(dec_config.RESULT_DIR) / args.subject / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{args.task}_fmri_flamingo"
    decoder.save(str(out_path))
    print(f"Saved decoding result to {out_path}.npz")


if __name__ == "__main__":
    main()
