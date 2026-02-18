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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def score_candidates_with_model(model, tokenizer, device, fmri_segment, context_words, candidate_words):
    """
    Score each candidate continuation given fMRI and context using our model.
    fmri_segment: (voxels, n_trs) tensor on device
    """
    # Hard cap to 120,000 to strictly enforce model limit (debug fix)
    LIMIT = 120000
    if fmri_segment.shape[0] > LIMIT:
        # Use tqdm.write if available to avoid eating output
        try:
            from tqdm import tqdm
            tqdm.write(f"⚠️  DEBUG: Capping fMRI from {fmri_segment.shape[0]} to {LIMIT} voxels.")
        except:
            print(f"⚠️  DEBUG: Capping fMRI from {fmri_segment.shape[0]} to {LIMIT} voxels.")
        fmri_segment = fmri_segment[:LIMIT, :]
    
    scores = []
    context_str = " ".join(context_words) if context_words else ""
    with torch.no_grad():
        for cw in candidate_words:
            if context_str:
                text = f"{context_str} <image> {cw}"
                prompt_str = f"{context_str} <image>"
            else:
                text = f"<image> {cw}"
                prompt_str = "<image>"
            input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            prompt_len = len(prompt_ids)
            batch = [{
                "time_series": fmri_segment.cpu(),
                "input_ids": input_ids,
                "prompt_len": prompt_len,
            }]
            loss = model.compute_loss(batch)
            scores.append(-loss.item())
    return scores


def main():
    parser = argparse.ArgumentParser(description="Huth-style decoding with FMRIFlamingo scoring.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FMRIFlamingo checkpoint")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID (e.g. UTS09)")
    parser.add_argument("--task", type=str, required=True, help="Story/task name (e.g. adollshouse)")
    parser.add_argument("--experiment", type=str, default="perceived_speech", help="Experiment type for word times")
    parser.add_argument("--ignore-overlap", action="store_true", help="Allow decoding on training stories")
    parser.add_argument("--beam-width", type=int, default=200, help="Beam width (default 200)")
    parser.add_argument("--extensions", type=int, default=5, help="Extensions per hypothesis")
    args = parser.parse_args()

    device = get_device()

    # Load our model and tokenizer
    from src.models.fmri_flamingo import FMRIFlamingo
    from transformers import AutoTokenizer

    print(f"Loading FMRIFlamingo from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = "<PAD>"

    model = FMRIFlamingo(
        device=device,
        llm_id=LLM_ID,
        num_rois=NUM_ROIS,
        cross_attn_every_n_layers=getattr(fmri_config, "CROSS_ATTN_EVERY_N_LAYERS", 1),
        gradient_checkpointing=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load decoding config and data (same as run_decoder.py)
    try:
        import config as dec_config
    except ImportError:
        print("decoding/config.py not found. Set PYTHONPATH to include decoding/ or run from repo root.")
        sys.exit(1)

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

    # Voxel selection: match training; cap at tokenizer max_rois (MAX_PATCHES) to avoid ValueError
    n_vox = resp.shape[1]
    print(f"DEBUG: n_vox={n_vox}, MAX_PATCHES={MAX_PATCHES}, ROI_METHOD={ROI_SELECTION_METHOD}")
    if ROI_SELECTION_METHOD == "all_voxels":
        voxel_inds = np.arange(min(n_vox, MAX_PATCHES))
        if n_vox > MAX_PATCHES:
            print(f"⚠️  Capping voxels to {MAX_PATCHES} (model max_rois); data has {n_vox} voxels.")
    else:
        if n_vox > NUM_ROIS:
            voxel_inds = np.arange(NUM_ROIS)
        else:
            voxel_inds = np.arange(n_vox)
        print(f"⚠️  ROI_SELECTION_METHOD={ROI_SELECTION_METHOD}; using first {len(voxel_inds)} voxels.")
    print(f"DEBUG: Final voxel_inds length: {len(voxel_inds)}")

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

    # Decode loop: same as run_decoder, but score with our model
    from StimulusModel import LMFeatures
    features = LMFeatures(gpt, dec_config.GPT_LAYER, dec_config.GPT_WORDS)

    for sample_index in tqdm(range(len(word_times)), desc="Decoding"):
        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        ncontext = decoder.time_window(sample_index, dec_config.LM_TIME, floor=5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)

        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1:
                continue
            extend_words = [hyp.words + [x] for x in nuc]
            candidate_words = list(nuc)

            # fMRI segment: (n_trs, voxels) -> (voxels, n_trs) for our model
            fmri_slice = resp[np.ix_(trs, voxel_inds)]
            fmri_segment = torch.from_numpy(fmri_slice.T).float().to(device)

            scores = score_candidates_with_model(
                model, tokenizer, device,
                fmri_segment, hyp.words, candidate_words,
            )
            extend_embs = list(features.extend(extend_words))
            local_extensions = [Hypothesis(parent=hyp, extension=(nuc[i], logprobs[i], extend_embs[i])) for i in range(len(nuc))]
            decoder.add_extensions(local_extensions, scores, nextensions)
        decoder.extend(verbose=False)

    if args.experiment in ["perceived_movie", "perceived_multispeaker"]:
        decoder.word_times += 10

    save_dir = Path(dec_config.RESULT_DIR) / args.subject / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{args.task}_fmri_flamingo"
    decoder.save(str(out_path))
    print(f"Saved decoding result to {out_path}.npz")


if __name__ == "__main__":
    main()
