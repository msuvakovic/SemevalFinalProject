#!/usr/bin/env python3
"""
Visualize FMRIFlamingo model graph in TensorBoard.

Builds the model, runs one batch through pad_and_apply_batch, and logs the inner
model graph (forward pass structure and tensor shapes) via SummaryWriter.add_graph.

Usage:
    python scripts/visualize_model_tensorboard.py [--logdir DIR] [--dummy]
    tensorboard --logdir runs/viz   # then open http://localhost:6006

Options:
    --logdir   Directory for TensorBoard logs (default: runs/viz).
    --dummy    Use a minimal dummy batch instead of loading the train dataset
               (avoids data/TextGrid dependency; useful when data is not present).

Requirements:
    Same environment as train.py (open-flamingo, transformers, etc.).
    TensorBoard: pip install tensorboard
"""

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    raise ImportError(
        "TensorBoard is required for this script. Install with: pip install tensorboard"
    ) from None

# Project root
SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

# Config (same as train.py)
import importlib.util
config_path = FMRI_FLAMINGO_DIR / "config.py"
spec = importlib.util.spec_from_file_location("fmri_config", config_path)
fmri_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmri_config)

LLM_ID = getattr(fmri_config, "LLM_ID", "meta-llama/Llama-3.2-1B")
NUM_ROIS = getattr(fmri_config, "NUM_ROIS", 200)
CROSS_ATTN_EVERY_N_LAYERS = getattr(fmri_config, "CROSS_ATTN_EVERY_N_LAYERS", 1)
GRADIENT_CHECKPOINTING = getattr(fmri_config, "GRADIENT_CHECKPOINTING", True)
DEFAULT_WINDOW_SIZE = getattr(fmri_config, "DEFAULT_WINDOW_SIZE", 10)
DEFAULT_STRIDE = getattr(fmri_config, "DEFAULT_STRIDE", 10)
WINDOW_SIZE = DEFAULT_WINDOW_SIZE
STRIDE = DEFAULT_STRIDE


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dummy_batch(tokenizer, num_voxels: int, num_trs: int):
    """Build a single-sample batch with correct keys and shapes for pad_and_apply_batch."""
    # time_series: (voxels, TRs)
    time_series = torch.randn(num_voxels, num_trs)
    # Minimal text: <image> word <|endofchunk|>
    text = "<image> hello <|endofchunk|>"
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    prompt_text = "<image> hello"
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    prompt_len = len(prompt_tokens)
    return [{
        "time_series": time_series,
        "input_ids": tokens,
        "prompt_len": prompt_len,
    }]


def main():
    parser = argparse.ArgumentParser(description="Log FMRIFlamingo model graph to TensorBoard.")
    parser.add_argument("--logdir", type=str, default="runs/viz", help="TensorBoard log directory")
    parser.add_argument("--dummy", action="store_true", help="Use dummy batch (no dataset load)")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    device = get_device()
    print(f"Device: {device}")
    print(f"Log dir: {logdir.absolute()}")

    # Tokenizer (same as train)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.pad_token = "<PAD>"

    # Batch: real or dummy
    if args.dummy:
        print("Using dummy batch (no dataset).")
        batch = build_dummy_batch(tokenizer, num_voxels=NUM_ROIS, num_trs=WINDOW_SIZE)
    else:
        from src.datasets.huth_fmri_dataset import create_splits, HuthFMRIDataset
        from torch.utils.data import DataLoader

        train_stories, _, _ = create_splits()
        train_dataset = HuthFMRIDataset(
            split="train",
            story_names=train_stories,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            tokenizer=tokenizer,
        )
        if len(train_dataset) == 0:
            print("No train samples; re-run with --dummy to visualize without data.")
            sys.exit(1)

        def collate_fn(batch):
            collated = {}
            for key in batch[0].keys():
                if key in ["time_series", "input_ids"]:
                    collated[key] = [item[key] for item in batch]
                else:
                    collated[key] = [item[key] for item in batch]
            return collated

        loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        batch = next(iter(loader))
        print("Using one batch from train loader.")

    # Model (same as train.py)
    from src.models.fmri_flamingo import FMRIFlamingo
    print("Building model...")
    model = FMRIFlamingo(
        device=device,
        llm_id=LLM_ID,
        num_rois=NUM_ROIS,
        cross_attn_every_n_layers=CROSS_ATTN_EVERY_N_LAYERS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )
    model.eval()

    # Prepare inputs via pad_and_apply_batch (moves to device)
    with torch.no_grad():
        input_ids, images, attention_mask, labels = model.pad_and_apply_batch(batch, include_labels=True)

    print(f"Batch shapes: input_ids={input_ids.shape}, images={images.shape}, attention_mask={attention_mask.shape}")
    if labels is not None:
        print(f"  labels={labels.shape}")

    # Log graph: inner model forward(vision_x, lang_x, attention_mask, labels)
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))
    try:
        # add_graph expects model and a tuple of inputs (passed as *args to forward)
        writer.add_graph(model.model, (images, input_ids, attention_mask, labels))
        print("Model graph written to TensorBoard.")
    except Exception as e:
        print(f"add_graph failed: {e}")
        print("Trying without labels (some Flamingo forwards accept labels as optional).")
        try:
            writer.add_graph(model.model, (images, input_ids, attention_mask))
            print("Model graph written (without labels).")
        except Exception as e2:
            print(f"add_graph also failed without labels: {e2}")
            raise
    finally:
        writer.close()

    print()
    print("Done. To view the graph, run:")
    print(f"  tensorboard --logdir {logdir.absolute()}")
    print("  Then open http://localhost:6006 and check the Graphs tab.")


if __name__ == "__main__":
    main()
