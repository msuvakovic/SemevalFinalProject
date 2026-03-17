"""
Configuration file for fMRI-Flamingo project.

Centralizes all hyperparameters, paths, and settings.
"""

from pathlib import Path

# ============================================================================
# Directory Paths
# ============================================================================

# Project root: brain-model-alignment
FMRI_FLAMINGO_DIR = Path(__file__).parent
BASE_DIR = FMRI_FLAMINGO_DIR.parent

# Data paths: use project-relative data/ so Huth lives in brain-model-alignment/data/Huth/
DATA_DIR = FMRI_FLAMINGO_DIR / "data" / "Huth"
PREPROCESSED_DATA_DIR = DATA_DIR / "derivative" / "preprocessed_data"
RESPDICT_PATH = DATA_DIR / "derivative" / "respdict.json"

# TextGrid directories (try multiple locations)
TEXTGRID_DIRS = [
    DATA_DIR / "derivative" / "TextGrids",  # Capital T and G (preferred)
    DATA_DIR / "derivative" / "textgrids",  # Lowercase fallback
    BASE_DIR / "semantic-decoding" / "data" / "Huth" / "derivative" / "TextGrids",
]

# Checkpoints and outputs
CHECKPOINT_DIR = FMRI_FLAMINGO_DIR / "checkpoints"
LOGS_DIR = FMRI_FLAMINGO_DIR / "logs"

# Create directories
for dir_path in [CHECKPOINT_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Dataset Configuration
# ============================================================================

# Subject IDs (from Huth dataset)
AVAILABLE_SUBJECTS = ["UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08", "UTS09"]

# Default subjects for training/validation/test splits
# Per-subject setup: train/val on UTS01–UTS08; we leave out UTS09 (no separate test subject).
TRAIN_SUBJECTS = ["UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"]
TEST_SUBJECTS = []  # Empty: we do per-subject; subject 9 left out

# Data splitting
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# Hemodynamic delay (seconds)
# Typical BOLD response peaks at 4-6 seconds after stimulus
HEMODYNAMIC_DELAY = 4.0  # seconds
HEMODYNAMIC_DELAY_RANGE = [2.0, 3.0, 4.0, 5.0, 6.0]  # For multi-delay models

# TR (Repetition Time) in seconds
TR = 2.0  # seconds per TR

# Trim first/last N TRs (to account for scanner warm-up and cool-down)
TRIM_TRS = 5

# Sliding window parameters (for dataset)
DEFAULT_WINDOW_SIZE = 10  # Number of TRs per sample (default, can be overridden)
DEFAULT_STRIDE = 10  # Stride between windows (default, can be overridden)

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Model Configuration (TUNING SECTION)
# ============================================================================

# LLM backbone
LLM_ID = "meta-llama/Llama-3.2-1B"  # Can also use "google/gemma-3-270m"
# Attention: "eager" (default, most compatible), "sdpa" (faster, PyTorch 2 built-in), "flash_attention_2" (fastest, needs flash-attn + Ampere+ GPU)
ATTN_IMPLEMENTATION = "sdpa"

# Tokenization strategy: "roi" or "spatiotemporal"
TOKENIZATION_STRATEGY = "roi"  # Start with ROI-based (simpler, interpretable)

# ROI-based tokenization
NUM_ROIS = 2000  # 2k ROIs (averages ~40 voxels each for 81k voxels); all_voxels=~50k OOMs on 16GB
ROI_SELECTION_METHOD = "anatomical"  # Options: "anatomical", "learned", "random", "all_voxels"
# "anatomical" = deterministic seeded shuffle + contiguous blocks (reproducible, balanced ROI sizes)
# "random" = random assignment (varies per run). "all_voxels" = no ROI reduction.

# Spatiotemporal patch tokenization (alternative to ROI)
PATCH_SIZE_TEMPORAL = 4  # TRs per patch
PATCH_SIZE_SPATIAL = 100  # Voxels per patch (if using spatiotemporal)

# Encoder output dimensions
ENCODER_OUTPUT_DIM = 512  # Dimension of token embeddings (matches OpenTSLM default)
TRANSFORMER_INPUT_DIM = 128  # Input dimension to transformer layers

# Positional embeddings
MAX_PATCHES = 120000  # Increased to 120k to support "all_voxels" (some subjects have >100k voxels)

# Flamingo architecture
# --- Structural levers (if loss won't go down or creeps up, LR alone may not fix it) ---
# • CROSS_ATTN_EVERY_N_LAYERS: 1 = every layer sees fMRI (stronger conditioning). 2,4,8 = fewer injection points.
# • Perceiver num_latents: open_flamingo default (e.g. 64). More latents = more "slots" per <image> (requires code change to pass num_latents).
# • LOG_GRAD_NORMS_EVERY: set to 100 and check encoder/perceiver norms; if ~0, gradients aren't reaching the fMRI path.
# • DROPOUT: 0.2 is moderate; 0.1 if underfitting, 0.3 if overfitting.
# • LLM frozen in ORPO: only encoder + perceiver + gated_cross_attn train; unfreezing LLM (train.py LR_LLM) would need a separate run.
# • CROSS_ATTN_GATE_INIT: 0 = vision path off at init (default Flamingo). 0.5 = start half-open so fMRI is used from step 1 (try if loss/acc stuck).
CROSS_ATTN_EVERY_N_LAYERS = 4  # 4 = cross-attn every 4th layer (4 injection points in 16-layer Llama-1B); 1 OOMs on 16GB
CROSS_ATTN_GATE_INIT = 0.5  # 0.5 = vision path half-open from step 1 so fMRI encoder gets gradients immediately

# TUNING: 0.5 often prevents loss from going down (encoder/cross-attn get too much dropout). Try 0.1–0.2 first.
DROPOUT = 0.2  # Dropout probability for fMRI encoder and cross-attention

# TUNING: Probability of masking text tokens during training to force fMRI usage.
#         Set to 0.0 to disable. Set to 0.5-0.75 to fix Posterior Collapse.
TEXT_MASKING_PROB = 0.5  # Mask 50% of text tokens to force fMRI dependency

# Contrastive/ranking loss: train "score correct continuation > distractors" given prompt + fMRI.
# RANKING_LOSS_WEIGHT: weight of ranking loss vs CE loss when both are applied.
# RANKING_LOSS_FRAC: fraction of batches that add ranking loss (e.g. 0.5 = every other batch).
# NUM_RANKING_DISTRACTORS: number of in-batch or pooled distractors per sample (1 correct + K distractors).
# RANKING_CHUNK_SIZE: process ranking candidates in chunks to avoid OOM (e.g. 2 = 2 candidates per forward).
# TELEPATHY_RANKING_FRAC: of ranking batches, fraction that are Telepathy (no prompt, only fMRI → rank 100).
#   Set to 1.0 to only train for Telepathy (maximize fMRI-only eval). Set to 0.0 for only full-prompt ranking.
# RESTRICT_GENERATION_TO_WORD_VOCAB: if True, validation generation only allows tokens that appear in
#   English words (avoids code/special tokens like sp.ArgumentParser, {BR}). Requires src.utils.word_vocab.
RANKING_LOSS_WEIGHT = 0.5
RESTRICT_GENERATION_TO_WORD_VOCAB = False
RANKING_LOSS_FRAC = 1  # Disabled: LLM unfrozen needs gradient memory, ranking OOMs
NUM_RANKING_DISTRACTORS = 7  # 1 correct + 7 distractors = 8 candidates when batch_size=1
RANKING_CHUNK_SIZE = 1  # Forward 1 candidate at a time to stay under GPU memory
TELEPATHY_RANKING_FRAC = 1.0  # 1.0 = only Telepathy ranking (no prompt); 0.0 = only full-prompt ranking

# Telepathy ORPO (scripts/train_telepathy_orpo.py): preference loss for fMRI-only ranking.
NUM_ORPO_REJECTS = 7  # number of rejected (distractor) sequences per chosen sequence
ORPO_BETA = 0.1  # temperature in ORPO loss: -log σ(β * (log P(chosen) - log P(rejected)))


# ============================================================================
# Training Configuration (TUNING SECTION)
# ============================================================================

# Batch size (reduced for memory efficiency)
BATCH_SIZE = 2  # Reduced from 4 to avoid OOM - use gradient accumulation instead
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS = 4

# Learning rates
# TUNING: Lower these (e.g. 5e-5) if loss is unstable or overfits quickly.
LR_ENCODER = 1e-4  # Restored: grad explosion was from pos_embed init, not LR
LR_PROJECTOR = 2e-4  # Bumped: perceiver was starved (grad_norm=0.00)
LR_BASE = 1e-4  # Base LR for cross-attn layers
LR_LLM = 0.0  # Frozen: unfreezing caused LLM to memorize stories, not use fMRI
LR_ORPO = 2e-5  # For train_telepathy_orpo.py. If avg loss creeps up over epoch, try 1e-5.

# Training schedule
NUM_EPOCHS = 20
WARMUP_FRAC = 0.1  # Increased warmup (was 0.03)

# TUNING: Increase (e.g. 0.1, 0.2) for stronger regularization.
WEIGHT_DECAY = 0.3  # Increased from 1e-2 for stronger regularization
GRAD_CLIP_NORM = 1.0  # Gradient clipping

# Early stopping
EARLY_STOP_PATIENCE = 5  # Stop if no improvement for N epochs
EARLY_STOP_METRIC = "val_loss"  # Metric to monitor

# Checkpointing
SAVE_EVERY_N_EPOCHS = 1  # Save checkpoint every N epochs
KEEP_N_CHECKPOINTS = 3  # Keep only last N checkpoints

# Memory optimization
USE_GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing (trades compute for memory)
USE_MIXED_PRECISION = True  # Use FP16/BF16 mixed precision training
MIXED_PRECISION_DTYPE = "bf16"  # "bf16" (better) or "fp16" (more compatible)
DATALOADER_NUM_WORKERS = 8  # Set to 0 to avoid memory issues with multiprocessing

# Stability and observability (WSL2/OOM prevention)
ENABLE_MEMORY_LOGGING = True  # Log memory usage to logs/metrics.jsonl
MEMORY_LOG_INTERVAL = 10  # Log every N batches
LOG_GRAD_NORMS_EVERY = 100  # Log encoder/perceiver grad norms every 100 optimizer steps to verify fMRI path gets gradients
ENABLE_SAFETY_STOP = False  # Safety stop if RSS exceeds threshold (default OFF, opt-in)
SAFETY_STOP_RSS_GB = 12.0  # RSS threshold in GB (only used if ENABLE_SAFETY_STOP=True)

# Gradient checkpointing (for model config)
GRADIENT_CHECKPOINTING = USE_GRADIENT_CHECKPOINTING  # Alias for compatibility

# ============================================================================
# Helper Functions
# ============================================================================

def _normalize_story_name(s: str) -> str:
    """Normalize for matching: lowercase, remove underscores (e.g. a_fathers_cover -> afatherscover)."""
    return s.lower().replace("_", "")


def get_fmri_path(subject_id: str, story_name: str) -> Path:
    """
    Get path to fMRI .hf5 file for a subject and story.
    Tries: exact name, case-insensitive, then normalized (underscores removed).
    
    Args:
        subject_id: Subject ID (e.g., "UTS01")
        story_name: Story name (e.g., "adollshouse" or "a_fathers_cover")
    
    Returns:
        Path to .hf5 file
    
    Raises:
        FileNotFoundError: If no matching .hf5 file exists
    """
    subj_dir = PREPROCESSED_DATA_DIR / subject_id
    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject dir not found: {subj_dir}")
    exact = subj_dir / f"{story_name}.hf5"
    if exact.exists():
        return exact
    story_lower = story_name.lower()
    story_norm = _normalize_story_name(story_name)
    for p in subj_dir.glob("*.hf5"):
        if p.stem.lower() == story_lower:
            return p
        if _normalize_story_name(p.stem) == story_norm:
            return p
    raise FileNotFoundError(f"No .hf5 file for story {story_name!r} in {subj_dir}")


def get_textgrid_path(story_name: str) -> Path:
    """
    Get path to TextGrid file for a story.
    Tries: exact name, case-insensitive, then normalized (underscores removed).
    
    Searches multiple possible locations.
    
    Args:
        story_name: Story name (e.g., "adollshouse" or "a_fathers_cover")
    
    Returns:
        Path to TextGrid file (first found)
    
    Raises:
        FileNotFoundError: If TextGrid file not found in any location
    """
    story_lower = story_name.lower()
    story_norm = _normalize_story_name(story_name)
    for textgrid_dir in TEXTGRID_DIRS:
        if not textgrid_dir.exists():
            continue
        exact = textgrid_dir / f"{story_name}.TextGrid"
        if exact.exists():
            return exact
        for p in textgrid_dir.glob("*.TextGrid"):
            if p.stem.lower() == story_lower:
                return p
            if _normalize_story_name(p.stem) == story_norm:
                return p
    
    raise FileNotFoundError(
        f"TextGrid file not found for story '{story_name}' in any of these locations:\n"
        + "\n".join(f"  - {d}" for d in TEXTGRID_DIRS)
    )


def get_checkpoint_path(experiment_name: str = None, epoch: int = None) -> Path:
    """
    Get path to checkpoint file.
    
    Args:
        experiment_name: Optional experiment name
        epoch: Optional epoch number
    
    Returns:
        Path to checkpoint file
    """
    if experiment_name and epoch is not None:
        return CHECKPOINT_DIR / f"{experiment_name}_epoch_{epoch}.pt"
    elif epoch is not None:
        return CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
    elif experiment_name:
        return CHECKPOINT_DIR / f"{experiment_name}_checkpoint.pt"
    else:
        return CHECKPOINT_DIR / "checkpoint.pt"
