"""
Huth fMRI Dataset Loader

Loads fMRI data from .hf5 files and aligns with TextGrid word timings.
Formats data for Flamingo-style training (tokenized fMRI → language).
"""
from __future__ import annotations

import os
import sys
import json
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Literal, Optional
from torch.utils.data import Dataset

# Add parent directories to path
# File is at: brain-model-alignment/src/datasets/huth_fmri_dataset.py
FMRI_FLAMINGO_DIR = Path(__file__).parent.parent.parent
BASE_DIR = FMRI_FLAMINGO_DIR.parent

# In-repo decoding dir has utils_ridge/textgrid.py (TextGrid parser)
PROJECT_DECODING_DIR = FMRI_FLAMINGO_DIR / "decoding"
SEMANTIC_DECODING_DIR = BASE_DIR / "semantic-decoding" / "decoding"

# IMPORTANT: Add project root first so our config.py is found
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))
if str(SEMANTIC_DECODING_DIR) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_DECODING_DIR))
# Add in-repo decoding last so it is first in path: use our TextGrid parser (decoding/utils_ridge/textgrid.py)
if str(PROJECT_DECODING_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DECODING_DIR))

# Import config from fmri-flamingo (not semantic-decoding)
# Use importlib to ensure we get the right config
import importlib.util
config_path = FMRI_FLAMINGO_DIR / "config.py"
spec = importlib.util.spec_from_file_location("fmri_config", config_path)
fmri_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmri_config)

# Import from our config module
# Import config variables
PREPROCESSED_DATA_DIR = fmri_config.PREPROCESSED_DATA_DIR
RESPDICT_PATH = fmri_config.RESPDICT_PATH
TEXTGRID_DIRS = fmri_config.TEXTGRID_DIRS
get_textgrid_path = fmri_config.get_textgrid_path
get_fmri_path = fmri_config.get_fmri_path
TR = fmri_config.TR
HEMODYNAMIC_DELAY = fmri_config.HEMODYNAMIC_DELAY
TRIM_TRS = fmri_config.TRIM_TRS
TRAIN_SUBJECTS = fmri_config.TRAIN_SUBJECTS
TEST_SUBJECTS = fmri_config.TEST_SUBJECTS
TRAIN_FRAC = fmri_config.TRAIN_FRAC
VAL_FRAC = fmri_config.VAL_FRAC
TEST_FRAC = fmri_config.TEST_FRAC
RANDOM_SEED = fmri_config.RANDOM_SEED

# Import TextGrid parser from semantic-decoding
try:
    from utils_ridge.textgrid import TextGrid
    HAS_TEXTGRID = True
except ImportError:
    HAS_TEXTGRID = False
    print("⚠️  Warning: TextGrid parser not found. Word timing alignment will be limited.")


class TRFile:
    """Simulates TR timing for a given number of TRs."""
    
    def __init__(self, tr: float = TR):
        self.tr = tr
        self.trtimes = []
        self.soundstarttime = 0.0
        self.soundstoptime = 0.0
    
    def simulate(self, num_trs: int, start_time: float = 0.0):
        """Generate TR times for given number of TRs."""
        self.soundstarttime = start_time
        self.trtimes = [start_time + i * self.tr for i in range(num_trs)]
        self.soundstoptime = self.trtimes[-1] + self.tr
    
    def get_reltriggertimes(self) -> np.ndarray:
        """Get relative trigger times."""
        return np.array(self.trtimes)


def load_fmri_data(subject_id: str, story_name: str, voxels: Optional[np.ndarray] = None, 
                   tr_start: Optional[int] = None, tr_end: Optional[int] = None) -> np.ndarray:
    """
    Load fMRI data from .hf5 file.
    
    Args:
        subject_id: Subject ID (e.g., "UTS01")
        story_name: Story name (e.g., "adollshouse")
        voxels: Optional array of voxel indices to select
        tr_start: Optional start TR index (for loading only a window)
        tr_end: Optional end TR index (for loading only a window)
    
    Returns:
        fMRI data array of shape (TRs, voxels) or (window_size, voxels) if tr_start/tr_end specified
    """
    fmri_path = get_fmri_path(subject_id, story_name)
    
    with h5py.File(fmri_path, "r") as hf:
        data_key = hf["data"]
        
        # Load only the window we need (much faster!)
        if tr_start is not None and tr_end is not None:
            # Load only the window: (tr_end - tr_start, voxels)
            if voxels is not None:
                # Load window and select voxels
                window_data = data_key[tr_start:tr_end, :]  # (window_size, all_voxels)
                data = window_data[:, voxels]  # (window_size, selected_voxels)
            else:
                # Load window only
                data = data_key[tr_start:tr_end, :]  # (window_size, voxels)
        else:
            # Load all data (slower, but needed for some operations)
            if voxels is not None:
                # Load all and select voxels
                data = data_key[:, voxels]  # (TRs, selected_voxels)
            else:
                # Load everything (slow for large files!)
                data = data_key[:]  # (TRs, voxels)
        
        # Convert to numpy and handle NaNs/Infs
        data = np.array(data)
        data = np.nan_to_num(data)
    
    return data


def _norm_story(s: str) -> str:
    """Normalize story name for matching: lowercase, no underscores."""
    return s.lower().replace("_", "")


def load_textgrid(story_name: str) -> Optional["TextGrid"]:
    """Load TextGrid file for a story. Tries get_textgrid_path, then scan dirs by normalized name."""
    if not HAS_TEXTGRID:
        return None
    
    # 1) Use config path helper (exact / case / normalized)
    try:
        textgrid_path = get_textgrid_path(story_name)
        if textgrid_path.exists():
            with open(textgrid_path, "r", encoding="utf-8", errors="replace") as f:
                return TextGrid(f.read())
    except FileNotFoundError:
        pass
    except (TypeError, ValueError, Exception):
        pass
    
    # 2) Fallback: scan dirs and open first file that matches (normalized) and parses
    story_norm = _norm_story(story_name)
    for tg_dir in TEXTGRID_DIRS:
        if not tg_dir.exists():
            continue
        for p in tg_dir.glob("*.TextGrid"):
            if _norm_story(p.stem) != story_norm:
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    return TextGrid(f.read())
            except (OSError, TypeError, ValueError, Exception):
                continue
    return None


def extract_words_from_textgrid(textgrid: "TextGrid") -> List[Tuple[str, float, float]]:
    """
    Extract words and their timing from TextGrid.
    
    Returns:
        List of (word, start_time, end_time) tuples
    """
    if textgrid is None:
        return []
    
    words = []
    # Prefer a tier named words/word/transcript; fallback to first tier with non-empty transcript
    chosen_tier = None
    for tier in textgrid.tiers:
        tier_name = (tier.nameid or "").lower() if hasattr(tier, "nameid") else ""
        if tier_name in ["words", "word", "transcript"]:
            chosen_tier = tier
            break
    if chosen_tier is None:
        for tier in textgrid.tiers:
            if getattr(tier, "simple_transcript", None) and len(tier.simple_transcript) > 0:
                chosen_tier = tier
                break
    if chosen_tier is not None:
        for start_time, end_time, text in chosen_tier.simple_transcript:
            word = (text or "").strip()
            if word:
                words.append((word, float(start_time), float(end_time)))
    return words


def align_trs_to_words(
    tr_times: np.ndarray,
    words: List[Tuple[str, float, float]],
    hemodynamic_delay: float = HEMODYNAMIC_DELAY
) -> List[Dict]:
    """
    Align TRs with words, accounting for hemodynamic delay.
    
    Args:
        tr_times: Array of TR times in seconds
        words: List of (word, start_time, end_time) tuples
        hemodynamic_delay: Delay in seconds (BOLD response peaks after this delay)
    
    Returns:
        List of dicts with keys: 'tr_idx', 'tr_time', 'words', 'word_times'
    """
    alignments = []
    
    for tr_idx, tr_time in enumerate(tr_times):
        # Account for hemodynamic delay: fMRI at time T reflects stimulus at time T - delay
        stimulus_time = tr_time - hemodynamic_delay
        
        # Find words that were active around this stimulus time
        # Use a window around the stimulus time (e.g., ±1 second)
        window = 1.0
        active_words = []
        word_times = []
        
        for word, start_time, end_time in words:
            # Word is active if it overlaps with the stimulus time window
            if (start_time <= stimulus_time + window and 
                end_time >= stimulus_time - window):
                active_words.append(word)
                word_times.append((start_time, end_time))
        
        alignments.append({
            'tr_idx': tr_idx,
            'tr_time': tr_time,
            'stimulus_time': stimulus_time,
            'words': active_words,
            'word_times': word_times,
        })
    
    return alignments


def create_word_context(words: List[str], context_window: int = 5) -> str:
    """Create text context from words."""
    return " ".join(words[-context_window:]) if words else ""


class HuthFMRIDataset(Dataset):
    """
    PyTorch Dataset for Huth fMRI data.
    
    Each sample contains:
    - fMRI signal: (voxels, TRs) tensor
    - Text context: Words aligned with the fMRI window
    - Target: Next word or continuation
    """
    
    def __init__(
        self,
        split: Literal["train", "validation", "test"],
        subject_ids: Optional[List[str]] = None,
        story_names: Optional[List[str]] = None,
        window_size: int = 10,  # Number of TRs per sample
        stride: int = 5,  # Stride between windows
        context_words: int = 5,  # Number of words for context
        voxels: Optional[np.ndarray] = None,  # Voxel selection
        max_samples: Optional[int] = None,  # Limit number of samples
        EOS_TOKEN: str = "<|endofchunk|>",
        tokenizer = None, # Optional tokenizer for pre-tokenization
    ):
        """
        Initialize dataset.
        
        Args:
            split: Dataset split
            subject_ids: List of subject IDs to use (None = use config defaults)
            story_names: List of story names to use (None = use all available)
            window_size: Number of consecutive TRs per sample
            stride: Stride between windows
            context_words: Number of words to use as context
            voxels: Optional voxel indices to select
            max_samples: Maximum number of samples (for quick testing)
            EOS_TOKEN: End-of-sequence token
            tokenizer: HuggingFace tokenizer (optional). If provided, performs tokenization.
        """
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.context_words = context_words
        self.voxels = voxels
        self.EOS_TOKEN = EOS_TOKEN
        self.tokenizer = tokenizer
        
        # Cache for H5 handles (initialized lazily in __getitem__)
        self._h5_handles = {}
        
        # Determine subjects and stories
        if subject_ids is None:
            if split == "test":
                subject_ids = TEST_SUBJECTS
            else:
                subject_ids = TRAIN_SUBJECTS
        
        # Story names: prefer discovery from files on disk so we match actual .hf5 and .TextGrid
        available_stories = []
        if story_names is None:
            # Discover stories that have BOTH .hf5 (for at least one subject) AND .TextGrid
            hf5_stems = set()
            for sid in subject_ids:
                subj_dir = PREPROCESSED_DATA_DIR / sid
                if subj_dir.exists():
                    for p in subj_dir.glob("*.hf5"):
                        hf5_stems.add(p.stem)
            tg_stems = set()
            for tg_dir in TEXTGRID_DIRS:
                if tg_dir.exists():
                    for p in tg_dir.glob("*.TextGrid"):
                        tg_stems.add(p.stem)
            overlap = hf5_stems & tg_stems
            if overlap:
                available_stories = sorted(overlap)
            else:
                # Case-insensitive overlap (e.g. Life.hf5 vs life.TextGrid)
                norm_overlap = set()
                for h in hf5_stems:
                    for t in tg_stems:
                        if h.lower() == t.lower():
                            norm_overlap.add(h)  # use hf5 stem so get_fmri_path finds file
                            break
                if norm_overlap:
                    available_stories = sorted(norm_overlap)
                else:
                    # Normalized overlap (underscores removed: afatherscover vs a_fathers_cover)
                    def _norm(s: str) -> str:
                        return s.lower().replace("_", "")
                    norm2 = set()
                    for h in hf5_stems:
                        for t in tg_stems:
                            if _norm(h) == _norm(t):
                                norm2.add(h)
                                break
                    if norm2:
                        available_stories = sorted(norm2)
            if not available_stories and RESPDICT_PATH.exists():
                with open(RESPDICT_PATH, "r") as f:
                    respdict = json.load(f)
                available_stories = list(respdict.keys())
                print("⚠️  Using respdict for story names; no overlap found between .hf5 and .TextGrid filenames.")
            if not available_stories:
                print("⚠️  No stories found. Check .hf5 and .TextGrid paths and respdict.json.")
            story_names = available_stories
        
        # Filter to stories that exist for subjects
        self.story_names = story_names
        self.subject_ids = subject_ids
        
        # Load and process samples
        self.samples = self._load_samples(max_samples)
        
        print(f"✅ Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Load and process all samples.
        
        NOTE: Uses lazy loading - stores only metadata (not fMRI tensors) in memory.
        fMRI data is loaded on-demand in __getitem__. This prevents storing all
        tensors in memory, saving ~13GB RAM for large datasets.
        Memory usage: ~0.01-0.02 MB per sample (metadata only) vs ~3MB per sample (with tensors).
        """
        samples = []
        n_skipped_no_file = 0
        n_skipped_no_words = 0
        n_pairs_tried = 0
        
        for subject_id in self.subject_ids:
            for story_name in self.story_names:
                n_pairs_tried += 1
                try:
                    fmri_path = get_fmri_path(subject_id, story_name)
                except FileNotFoundError:
                    n_skipped_no_file += 1
                    continue  # This subject doesn't have this story; skip silently
                if not fmri_path.exists():
                    n_skipped_no_file += 1
                    continue
                try:
                    # First, get the total number of TRs (without loading all data)
                    with h5py.File(fmri_path, "r") as hf:
                        total_trs = hf["data"].shape[0]
                    
                    # Trim edges
                    effective_start = TRIM_TRS
                    effective_end = total_trs - TRIM_TRS
                    num_trs = effective_end - effective_start
                    
                    # Load TextGrid
                    textgrid = load_textgrid(story_name)
                    words = extract_words_from_textgrid(textgrid) if textgrid else []
                    if not words:
                        n_skipped_no_words += 1
                        continue
                    
                    # Generate TR times (accounting for trimming)
                    tr_file = TRFile(tr=TR)
                    tr_file.simulate(num_trs)
                    tr_times = tr_file.get_reltriggertimes()
                    
                    # Align TRs with words
                    alignments = align_trs_to_words(tr_times, words, HEMODYNAMIC_DELAY)
                    
                    # Create sliding windows (load data on-demand per window)
                    for start_idx in range(0, num_trs - self.window_size + 1, self.stride):
                        end_idx = start_idx + self.window_size
                        
                        # Load only this window (much faster!)
                        tr_start_abs = effective_start + start_idx
                        tr_end_abs = effective_start + end_idx
                        fmri_window = load_fmri_data(
                            subject_id, story_name, self.voxels,
                            tr_start=tr_start_abs, tr_end=tr_end_abs
                        )  # Shape: (window_size, voxels)
                        
                        # Transpose to (voxels, window_size) for tokenization
                        fmri_window = fmri_window.T  # (voxels, window_size)
                        
                        # Get words for this window
                        window_alignments = alignments[start_idx:end_idx]
                        all_words = []
                        for align in window_alignments:
                            all_words.extend(align['words'])
                        
                        # Create context and target
                        if len(all_words) > self.context_words:
                            context_words = all_words[:-1][-self.context_words:]
                            target_word = all_words[-1]
                        elif len(all_words) > 0:
                            context_words = all_words[:-1] if len(all_words) > 1 else []
                            target_word = all_words[-1] if len(all_words) > 0 else ""
                        else:
                            # No words aligned, skip this window
                            continue
                        
                        context_text = " ".join(context_words)
                        target_text = target_word + self.EOS_TOKEN
                        
                        # Store only metadata (lazy loading - load fMRI data on-demand)
                        # This prevents storing all fMRI tensors in memory (saves ~13GB)
                        sample = {
                            # Don't store 'fmri' tensor - load on-demand in __getitem__
                            'pre_prompt': context_text,
                            'post_prompt': "Predict the next word:",
                            'answer': target_text,
                            'subject_id': subject_id,
                            'story_name': story_name,
                            'tr_start': effective_start + start_idx,  # Absolute TR index
                            'tr_end': effective_start + end_idx,  # Absolute TR index
                            'voxels': self.voxels,  # Store voxel selection for lazy loading
                        }
                        
                        samples.append(sample)
                        
                        if max_samples and len(samples) >= max_samples:
                            return samples
                
                except Exception as e:
                    print(f"⚠️  Error loading {subject_id}/{story_name}: {e}")
                    continue
        
        if len(samples) == 0 and n_pairs_tried > 0:
            print(f"   ⚠️  Diagnosis: tried {n_pairs_tried} (subject,story) pairs; "
                  f"skipped {n_skipped_no_file} (no .hf5), {n_skipped_no_words} (no words from TextGrid). "
                  f"Stories: {len(self.story_names)}, subjects: {len(self.subject_ids)}.")
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_fmri_data_cached(self, subject_id: str, story_name: str, 
                              tr_start: int, tr_end: int, voxels: Optional[np.ndarray]) -> np.ndarray:
        """
        Load fMRI data using cached file handle.
        """
        # Ensure cache exists (needed for pickling safety across workers)
        if not hasattr(self, '_h5_handles'):
            self._h5_handles = {}
            
        key = (subject_id, story_name)
        if key not in self._h5_handles:
            fmri_path = get_fmri_path(subject_id, story_name)
            # Use default mode
            self._h5_handles[key] = h5py.File(fmri_path, "r")
            
        hf = self._h5_handles[key]
        data_key = hf["data"]
        
        # Load specific window
        if voxels is not None:
            # Load window and select voxels
            window_data = data_key[tr_start:tr_end, :]
            data = window_data[:, voxels]
        else:
            # Load window only
            data = data_key[tr_start:tr_end, :]
            
        # Convert to numpy and handle NaNs/Infs
        data = np.array(data)
        data = np.nan_to_num(data)
        
        return data

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        sample = self.samples[idx].copy()
        
        # Lazy loading: Load fMRI data on-demand (not stored in memory)
        # This prevents storing all tensors in memory, saving ~13GB RAM
        subject_id = sample['subject_id']
        story_name = sample['story_name']
        tr_start = sample['tr_start']
        tr_end = sample['tr_end']
        voxels = sample.get('voxels', None)
        
        # Load only this window's fMRI data using CACHED handle
        # Replaces load_fmri_data call which opens/closes file every time
        fmri_window = self._get_fmri_data_cached(
            subject_id, story_name, tr_start, tr_end, voxels
        )
        
        # Transpose to (voxels, window_size) for tokenization
        fmri_window = fmri_window.T  # (voxels, window_size)
        fmri = torch.from_numpy(fmri_window).float()

        
        result = {
            'time_series': fmri,  # (voxels, TRs) tensor
            'subject_id': sample['subject_id'],
            'story_name': sample['story_name'],
            'tr_start': sample['tr_start'],
            'tr_end': sample['tr_end'],
        }

        # If tokenizer is present, pre-tokenize the text
        if self.tokenizer is not None:
            pre_prompt = sample['pre_prompt']
            post_prompt = sample['post_prompt']
            answer = sample['answer']
            
            # Format: pre_prompt <image> post_prompt answer <|endofchunk|>
            if pre_prompt:
                text = f"{pre_prompt} <image> {post_prompt} {answer} <|endofchunk|>"
            else:
                text = f"<image> {post_prompt} {answer} <|endofchunk|>"
            
            # Tokenize
            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            
            # Calculate prompt length for labels
            prompt_text = f"{pre_prompt} <image> {post_prompt}" if pre_prompt else f"<image> {post_prompt}"
            prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            prompt_len = len(prompt_tokens)
            
            result['input_ids'] = tokens
            result['prompt_len'] = prompt_len
        else:
            # Fallback for compatibility (return raw text)
            result['pre_prompt'] = sample['pre_prompt']
            result['post_prompt'] = sample['post_prompt']
            result['answer'] = sample['answer']
            
        return result


def create_splits(
    subject_ids: Optional[List[str]] = None,
    story_names: Optional[List[str]] = None,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    random_seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits of stories.
    
    Returns:
        Tuple of (train_stories, val_stories, test_stories)
    """
    if story_names is None:
        if RESPDICT_PATH.exists():
            with open(RESPDICT_PATH, "r") as f:
                respdict = json.load(f)
            story_names = list(respdict.keys())
        else:
            raise ValueError("Cannot create splits: respdict.json not found")
    
    np.random.seed(random_seed)
    story_names = np.array(story_names)
    np.random.shuffle(story_names)
    
    n = len(story_names)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    
    train_stories = story_names[:n_train].tolist()
    val_stories = story_names[n_train:n_train + n_val].tolist()
    test_stories = story_names[n_train + n_val:].tolist()
    
    return train_stories, val_stories, test_stories


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing HuthFMRIDataset...")
    
    # Create a small test dataset
    dataset = HuthFMRIDataset(
        split="train",
        max_samples=10,
        window_size=10,
        stride=10,
    )
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n✅ Sample loaded successfully!")
        print(f"   Keys: {sample.keys()}")
        if 'input_ids' in sample:
            print(f"   Input IDs shape: {sample['input_ids'].shape}")
        else:
            print(f"   Pre-prompt: {sample.get('pre_prompt')}")
            print(f"   Answer: {sample.get('answer')}")
        print(f"   Time series shape: {len(sample['time_series'])} voxels × {len(sample['time_series'][0])} TRs")
        print(f"   Subject: {sample['subject_id']}, Story: {sample['story_name']}")
    else:
        print("❌ No samples loaded. Check data paths and TextGrid files.")
