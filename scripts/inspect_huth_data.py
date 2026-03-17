#!/usr/bin/env python3
"""
Data Inspection Script for Huth fMRI Dataset

This script inspects the structure and dimensions of the Huth fMRI dataset
to understand:
- .hf5 file structure and dimensions (TRs, voxels)
- TextGrid file availability
- Story-to-subject mappings
- Data statistics

Usage:
    python scripts/inspect_huth_data.py [--subject UTS01] [--story adollshouse]
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("⚠️  Warning: h5py not installed. Install with: pip install h5py")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  Warning: numpy not installed. Install with: pip install numpy")

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use same paths as config: data lives in project/data/Huth/
HUTH_DATA_DIR = PROJECT_ROOT / "data" / "Huth"
PREPROCESSED_DIR = HUTH_DATA_DIR / "derivative" / "preprocessed_data"
RESPDICT_PATH = HUTH_DATA_DIR / "derivative" / "respdict.json"


def find_hf5_files(base_dir: Path) -> Dict[str, List[Path]]:
    """Find all .hf5 files organized by subject."""
    files_by_subject = {}
    
    if not base_dir.exists():
        print(f"⚠️  Directory not found: {base_dir}")
        return files_by_subject
    
    for subject_dir in base_dir.iterdir():
        if subject_dir.is_dir():
            subject_id = subject_dir.name
            hf5_files = list(subject_dir.glob("*.hf5"))
            if hf5_files:
                files_by_subject[subject_id] = sorted(hf5_files)
    
    return files_by_subject


def inspect_hf5_file(file_path: Path) -> Dict:
    """Inspect a single .hf5 file and return metadata."""
    info = {
        "file": str(file_path.name),
        "exists": False,
        "shape": None,
        "dtype": None,
        "keys": None,
        "size_mb": None,
        "stats": None
    }
    
    if not file_path.exists():
        return info
    
    # File exists - mark as found
    info["exists"] = True
    info["size_mb"] = file_path.stat().st_size / (1024 * 1024)
    
    # Check if it's a symlink
    if file_path.is_symlink():
        info["is_symlink"] = True
        # Check if symlink target exists
        try:
            target = file_path.resolve()
            if not target.exists():
                info["symlink_broken"] = True
                info["error"] = "Symlink exists but target not downloaded"
                return info
        except Exception:
            info["symlink_broken"] = True
            info["error"] = "Symlink target cannot be resolved"
            return info
    
    if not HAS_H5PY or not HAS_NUMPY:
        info["error"] = "Missing dependencies: h5py and/or numpy required to read file contents"
        return info
    
    try:
        
        with h5py.File(file_path, "r") as hf:
            # Get all keys in the file
            info["keys"] = list(hf.keys())
            
            # Check for 'data' key (expected format)
            if "data" in hf.keys():
                data = hf["data"]
                info["shape"] = data.shape
                info["dtype"] = str(data.dtype)
                
                # Load a small sample for statistics (first 100 TRs)
                sample_size = min(100, data.shape[0])
                sample = data[:sample_size, :]
                
                info["stats"] = {
                    "mean": float(np.mean(sample)),
                    "std": float(np.std(sample)),
                    "min": float(np.min(sample)),
                    "max": float(np.max(sample)),
                    "nan_count": int(np.isnan(sample).sum()),
                    "inf_count": int(np.isinf(sample).sum()),
                }
            else:
                # If no 'data' key, report all datasets
                info["keys"] = list(hf.keys())
                if len(hf.keys()) > 0:
                    first_key = list(hf.keys())[0]
                    info["shape"] = hf[first_key].shape
                    info["dtype"] = str(hf[first_key].dtype)
    
    except Exception as e:
        info["error"] = str(e)
    
    return info


def load_respdict() -> Dict[str, int]:
    """Load the respdict.json which maps stories to TR counts."""
    if not RESPDICT_PATH.exists():
        print(f"⚠️  respdict.json not found at: {RESPDICT_PATH}")
        return {}
    
    with open(RESPDICT_PATH, "r") as f:
        return json.load(f)


def find_textgrid_files() -> Dict[str, Path]:
    """Find TextGrid files (if they exist in the dataset)."""
    textgrid_files = {}
    
    # Check common locations
    possible_locations = [
        HUTH_DATA_DIR / "derivative" / "textgrids",
        HUTH_DATA_DIR / "textgrids",
        PROJECT_ROOT / "semantic-decoding" / "data_train" / "train_stimulus",
    ]
    
    for location in possible_locations:
        if location.exists():
            for tg_file in location.glob("*.TextGrid"):
                story_name = tg_file.stem
                textgrid_files[story_name] = tg_file
            if textgrid_files:
                break
    
    return textgrid_files


def inspect_subject(subject_id: str, story_name: Optional[str] = None) -> None:
    """Inspect data for a specific subject and optionally a specific story."""
    print(f"\n{'='*80}")
    print(f"INSPECTING SUBJECT: {subject_id}")
    print(f"{'='*80}\n")
    
    subject_dir = PREPROCESSED_DIR / subject_id
    if not subject_dir.exists():
        print(f"❌ Subject directory not found: {subject_dir}")
        return
    
    hf5_files = list(subject_dir.glob("*.hf5"))
    if not hf5_files:
        print(f"⚠️  No .hf5 files found in {subject_dir}")
        return
    
    print(f"📁 Found {len(hf5_files)} .hf5 files\n")
    
    # If specific story requested, only inspect that
    if story_name:
        story_file = subject_dir / f"{story_name}.hf5"
        if story_file.exists():
            hf5_files = [story_file]
        else:
            print(f"❌ Story file not found: {story_file}")
            return
    
    # Inspect each file
    for hf5_file in hf5_files:
        info = inspect_hf5_file(hf5_file)
        
        print(f"📄 File: {info['file']}")
        if not info["exists"]:
            print("   ❌ File not found\n")
            continue
        
        # Show symlink status
        if info.get("is_symlink"):
            if info.get("symlink_broken"):
                print(f"   ⚠️  Symlink (broken - target not downloaded)")
                print(f"      Run: datalad get {subject_dir.name}/{info['file']}")
            else:
                print(f"   🔗 Symlink (downloaded)")
        
        print(f"   Size: {info['size_mb']:.2f} MB")
        
        if "error" in info:
            print(f"   ⚠️  {info['error']}")
            if "h5py" in info["error"].lower():
                print(f"      Install with: pip install h5py numpy")
            print()
            continue
        
        if info.get("keys"):
            print(f"   Keys: {info['keys']}")
        
        if info["shape"]:
            trs, voxels = info["shape"]
            print(f"   Shape: {info['shape']} → {trs} TRs × {voxels} voxels")
            print(f"   Dtype: {info['dtype']}")
            
            if info["stats"]:
                stats = info["stats"]
                print(f"   Statistics (sample):")
                print(f"      Mean: {stats['mean']:.4f}")
                print(f"      Std:  {stats['std']:.4f}")
                print(f"      Min:  {stats['min']:.4f}")
                print(f"      Max:  {stats['max']:.4f}")
                if stats["nan_count"] > 0:
                    print(f"      ⚠️  NaN count: {stats['nan_count']}")
                if stats["inf_count"] > 0:
                    print(f"      ⚠️  Inf count: {stats['inf_count']}")
        
        if "error" in info:
            print(f"   ❌ Error: {info['error']}")
        
        print()


def print_summary() -> None:
    """Print overall dataset summary."""
    print(f"\n{'='*80}")
    print("DATASET SUMMARY")
    print(f"{'='*80}\n")
    
    # Find all subjects and files
    files_by_subject = find_hf5_files(PREPROCESSED_DIR)
    
    if not files_by_subject:
        print("❌ No .hf5 files found in preprocessed_data directory")
        print(f"   Expected location: {PREPROCESSED_DIR}")
        return
    
    print(f"📊 Found {len(files_by_subject)} subjects with .hf5 files:\n")
    
    total_files = 0
    total_size_mb = 0
    
    for subject_id, files in sorted(files_by_subject.items()):
        print(f"  {subject_id}: {len(files)} files")
        total_files += len(files)
        
        # Get total size for this subject
        subject_size = sum(f.stat().st_size for f in files if f.exists())
        total_size_mb += subject_size / (1024 * 1024)
    
    print(f"\n📦 Total: {total_files} files, {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    
    # Load respdict
    respdict = load_respdict()
    if respdict:
        print(f"\n📖 Stories in respdict.json: {len(respdict)} stories")
        print(f"   TR counts range: {min(respdict.values())} - {max(respdict.values())} TRs")
    
    # Check for TextGrid files
    textgrid_files = find_textgrid_files()
    if textgrid_files:
        print(f"\n📝 TextGrid files found: {len(textgrid_files)} files")
        print(f"   Location: {textgrid_files[list(textgrid_files.keys())[0]].parent}")
    else:
        print(f"\n⚠️  TextGrid files not found")
        print(f"   Checked locations:")
        for loc in [HUTH_DATA_DIR / "derivative" / "textgrids",
                    HUTH_DATA_DIR / "textgrids",
                    PROJECT_ROOT / "semantic-decoding" / "data_train" / "train_stimulus"]:
            print(f"      - {loc}")
    
    print()


def main():
    """Main inspection function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect Huth fMRI dataset")
    parser.add_argument("--subject", type=str, help="Subject ID (e.g., UTS01)")
    parser.add_argument("--story", type=str, help="Story name (e.g., adollshouse)")
    parser.add_argument("--summary-only", action="store_true", 
                       help="Only show summary, don't inspect individual files")
    
    args = parser.parse_args()
    
    print("🔍 Huth fMRI Dataset Inspection")
    print(f"Brain-model-alignment directory: {PROJECT_ROOT}")
    print(f"Data directory: {HUTH_DATA_DIR}")
    print(f"Preprocessed directory: {PREPROCESSED_DIR}")
    
    if args.summary_only:
        print_summary()
        return
    
    if args.subject:
        inspect_subject(args.subject, args.story)
    else:
        print_summary()
        
        # Show first subject as example
        files_by_subject = find_hf5_files(PREPROCESSED_DIR)
        if files_by_subject:
            first_subject = sorted(files_by_subject.keys())[0]
            print(f"\n💡 Example: Inspecting first subject ({first_subject})")
            inspect_subject(first_subject)


if __name__ == "__main__":
    main()
