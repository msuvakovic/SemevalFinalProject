#!/usr/bin/env python3
"""
Create Huth data directory structure and report what's present vs missing.

Run from project root:
    python scripts/setup_huth_data.py

See data/Huth/README.md for where to get .hf5, respdict.json, and TextGrids.
"""

import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def main():
    print("=" * 60)
    print("Huth data setup")
    print("=" * 60)
    print(f"\nExpected base dir: {config.DATA_DIR}")
    print(f"  (from config: FMRI_FLAMINGO_DIR / 'data' / 'Huth')")
    
    # Create directories
    dirs_to_create = [
        config.DATA_DIR,
        config.PREPROCESSED_DATA_DIR,
        config.DATA_DIR / "derivative" / "TextGrids",
        config.DATA_DIR / "derivative" / "textgrids",
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [dir] {d}")
    
    # Subject dirs (empty placeholders)
    for i in range(1, 10):
        subj_dir = config.PREPROCESSED_DATA_DIR / f"UTS{i:02d}"
        subj_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Subject dirs: {config.PREPROCESSED_DATA_DIR}/UTS01 .. UTS09")
    
    # Check respdict.json
    print("\n--- respdict.json ---")
    if config.RESPDICT_PATH.exists():
        try:
            with open(config.RESPDICT_PATH) as f:
                respdict = json.load(f)
            n_stories = len(respdict)
            print(f"  OK  {config.RESPDICT_PATH}")
            print(f"      {n_stories} stories listed")
        except Exception as e:
            print(f"  FAIL  {config.RESPDICT_PATH}: {e}")
    else:
        print(f"  MISSING  {config.RESPDICT_PATH}")
        print("      Add a JSON file mapping story name -> TR count, e.g. {\"adollshouse\": 177, ...}")
    
    # Check preprocessed_data/UTS0x/*.hf5
    print("\n--- Preprocessed .hf5 files ---")
    total_hf5 = 0
    for i in range(1, 10):
        subj = f"UTS{i:02d}"
        subj_dir = config.PREPROCESSED_DATA_DIR / subj
        if subj_dir.exists():
            hf5_list = list(subj_dir.glob("*.hf5"))
            n = len(hf5_list)
            total_hf5 += n
            if n > 0:
                print(f"  OK  {subj}/  ({n} .hf5 files)")
            else:
                print(f"  --  {subj}/  (no .hf5 files yet)")
        else:
            print(f"  --  {subj}/  (dir missing)")
    
    if total_hf5 == 0:
        print("\n  No .hf5 files found. Put preprocessed BOLD files in:")
        print(f"      {config.PREPROCESSED_DATA_DIR}/UTS01/<story>.hf5  etc.")
        print("  See data/Huth/README.md for OpenNeuro / datalad / semantic-decoding.")
    else:
        print(f"\n  Total .hf5 files: {total_hf5}")
    
    # Check TextGrids
    print("\n--- TextGrids ---")
    textgrid_dir = config.DATA_DIR / "derivative" / "TextGrids"
    textgrid_dir_lower = config.DATA_DIR / "derivative" / "textgrids"
    tg_dirs = [textgrid_dir, textgrid_dir_lower]
    found_tg = False
    for tg_dir in tg_dirs:
        if tg_dir.exists():
            tg_files = list(tg_dir.glob("*.TextGrid"))
            if tg_files:
                print(f"  OK  {tg_dir}  ({len(tg_files)} .TextGrid files)")
                found_tg = True
                break
    if not found_tg:
        print(f"  MISSING  No .TextGrid files in {textgrid_dir} or {textgrid_dir_lower}")
        print("      Add word-timing files for each story. See data/Huth/README.md.")
    
    # Summary
    print("\n" + "=" * 60)
    if config.RESPDICT_PATH.exists() and total_hf5 > 0 and found_tg:
        print("Data looks ready. Run: python scripts/test_training_setup.py")
    else:
        print("Next steps:")
        if not config.RESPDICT_PATH.exists():
            print("  1. Add respdict.json (story name -> TR count)")
        if total_hf5 == 0:
            print("  2. Add preprocessed .hf5 files (see data/Huth/README.md)")
        if not found_tg:
            print("  3. Add TextGrids for word timing (see data/Huth/README.md)")
        print("\n  Full instructions: data/Huth/README.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
