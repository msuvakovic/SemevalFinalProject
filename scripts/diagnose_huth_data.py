#!/usr/bin/env python3
"""
Diagnose why the Huth dataset loads 0 samples.
Uses the same config and paths as the dataset. Run from project root:
    python scripts/diagnose_huth_data.py
"""

import sys
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Use same config as dataset
import importlib.util
spec = importlib.util.spec_from_file_location("config", PROJECT_ROOT / "config.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

PREPROCESSED_DATA_DIR = config.PREPROCESSED_DATA_DIR
RESPDICT_PATH = config.RESPDICT_PATH
TEXTGRID_DIRS = config.TEXTGRID_DIRS
get_fmri_path = config.get_fmri_path
get_textgrid_path = config.get_textgrid_path

def main():
    print("=" * 60)
    print("HUTH DATA DIAGNOSIS")
    print("=" * 60)
    
    # 1) Paths
    print("\n1) CONFIG PATHS (what the dataset uses)")
    print(f"   PREPROCESSED_DATA_DIR: {PREPROCESSED_DATA_DIR}")
    print(f"   exists: {PREPROCESSED_DATA_DIR.exists()}")
    print(f"   RESPDICT_PATH: {RESPDICT_PATH}")
    print(f"   exists: {RESPDICT_PATH.exists()}")
    print(f"   TEXTGRID_DIRS[0]: {TEXTGRID_DIRS[0]}")
    print(f"   exists: {TEXTGRID_DIRS[0].exists()}")
    
    # 2) respdict vs actual files
    if not RESPDICT_PATH.exists():
        print("\n   FAIL: respdict.json not found. Put it at the path above.")
        return
    with open(RESPDICT_PATH) as f:
        respdict = json.load(f)
    respdict_stories = list(respdict.keys())
    print(f"\n2) RESPDICT: {len(respdict)} stories. First 5: {respdict_stories[:5]}")
    
    # 2b) Actual .hf5 and .TextGrid filenames (all of them for overlap; show first 10)
    hf5_dir = PREPROCESSED_DATA_DIR / "UTS01"
    tg_dir = TEXTGRID_DIRS[0]
    hf5_files_all = list(hf5_dir.glob("*.hf5")) if hf5_dir.exists() else []
    tg_files_all = list(tg_dir.glob("*.TextGrid")) if tg_dir.exists() else []
    hf5_stems_all = set(f.stem for f in hf5_files_all)
    tg_stems_all = set(f.stem for f in tg_files_all)
    # Case/normalized overlap (same logic as dataset)
    def _norm(s):
        return s.lower().replace("_", "")
    overlap_exact = hf5_stems_all & tg_stems_all
    overlap_norm = set()
    for h in hf5_stems_all:
        for t in tg_stems_all:
            if h.lower() == t.lower() or _norm(h) == _norm(t):
                overlap_norm.add(h)
                break
    overlap = overlap_exact or overlap_norm
    n_overlap = len(overlap)
    print(f"\n2b) ACTUAL FILES ON DISK")
    print(f"   UTS01/*.hf5: {len(hf5_stems_all)} files. First 5 stems: {sorted(hf5_stems_all)[:5]}")
    print(f"   TextGrids/*.TextGrid: {len(tg_stems_all)} files. First 5 stems: {sorted(tg_stems_all)[:5]}")
    print(f"   Overlap (stories with BOTH .hf5 and .TextGrid): {n_overlap}")
    if respdict_stories and hf5_stems_all:
        match = respdict_stories[0] in hf5_stems_all
        print(f"   respdict first story {respdict_stories[0]!r} in UTS01 .hf5? {match}")
    if respdict_stories and tg_stems_all:
        match = respdict_stories[0] in tg_stems_all
        print(f"   respdict first story {respdict_stories[0]!r} in TextGrids? {match}")
    
    # 3) .hf5 for first (subject, story) — use first story that exists in BOTH .hf5 and .TextGrid
    subject_id = "UTS01"
    story_name = respdict_stories[0]
    if overlap:
        story_name = sorted(overlap)[0]
    elif hf5_stems_all:
        story_name = sorted(hf5_stems_all)[0]
    try:
        fmri_path = get_fmri_path(subject_id, story_name)
    except FileNotFoundError:
        print(f"\n3) fMRI FILE for {subject_id}/{story_name}: get_fmri_path FAILED (no .hf5)")
        fmri_path = None
    if fmri_path:
        print(f"\n3) fMRI FILE for {subject_id}/{story_name}")
        print(f"   path: {fmri_path}")
        print(f"   exists: {fmri_path.exists()}")
        if fmri_path.exists():
            import h5py
            with h5py.File(fmri_path, "r") as hf:
                sh = hf["data"].shape
            print(f"   shape: {sh}")
    
    # 4) TextGrid path and file (try config path, then scan dir like dataset fallback)
    print(f"\n4) TEXTGRID for {story_name}")
    tg_path = None
    try:
        tg_path = get_textgrid_path(story_name)
        print(f"   get_textgrid_path -> {tg_path}")
        print(f"   exists: {tg_path.exists()}")
    except FileNotFoundError as e:
        print(f"   get_textgrid_path FAILED: {e}")
    if tg_path is None or not tg_path.exists():
        # Fallback: find by normalized name (same as dataset)
        def _norm(s):
            return s.lower().replace("_", "")
        story_norm = _norm(story_name)
        for d in TEXTGRID_DIRS:
            if not d.exists():
                continue
            for p in d.glob("*.TextGrid"):
                if _norm(p.stem) == story_norm:
                    print(f"   Fallback found: {p}")
                    print(f"   exists: {p.exists()}")
                    try:
                        with open(p, "r", encoding="utf-8", errors="replace") as f:
                            _ = f.read()
                        print(f"   readable: True (opened successfully)")
                    except Exception as e:
                        print(f"   readable: False ({e})")
                    tg_path = p
                    break
            if tg_path is not None:
                break
        if tg_path is None:
            print("   No matching .TextGrid found in any dir.")
    
    # 5) Try to parse TextGrid and list tiers
    print("\n5) TEXTGRID PARSER (same as dataset)")
    try:
        # Same path setup as dataset
        decoding_dir = PROJECT_ROOT / "decoding"
        if str(decoding_dir) not in sys.path:
            sys.path.insert(0, str(decoding_dir))
        from utils_ridge.textgrid import TextGrid
        print("   TextGrid import: OK")
    except ImportError as e:
        print(f"   TextGrid import FAILED: {e}")
        return
    
    if tg_path and tg_path.exists():
        try:
            with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            tg = TextGrid(content)
            print(f"   Tiers: {len(tg.tiers)}")
            for i, tier in enumerate(tg.tiers):
                name = getattr(tier, "nameid", "?")
                n = len(getattr(tier, "simple_transcript", []))
                print(f"      tier {i}: nameid={name!r}, simple_transcript length={n}")
            if tg.tiers:
                t0 = tg.tiers[0]
                st = getattr(t0, "simple_transcript", [])
                if st:
                    print(f"   First 3 entries (tier 0): {st[:3]}")
        except Exception as e:
            print(f"   Parse FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   Skipped (no TextGrid path or file missing)")
    
    # 6) Quick sample count for one (subject, story)
    print("\n6) WOULD WE GET SAMPLES for one (subject, story)?")
    if tg_path and tg_path.exists():
        try:
            with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
                tg = TextGrid(f.read())
            words = []
            for tier in tg.tiers:
                tier_name = (getattr(tier, "nameid", None) or "").lower()
                if tier_name in ["words", "word", "transcript"]:
                    for start, end, text in getattr(tier, "simple_transcript", []):
                        if (text or "").strip():
                            words.append((text.strip(), float(start), float(end)))
                    break
            if not words:
                for tier in tg.tiers:
                    st = getattr(tier, "simple_transcript", None) or []
                    if len(st) > 0:
                        for start, end, text in st:
                            if (text or "").strip():
                                words.append((text.strip(), float(start), float(end)))
                        break
            print(f"   Words extracted: {len(words)}")
            if words:
                print(f"   First 3: {words[:3]}")
            else:
                print("   -> No words => no windows get a (context, target) => 0 samples.")
        except Exception as e:
            print(f"   Error: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
