#!/usr/bin/env python3
"""
Quick test script for dataset loader.
Can be run from any directory.
"""

import sys
from pathlib import Path

# Get the fmri-flamingo directory (parent of scripts directory)
SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent

# Add fmri-flamingo directory to path
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

# Also add semantic-decoding for TextGrid parser
BASE_DIR = FMRI_FLAMINGO_DIR.parent
SEMANTIC_DECODING_DIR = BASE_DIR / "semantic-decoding" / "decoding"
if str(SEMANTIC_DECODING_DIR) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_DECODING_DIR))

try:
    from src.datasets.huth_fmri_dataset import HuthFMRIDataset
    
    print("Testing HuthFMRIDataset...")
    print("=" * 60)
    
    # Create a small test dataset
    dataset = HuthFMRIDataset(
        split="train",
        max_samples=5,
        window_size=10,
        stride=10,
    )
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n📊 Sample 0:")
        print(f"   Keys: {list(sample.keys())}")
        print(f"   Pre-prompt: {sample['pre_prompt'][:50]}...")
        print(f"   Post-prompt: {sample['post_prompt']}")
        print(f"   Answer: {sample['answer']}")
        print(f"   Time series: {len(sample['time_series'])} voxels × {len(sample['time_series'][0])} TRs")
        print(f"   Subject: {sample['subject_id']}, Story: {sample['story_name']}")
        print(f"   TR range: {sample['tr_start']} - {sample['tr_end']}")
    else:
        print("\n⚠️  No samples loaded. Possible issues:")
        print("   - TextGrid files not found (word alignment required)")
        print("   - No words aligned with fMRI windows")
        print("   - Data files not fully downloaded")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're running from fmri-flamingo directory")
    print("2. Check that config.py exists")
    print("3. Verify Python path includes fmri-flamingo directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
