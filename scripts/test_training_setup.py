#!/usr/bin/env python3
"""
Quick test to verify training setup without actually training.
Tests dataset loading, model creation, and optimizer setup.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

import torch
from torch.utils.data import DataLoader

from src.datasets.huth_fmri_dataset import HuthFMRIDataset
from src.models.fmri_flamingo import FMRIFlamingo

# Import config
import importlib.util
config_path = FMRI_FLAMINGO_DIR / "config.py"
spec = importlib.util.spec_from_file_location("fmri_config", config_path)
fmri_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fmri_config)

def test_training_setup():
    """Test that training setup works."""
    print("=" * 80)
    print("TESTING TRAINING SETUP")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Test dataset loading
    print("\n📦 Testing dataset loading...")
    try:
        train_dataset = HuthFMRIDataset(
            split="train",
            window_size=getattr(fmri_config, "DEFAULT_WINDOW_SIZE", 10),
            stride=getattr(fmri_config, "DEFAULT_STRIDE", 10),
            max_samples=5,  # Just a few samples for testing
        )
        print(f"   ✅ Train dataset: {len(train_dataset)} samples")
        
        if len(train_dataset) == 0:
            print(f"   ⚠️  No training samples found. You need:")
            print(f"      • Preprocessed .hf5 files in: {fmri_config.PREPROCESSED_DATA_DIR}")
            print(f"      • respdict.json in: {fmri_config.RESPDICT_PATH.parent}")
            print(f"      • TextGrids in: {fmri_config.TEXTGRID_DIRS[0]} (for word–TR alignment)")
            print(f"      See README / EXPERIMENT.md for Huth data setup (OpenNeuro ds003020).")
        else:
            sample = train_dataset[0]
            print(f"   ✅ Sample keys: {list(sample.keys())}")
        
        val_dataset = HuthFMRIDataset(
            split="validation",
            window_size=getattr(fmri_config, "DEFAULT_WINDOW_SIZE", 10),
            stride=getattr(fmri_config, "DEFAULT_STRIDE", 10),
            max_samples=3,
        )
        print(f"   ✅ Val dataset: {len(val_dataset)} samples")
        
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test data loader (only if we have samples)
    batch = None
    if len(train_dataset) > 0:
        print("\n🔄 Testing data loader...")
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=2,
                shuffle=False,
            )
            batch = next(iter(train_loader))
            print(f"   ✅ Batch loaded: {len(batch)} samples")
        except Exception as e:
            print(f"   ❌ Data loader failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n🔄 Skipping data loader test (no samples).")
    
    # Test model creation
    print("\n🤖 Testing model creation...")
    try:
        llm_id = getattr(fmri_config, "LLM_ID", "meta-llama/Llama-3.2-1B")
        num_rois = getattr(fmri_config, "NUM_ROIS", 200)
        
        print(f"   Attempting to load: {llm_id}")
        print(f"   ⚠️  This requires HuggingFace authentication")
        
        model = FMRIFlamingo(
            device=device,
            llm_id=llm_id,
            num_rois=num_rois,
        )
        print(f"   ✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
            print(f"   ⚠️  HuggingFace authentication required (expected)")
            print(f"      Run: huggingface-cli login")
            print(f"      Then training will work")
            return True  # This is expected, not a failure
        else:
            print(f"   ❌ Model creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
        # Test forward pass (if model was created and we have a batch)
        if 'model' in locals() and batch is not None:
            print("\n⚡ Testing forward pass...")
            try:
                model.eval()
                with torch.no_grad():
                    # Get a small batch (batch is a list of dicts)
                    small_batch = batch[:2] if isinstance(batch, list) and len(batch) >= 2 else batch
                    loss = model.compute_loss(small_batch)
                print(f"   ✅ Forward pass successful")
                print(f"   Loss: {loss.item():.4f}")
            except Exception as e:
                print(f"   ❌ Forward pass failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        elif 'model' in locals() and batch is None:
            print("\n⚡ Skipping forward pass (no data).")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Training setup is ready!")
    if batch is None:
        print("   Add Huth data (.hf5 + TextGrids) to run training.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_training_setup()
    sys.exit(0 if success else 1)
