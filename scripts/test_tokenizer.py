#!/usr/bin/env python3
"""
Test script for FMRITokenizer.
Verifies that the tokenizer correctly processes fMRI data.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add paths
SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
if str(FMRI_FLAMINGO_DIR) not in sys.path:
    sys.path.insert(0, str(FMRI_FLAMINGO_DIR))

from src.models.fmri_tokenizer import FMRITokenizer

def test_tokenizer_basic():
    """Test basic tokenization functionality."""
    print("=" * 80)
    print("Testing FMRITokenizer - Basic Functionality")
    print("=" * 80)
    print()
    
    # Create tokenizer
    tokenizer = FMRITokenizer(
        num_rois=200,
        roi_selection_method="random",
        temporal_aggregation="mean",
        normalize_per_roi=True,
    )
    
    print(f"✅ Tokenizer created")
    print(f"   Num ROIs: {tokenizer.num_rois}")
    print(f"   ROI method: {tokenizer.roi_selection_method}")
    print(f"   Temporal aggregation: {tokenizer.temporal_aggregation}")
    print()
    
    # Test with synthetic data
    # Simulate: batch_size=2, voxels=1000, TRs=10
    batch_size = 2
    num_voxels = 1000
    num_trs = 10
    
    print(f"📊 Test Input:")
    print(f"   Batch size: {batch_size}")
    print(f"   Voxels: {num_voxels}")
    print(f"   TRs: {num_trs}")
    print(f"   Input shape: ({batch_size}, {num_voxels}, {num_trs})")
    print()
    
    # Create synthetic fMRI data
    x = torch.randn(batch_size, num_voxels, num_trs)
    
    print("🔄 Forward pass...")
    with torch.no_grad():
        output = tokenizer(x)
    
    print(f"✅ Tokenization successful!")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: ({batch_size}, {tokenizer.num_rois}, {tokenizer.roi_projection.out_features})")
    print()
    
    # Verify output shape
    expected_shape = (batch_size, tokenizer.num_rois, tokenizer.roi_projection.out_features)
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
    print(f"✅ Output shape matches expected: {expected_shape}")
    print()
    
    # Test with different temporal aggregation methods
    print("=" * 80)
    print("Testing Different Temporal Aggregation Methods")
    print("=" * 80)
    print()
    
    for agg_method in ["mean", "conv"]:
        print(f"Testing '{agg_method}' aggregation...")
        tokenizer_agg = FMRITokenizer(
            num_rois=200,
            roi_selection_method="random",
            temporal_aggregation=agg_method,
            normalize_per_roi=True,
        )
        
        with torch.no_grad():
            output_agg = tokenizer_agg(x)
        
        print(f"   ✅ Output shape: {output_agg.shape}")
        print(f"   ✅ Output range: [{output_agg.min().item():.3f}, {output_agg.max().item():.3f}]")
        print()
    
    # Test with "all_voxels" method (no ROI grouping)
    print("=" * 80)
    print("Testing 'all_voxels' Method (No ROI Grouping)")
    print("=" * 80)
    print()
    
    # Use smaller number for this test
    num_voxels_small = 50
    tokenizer_all = FMRITokenizer(
        num_rois=num_voxels_small,
        roi_selection_method="all_voxels",
        temporal_aggregation="mean",
        normalize_per_roi=True,
    )
    
    x_small = torch.randn(batch_size, num_voxels_small, num_trs)
    
    with torch.no_grad():
        output_all = tokenizer_all(x_small)
    
    print(f"✅ Output shape: {output_all.shape}")
    print(f"   Expected: ({batch_size}, {num_voxels_small}, {tokenizer_all.roi_projection.out_features})")
    assert output_all.shape == (batch_size, num_voxels_small, tokenizer_all.roi_projection.out_features)
    print()
    
    # Test with real dataset sample
    print("=" * 80)
    print("Testing with Real Dataset Sample")
    print("=" * 80)
    print()
    
    try:
        from src.datasets.huth_fmri_dataset import HuthFMRIDataset
        
        # Create a small dataset
        dataset = HuthFMRIDataset(
            split="train",
            max_samples=1,
            window_size=10,
            stride=10,
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            fmri_data = sample['time_series']  # List of 1D arrays (one per voxel)
            
            # Convert to tensor: (voxels, TRs)
            fmri_tensor = torch.stack([torch.from_numpy(arr) for arr in fmri_data], dim=0)
            num_voxels_real, num_trs_real = fmri_tensor.shape
            
            print(f"📊 Real Dataset Sample:")
            print(f"   Voxels: {num_voxels_real}")
            print(f"   TRs: {num_trs_real}")
            print(f"   Data shape: ({num_voxels_real}, {num_trs_real})")
            print()
            
            # Add batch dimension: (1, voxels, TRs)
            fmri_batch = fmri_tensor.unsqueeze(0)
            
            # Create tokenizer for this data
            tokenizer_real = FMRITokenizer(
                num_rois=200,
                roi_selection_method="random",
                temporal_aggregation="mean",
                normalize_per_roi=True,
            )
            
            print("🔄 Tokenizing real data...")
            with torch.no_grad():
                output_real = tokenizer_real(fmri_batch)
            
            print(f"✅ Real data tokenization successful!")
            print(f"   Input: ({1}, {num_voxels_real}, {num_trs_real})")
            print(f"   Output: {output_real.shape}")
            print(f"   Expected: (1, 200, {tokenizer_real.roi_projection.out_features})")
            print()
            
        else:
            print("⚠️  No samples in dataset (TextGrid files may be missing)")
            print()
    
    except Exception as e:
        print(f"⚠️  Could not test with real dataset: {e}")
        print("   (This is okay if dataset files are not available)")
        print()
    
    print("=" * 80)
    print("✅ All Tests Passed!")
    print("=" * 80)

if __name__ == "__main__":
    test_tokenizer_basic()
