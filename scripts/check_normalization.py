import h5py
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append("/home/dmarhoef/brain-model-alignment")
from config import PREPROCESSED_DATA_DIR, TRAIN_SUBJECTS

def check_normalization():
    subject = TRAIN_SUBJECTS[0]
    subj_dir = PREPROCESSED_DATA_DIR / subject
    hf5_files = list(subj_dir.glob("*.hf5"))
    
    if not hf5_files:
        print(f"No files found for {subject}")
        return

    fpath = hf5_files[0]
    print(f"Checking {fpath}...")
    
    with h5py.File(fpath, "r") as f:
        data = f["data"][:]
        # Check mean and std of a few random voxels
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        
        print(f"Data shape: {data.shape}")
        print(f"Global Mean: {np.mean(data):.4f}")
        print(f"Global Std: {np.std(data):.4f}")
        print(f"Voxel Means (first 5): {means[:5]}")
        print(f"Voxel Stds (first 5): {stds[:5]}")
        
        is_centered = np.allclose(means, 0, atol=1e-1)
        is_unit_var = np.allclose(stds, 1, atol=1e-1)
        
        if is_centered and is_unit_var:
            print("✅ Data appears to be Z-scored (mean~0, std~1).")
        else:
            print("⚠️  Data is NOT Z-scored. Raw BOLD values likely.")

if __name__ == "__main__":
    check_normalization()
