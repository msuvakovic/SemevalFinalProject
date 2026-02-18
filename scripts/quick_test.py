#!/usr/bin/env python3
"""Quick test to verify files are downloaded."""

from pathlib import Path

# Path to data
data_dir = Path(__file__).parent.parent.parent / "data" / "Huth" / "derivative" / "preprocessed_data"

print(f"Checking data directory: {data_dir}")
print(f"Directory exists: {data_dir.exists()}\n")

if data_dir.exists():
    # Check UTS01
    uts01_dir = data_dir / "UTS01"
    if uts01_dir.exists():
        hf5_files = list(uts01_dir.glob("*.hf5"))
        print(f"UTS01: Found {len(hf5_files)} .hf5 files")
        
        # Check a specific file
        test_file = uts01_dir / "adollshouse.hf5"
        print(f"\nTesting: {test_file.name}")
        print(f"  Exists: {test_file.exists()}")
        print(f"  Is symlink: {test_file.is_symlink() if test_file.exists() else 'N/A'}")
        
        if test_file.exists():
            if test_file.is_symlink():
                try:
                    target = test_file.resolve()
                    print(f"  Symlink target: {target}")
                    print(f"  Target exists: {target.exists()}")
                    if target.exists():
                        print(f"  Target size: {target.stat().st_size / (1024*1024):.2f} MB")
                except Exception as e:
                    print(f"  Error resolving symlink: {e}")
            else:
                print(f"  File size: {test_file.stat().st_size / (1024*1024):.2f} MB")
            
            # Try to check if h5py can open it
            try:
                import h5py
                with h5py.File(test_file, "r") as f:
                    print(f"  ✅ Can open with h5py")
                    if "data" in f.keys():
                        print(f"  Data shape: {f['data'].shape}")
            except ImportError:
                print(f"  ⚠️  h5py not installed (cannot verify file contents)")
            except Exception as e:
                print(f"  ❌ Error opening file: {e}")
        else:
            print(f"  ❌ File not found at: {test_file}")
    else:
        print(f"UTS01 directory not found: {uts01_dir}")
else:
    print(f"Data directory not found: {data_dir}")
