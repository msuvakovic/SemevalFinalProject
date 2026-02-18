#!/usr/bin/env python3
"""
Preflight check script for training stability.

Reads system resources and warns if insufficient for training.
Does NOT modify anything - read-only diagnostics.

Exit codes:
  0: System ready for training
  1: Critical resource shortage detected
"""

import sys
import psutil
import shutil
from pathlib import Path

# Minimum requirements
MIN_RAM_GB = 6.0  # Minimum available RAM in GB
MIN_DISK_GB = 20.0  # Minimum free disk space in GB
WARN_RAM_GB = 8.0  # Warning threshold for RAM

def check_ram():
    """Check available RAM."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)
    used_gb = mem.used / (1024**3)
    
    print(f"💾 RAM:")
    print(f"   Total: {total_gb:.2f} GB")
    print(f"   Used: {used_gb:.2f} GB")
    print(f"   Available: {available_gb:.2f} GB")
    
    if available_gb < MIN_RAM_GB:
        print(f"   ❌ CRITICAL: Available RAM ({available_gb:.2f} GB) < minimum ({MIN_RAM_GB} GB)")
        return False
    elif available_gb < WARN_RAM_GB:
        print(f"   ⚠️  WARNING: Available RAM ({available_gb:.2f} GB) < recommended ({WARN_RAM_GB} GB)")
    else:
        print(f"   ✅ Available RAM sufficient")
    
    return True

def check_swap():
    """Check swap usage."""
    swap = psutil.swap_memory()
    total_gb = swap.total / (1024**3)
    used_gb = swap.used / (1024**3)
    
    print(f"\n💿 Swap:")
    print(f"   Total: {total_gb:.2f} GB")
    print(f"   Used: {used_gb:.2f} GB")
    
    if used_gb > 0.1:  # More than 100MB used
        print(f"   ⚠️  WARNING: Swap is being used (system may be under memory pressure)")
    else:
        print(f"   ✅ Swap not in use")
    
    return True

def check_disk():
    """Check available disk space."""
    disk = shutil.disk_usage(Path.cwd())
    free_gb = disk.free / (1024**3)
    total_gb = disk.total / (1024**3)
    used_gb = (disk.total - disk.free) / (1024**3)
    
    print(f"\n💽 Disk:")
    print(f"   Total: {total_gb:.2f} GB")
    print(f"   Used: {used_gb:.2f} GB")
    print(f"   Free: {free_gb:.2f} GB")
    
    if free_gb < MIN_DISK_GB:
        print(f"   ❌ CRITICAL: Free disk ({free_gb:.2f} GB) < minimum ({MIN_DISK_GB} GB)")
        return False
    else:
        print(f"   ✅ Free disk space sufficient")
    
    return True

def check_load():
    """Check system load average."""
    import os
    load_avg = os.getloadavg()
    cpu_count = psutil.cpu_count()
    
    print(f"\n⚙️  System Load:")
    print(f"   Load average (1m, 5m, 15m): {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
    print(f"   CPU cores: {cpu_count}")
    
    if load_avg[0] > cpu_count * 2:
        print(f"   ⚠️  WARNING: High load average ({load_avg[0]:.2f}) > 2x CPU cores ({cpu_count * 2})")
        print(f"      System may be under heavy load (check with 'top' or 'htop')")
    elif load_avg[0] > cpu_count:
        print(f"   ⚠️  WARNING: Load average ({load_avg[0]:.2f}) > CPU cores ({cpu_count})")
    else:
        print(f"   ✅ System load acceptable")
    
    return True

def check_cuda():
    """Check CUDA availability (optional)."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        print(f"\n🎮 CUDA:")
        if cuda_available:
            print(f"   ✅ CUDA available")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"   Memory: {props.total_memory / (1024**3):.2f} GB")
        else:
            print(f"   ℹ️  CUDA not available (will use CPU)")
    except ImportError:
        print(f"\n🎮 CUDA:")
        print(f"   ℹ️  PyTorch not installed, cannot check CUDA")
    
    return True

def main():
    """Run all preflight checks."""
    print("=" * 80)
    print("PREFLIGHT CHECK - System Resource Diagnostics")
    print("=" * 80)
    print()
    
    all_ok = True
    
    # Run checks
    all_ok &= check_ram()
    all_ok &= check_swap()
    all_ok &= check_disk()
    all_ok &= check_load()
    check_cuda()  # Optional, doesn't affect exit code
    
    print()
    print("=" * 80)
    if all_ok:
        print("✅ PREFLIGHT PASSED - System ready for training")
        print("=" * 80)
        return 0
    else:
        print("❌ PREFLIGHT FAILED - Critical resource shortage detected")
        print("=" * 80)
        print("\nRecommendations:")
        print("  - Free up RAM by closing other applications")
        print("  - Free up disk space if needed")
        print("  - Wait for system load to decrease")
        print("  - Consider reducing batch size or window size in config.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
