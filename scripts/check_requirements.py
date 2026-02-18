#!/usr/bin/env python3
"""
Check if all required packages are installed with correct versions.
Compares against requirements.txt and OpenTSLM requirements.
"""

import sys
from pathlib import Path
import importlib.util

# Add paths
SCRIPT_DIR = Path(__file__).parent
FMRI_FLAMINGO_DIR = SCRIPT_DIR.parent
BASE_DIR = FMRI_FLAMINGO_DIR.parent.parent

# Try to import packaging for version checking
try:
    from packaging import version
    from packaging.requirements import Requirement
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False
    print("⚠️  'packaging' not installed. Install with: pip install packaging")
    print("   Will do basic checks only.\n")

def get_installed_packages():
    """Get all installed packages and versions."""
    try:
        import pkg_resources
        return {pkg.key.lower(): pkg.version for pkg in pkg_resources.working_set}
    except Exception as e:
        print(f"⚠️  Error getting installed packages: {e}")
        return {}

def parse_requirements_file(req_file):
    """Parse a requirements.txt file."""
    requirements = {}
    if not req_file.exists():
        return requirements
    
    with open(req_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            # Skip editable installs
            if line.startswith('-e'):
                continue
            
            # Parse requirement
            if HAS_PACKAGING:
                try:
                    req = Requirement(line)
                    requirements[req.name.lower()] = str(req.specifier)
                except Exception:
                    # If parsing fails, just store the line
                    parts = line.split('>=')
                    if len(parts) == 2:
                        requirements[parts[0].strip().lower()] = f">={parts[1].strip()}"
            else:
                # Basic parsing
                parts = line.split('>=')
                if len(parts) == 2:
                    requirements[parts[0].strip().lower()] = f">={parts[1].strip()}"
    
    return requirements

def check_version(installed_version, requirement_spec):
    """Check if installed version meets requirement."""
    if not HAS_PACKAGING:
        # Basic check: just compare if >= is in spec
        if '>=' in requirement_spec:
            min_version = requirement_spec.split('>=')[1].strip()
            try:
                return version.parse(installed_version) >= version.parse(min_version)
            except:
                return True  # If we can't parse, assume it's OK
        return True
    
    try:
        req = Requirement(f"dummy{requirement_spec}")
        return req.specifier.contains(installed_version)
    except:
        return True

def check_requirements():
    """Check all requirements."""
    print("=" * 80)
    print("REQUIREMENTS CHECK")
    print("=" * 80)
    print()
    
    # Get installed packages
    installed = get_installed_packages()
    print(f"📦 Found {len(installed)} installed packages\n")
    
    # Load requirements from our requirements.txt
    our_reqs_file = BASE_DIR / "requirements.txt"
    our_requirements = parse_requirements_file(our_reqs_file)
    
    # Load OpenTSLM requirements
    opentslm_reqs_file = BASE_DIR / "brain-model-alignment" / "OpenTSLM" / "requirements.txt"
    opentslm_requirements = parse_requirements_file(opentslm_reqs_file)
    
    # Combine all requirements (ours take precedence)
    all_requirements = {**opentslm_requirements, **our_requirements}
    
    print(f"📋 Checking {len(all_requirements)} required packages\n")
    print("-" * 80)
    
    missing = []
    wrong_version = []
    ok = []
    
    for pkg_name, spec in sorted(all_requirements.items()):
        if pkg_name not in installed:
            missing.append((pkg_name, spec))
            print(f"❌ MISSING: {pkg_name} {spec}")
        elif not check_version(installed[pkg_name], spec):
            wrong_version.append((pkg_name, spec, installed[pkg_name]))
            print(f"⚠️  WRONG VERSION: {pkg_name} {spec} (installed: {installed[pkg_name]})")
        else:
            ok.append((pkg_name, installed[pkg_name]))
            print(f"✅ OK: {pkg_name} {installed[pkg_name]} (requires {spec})")
    
    print("-" * 80)
    print()
    
    # Summary
    print("SUMMARY:")
    print(f"  ✅ OK: {len(ok)}")
    print(f"  ⚠️  Wrong version: {len(wrong_version)}")
    print(f"  ❌ Missing: {len(missing)}")
    print()
    
    if missing:
        print("MISSING PACKAGES:")
        for pkg, spec in missing:
            print(f"  pip install '{pkg}{spec}'")
        print()
    
    if wrong_version:
        print("VERSION MISMATCHES:")
        for pkg, spec, installed_ver in wrong_version:
            print(f"  {pkg}: need {spec}, have {installed_ver}")
        print()
    
    # Check for critical packages that might be missing
    critical_packages = ['torch', 'transformers', 'h5py', 'numpy', 'open-flamingo']
    print("CRITICAL PACKAGES CHECK:")
    for pkg in critical_packages:
        if pkg.lower() in installed:
            print(f"  ✅ {pkg}: {installed[pkg.lower()]}")
        else:
            print(f"  ❌ {pkg}: NOT INSTALLED")
    print()
    
    # Check if we can import key modules
    print("IMPORT TESTS:")
    test_imports = {
        'torch': 'torch',
        'transformers': 'transformers',
        'h5py': 'h5py',
        'numpy': 'numpy',
        'open_flamingo': 'open_flamingo',
        'einops': 'einops',
    }
    
    for module_name, import_name in test_imports.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✅ {module_name}: imported (version: {version})")
        except ImportError as e:
            print(f"  ❌ {module_name}: FAILED - {e}")
    
    print()
    
    if missing or wrong_version:
        print("⚠️  Some requirements are not met!")
        return False
    else:
        print("✅ All requirements satisfied!")
        return True

if __name__ == "__main__":
    success = check_requirements()
    sys.exit(0 if success else 1)
