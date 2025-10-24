#!/usr/bin/env python3
"""
Simple system validation script to check if the pipeline is ready to run.
"""

import sys
import os
import importlib

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 8):
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version: {version.major}.{version.minor}.{version.micro} (3.8+ required)")
        return False

def check_required_packages():
    """Check required packages."""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_pipeline_modules():
    """Check if pipeline modules can be imported."""
    modules = [
        'crop_disease_detection.config',
        'crop_disease_detection.data_loader',
        'crop_disease_detection.preprocessor',
        'crop_disease_detection.model_trainer',
        'crop_disease_detection.pipeline'
    ]
    
    failed = []
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module} ({str(e)})")
            failed.append(module)
    
    return len(failed) == 0, failed

def check_dataset():
    """Check if dataset file exists."""
    dataset_file = "GHISACONUS_2008_001_speclib.csv"
    if os.path.exists(dataset_file):
        print(f"✓ Dataset file: {dataset_file}")
        return True
    else:
        print(f"✗ Dataset file: {dataset_file} (not found)")
        return False

def main():
    """Main validation function."""
    print("HYPERSPECTRAL CROP DISEASE DETECTION - SYSTEM VALIDATION")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Check Python version
    print("\n1. Python Version:")
    if not check_python_version():
        all_checks_passed = False
    
    # Check required packages
    print("\n2. Required Packages:")
    packages_ok, missing = check_required_packages()
    if not packages_ok:
        all_checks_passed = False
        print(f"\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing)}")
    
    # Check pipeline modules
    print("\n3. Pipeline Modules:")
    modules_ok, failed = check_pipeline_modules()
    if not modules_ok:
        all_checks_passed = False
    
    # Check dataset
    print("\n4. Dataset File:")
    dataset_ok = check_dataset()
    if not dataset_ok:
        all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ SYSTEM VALIDATION PASSED!")
        print("The pipeline is ready to run.")
        print("\nTo run the pipeline:")
        print("python main.py")
    else:
        print("❌ SYSTEM VALIDATION FAILED!")
        print("Please fix the issues above before running the pipeline.")
    print("=" * 60)
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)