#!/usr/bin/env python3
"""
Setup script for RadVLM evaluation environment.
This script helps configure the environment for running evaluation scripts.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_python_path():
    """Add the RadVLM directory to Python path."""
    radvlm_dir = Path(__file__).parent.absolute()
    python_path = os.environ.get('PYTHONPATH', '')
    
    if str(radvlm_dir) not in python_path:
        if python_path:
            new_python_path = f"{python_path}:{radvlm_dir}"
        else:
            new_python_path = str(radvlm_dir)
        
        os.environ['PYTHONPATH'] = new_python_path
        print(f"Added {radvlm_dir} to PYTHONPATH")
        print(f"Run: export PYTHONPATH={new_python_path}")
    else:
        print("RadVLM directory already in PYTHONPATH")

def check_data_dir():
    """Check if DATA_DIR environment variable is set."""
    data_dir = os.environ.get('DATA_DIR')
    if data_dir is None:
        print("WARNING: DATA_DIR environment variable is not set.")
        print("Please set it to your data directory:")
        print("export DATA_DIR=/path/to/your/data")
        return False
    else:
        print(f"DATA_DIR is set to: {data_dir}")
        if os.path.exists(data_dir):
            print("✓ DATA_DIR path exists")
            return True
        else:
            print("✗ DATA_DIR path does not exist")
            return False

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['torch', 'accelerate', 'transformers']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def print_usage_instructions():
    """Print instructions for running the evaluation script."""
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    
    print("\n1. For single GPU execution (recommended for testing):")
    print("   CUDA_VISIBLE_DEVICES=0 python radvlm/evaluation/evaluate_instructions.py \\")
    print("       --task abnormality_grounding \\")
    print("       --r1 \\")
    print("       --model_name /path/to/your/model \\")
    print("       --num_batches 10")
    
    print("\n2. For multi-GPU execution with accelerate:")
    print("   accelerate launch --num_processes=4 radvlm/evaluation/evaluate_instructions.py \\")
    print("       --task abnormality_grounding \\")
    print("       --r1 \\")
    print("       --model_name /path/to/your/model")
    
    print("\n3. Required environment variables:")
    print("   export DATA_DIR=/path/to/your/data")
    print("   export PYTHONPATH=$PYTHONPATH:/path/to/RadVLM")
    
    print("\n4. Available tasks:")
    tasks = [
        "abnormality_classification",
        "abnormality_grounding", 
        "abnormality_detection",
        "report_generation",
        "region_grounding",
        "object_grounding",
        "phrase_grounding",
        "vqa"
    ]
    for task in tasks:
        print(f"   - {task}")

def main():
    print("RadVLM Evaluation Setup")
    print("="*30)
    
    # Setup Python path
    setup_python_path()
    
    # Check DATA_DIR
    data_dir_ok = check_data_dir()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Print usage instructions
    print_usage_instructions()
    
    if not data_dir_ok or not deps_ok:
        print("\n" + "!"*60)
        print("SETUP INCOMPLETE - Please address the issues above")
        print("!"*60)
        sys.exit(1)
    else:
        print("\n" + "✓"*60)
        print("SETUP COMPLETE - Ready to run evaluation")
        print("✓"*60)

if __name__ == "__main__":
    main()