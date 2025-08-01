#!/usr/bin/env python3
"""
DiCoW Inference Script
Runs evaluation/inference on the pretrained DiCoW model from HuggingFace
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run DiCoW inference on AMI dataset using the pretrained model."""
    
    # Get the script directory (repository root)
    script_dir = Path(__file__).parent.absolute()
    
    # Set up environment variables
    os.environ["SRC_ROOT"] = str(script_dir)
    os.environ["MANIFEST_DIR"] = str(script_dir / "data" / "manifests")
    os.environ["PRETRAINED_MODEL_PATH"] = "BUT-FIT/DiCoW_v2"
    os.environ["EXPERIMENT_PATH"] = str(script_dir / "exp")
    os.environ["PYTHONPATH"] = f"{script_dir}:{os.environ.get('PYTHONPATH', '')}"
    
    # Set additional required environment variables with defaults
    os.environ["LIBRI_TRAIN_CACHED_PATH"] = ""
    os.environ["LIBRI_DEV_CACHED_PATH"] = ""
    os.environ["WANDB_PROJECT"] = ""
    os.environ["WANDB_ENTITY"] = ""
    os.environ["HF_TOKEN"] = ""
    os.environ["HF_HOME"] = ""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["AUDIO_PATH_PREFIX"] = ""
    os.environ["AUDIO_PATH_PREFIX_REPLACEMENT"] = ""
    os.environ["PRETRAINED_CTC_MODELS_PATH"] = ""
    
    # Print status
    print("=" * 60)
    print("DiCoW Inference Script")
    print("=" * 60)
    print(f"Running DiCoW inference with pretrained model: {os.environ['PRETRAINED_MODEL_PATH']}")
    print("Using AMI dataset for evaluation...")
    print(f"Repository root: {script_dir}")
    print(f"Manifest directory: {os.environ['MANIFEST_DIR']}")
    print("=" * 60)
    
    # Check if required files exist
    manifest_file = script_dir / "data" / "manifests" / "ami-sdm_test_sc_cutset.jsonl.gz"
    if not manifest_file.exists():
        print(f"ERROR: Required manifest file not found: {manifest_file}")
        print("Please run the data preparation script first: scripts/data/prepare.sh")
        sys.exit(1)
    
    config_file = script_dir / "configs" / "decode" / "dicow_ami_inference.yaml"
    if not config_file.exists():
        print(f"ERROR: Configuration file not found: {config_file}")
        sys.exit(1)
    
    # Find Python executable
    python_script = script_dir / "sge_tools" / "python"
    if python_script.exists():
        python_cmd = str(python_script)
    else:
        python_cmd = sys.executable
    
    # Construct the command
    main_script = script_dir / "src" / "main.py"
    cmd = [
        python_cmd,
        str(main_script),
        "+decode=dicow_ami_inference"
    ]
    
    try:
        print(f"Executing command: {' '.join(cmd)}")
        print("=" * 60)
        
        # Run the inference
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            check=False,  # Don't raise exception on non-zero exit
            text=True
        )
        
        print("=" * 60)
        if result.returncode == 0:
            print("✅ DiCoW inference completed successfully!")
            print(f"Check the output directory: {script_dir}/exp/dicow_ami_inference/")
        else:
            print(f"❌ DiCoW inference failed with return code: {result.returncode}")
            print("Check the error messages above for details.")
        print("=" * 60)
        
        return result.returncode
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required executable: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Inference interrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: Unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)