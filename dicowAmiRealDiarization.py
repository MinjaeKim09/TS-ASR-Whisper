#!/usr/bin/env python3
"""
DiCoW AMI Real Diarization Inference Script
Runs DiCoW inference on AMI dataset using real diarization from pyannote instead of ground truth
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run DiCoW inference with real diarization on AMI dataset."""
    
    # Get the script directory (repository root)
    script_dir = Path(__file__).parent.absolute()
    
    # Set up environment variables
    os.environ["SRC_ROOT"] = str(script_dir)
    os.environ["MANIFEST_DIR"] = str(script_dir / "data" / "manifests")
    os.environ["EXPERIMENT_PATH"] = str(script_dir / "exp")
    os.environ["PYTHONPATH"] = f"{script_dir}:{os.environ.get('PYTHONPATH', '')}"
    
    # Set additional required environment variables with defaults
    os.environ["LIBRI_TRAIN_CACHED_PATH"] = ""
    os.environ["LIBRI_DEV_CACHED_PATH"] = ""
    os.environ["WANDB_DISABLED"] = "true"
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
    print("DiCoW AMI Real Diarization Inference")
    print("=" * 60)
    print("Running DiCoW with REAL diarization (pyannote) on AMI...")
    print("Model: BUT-FIT/DiCoW_v2 (pretrained)")
    print("Diarization: pyannote/speaker-diarization-3.1")
    print("Dataset: AMI meeting corpus")
    print(f"Repository root: {script_dir}")
    print("=" * 60)
    
    # Check if required test file exists
    test_file = "ami-sdm_test_sc_cutset.jsonl.gz"
    manifest_dir = script_dir / "data" / "manifests"
    
    if not (manifest_dir / test_file).exists():
        print(f"ERROR: Required manifest file not found: {manifest_dir / test_file}")
        print("Please run the data preparation script first: scripts/data/prepare.sh")
        sys.exit(1)
    
    # Check if config exists, don't overwrite manually created one
    config_file = script_dir / "configs" / "decode" / "dicow_ami_real_diarization.yaml"
    if not config_file.exists():
        print(f"ERROR: Configuration file not found: {config_file}")
        print("Expected config should already exist with real diarization settings")
        sys.exit(1)
    
    print(f"Using existing configuration: {config_file}")
    
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
        "+decode=dicow_ami_real_diarization"
    ]
    
    try:
        print(f"Executing command: {' '.join(cmd)}")
        print("⚠️  Note: This will take significantly longer than ground truth due to real diarization")
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
            print("✅ DiCoW AMI real diarization inference completed successfully!")
            print(f"Check the output directory: {script_dir}/exp/dicow_ami_real_diarization/")
            print("Compare TCP-WER with ground truth version to see diarization impact")
        else:
            print(f"❌ DiCoW AMI real diarization inference failed with return code: {result.returncode}")
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