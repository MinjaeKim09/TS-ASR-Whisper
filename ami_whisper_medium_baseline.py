#!/usr/bin/env python3
"""
Base Whisper Medium Evaluation on AMI Dataset
Runs evaluation on regular Whisper medium model using AMI dataset
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run inference/evaluation on regular Whisper medium model using AMI dataset."""
    
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
    os.environ["WANDB_DISABLED"] = "true"  # Disable wandb to avoid authentication issues
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
    print("Base Whisper Medium Evaluation on AMI Dataset")
    print("=" * 60)
    print("Running evaluation on regular Whisper medium model...")
    print("Using AMI test dataset")
    print(f"Repository root: {script_dir}")
    print(f"Manifest directory: {os.environ['MANIFEST_DIR']}")
    print("=" * 60)
    
    # Check if required test file exists
    test_file = "ami-sdm_test_sc_cutset.jsonl.gz"
    manifest_dir = script_dir / "data" / "manifests"
    
    if not (manifest_dir / test_file).exists():
        print(f"ERROR: Required test manifest not found: {manifest_dir / test_file}")
        print("Please run the data preparation script first: scripts/data/prepare.sh")
        sys.exit(1)
    
    # Create decode config directory if it doesn't exist
    config_dir = script_dir / "configs" / "decode"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "whisper_medium_ami_baseline.yaml"
    
    # Create inference configuration
    config_content = """# @package _global_

experiment: "whisper_medium_ami_baseline"

model:
  whisper_model: "openai/whisper-medium"
  reinit_from: null

data:
  eval_cutsets: ${oc.env:MANIFEST_DIR}/ami-sdm_test_sc_cutset.jsonl.gz
  train_cutsets: 
    - ${oc.env:MANIFEST_DIR}/ami-sdm_test_sc_cutset.jsonl.gz
  dev_cutsets: 
    - ${oc.env:MANIFEST_DIR}/ami-sdm_test_sc_cutset.jsonl.gz
  audio_path_prefix: ${oc.env:AUDIO_PATH_PREFIX}
  audio_path_prefix_replacement: ${oc.env:AUDIO_PATH_PREFIX_REPLACEMENT}
  use_timestamps: true
  eval_text_norm: "whisper_nsf"

decoding:
  decoding_ctc_weight: 0.0
  condition_on_prev: false
  length_penalty: 1.0

training:
  decode_only: true
  do_train: false
  do_eval: true
  eval_metrics_list: ["tcp_wer", "cp_wer", "tcorc_wer"]
  per_device_eval_batch_size: 2
  generation_max_length: 150
  bf16: true
  bf16_full_eval: true
  predict_with_generate: true
  output_dir: ${oc.env:EXPERIMENT_PATH}/${experiment}
  run_name: ${experiment}
  remove_unused_columns: false
  use_target_amplifiers: false

hydra:
  output_subdir: null
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    print(f"Created inference configuration: {config_file}")
    
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
        "+decode=whisper_medium_ami_baseline"
    ]
    
    try:
        print(f"Executing command: {' '.join(cmd)}")
        print("=" * 60)
        print("🚀 Starting base Whisper medium evaluation on AMI...")
        print("This will evaluate the regular Whisper medium model on AMI test set.")
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
            print("✅ Base Whisper medium evaluation completed successfully!")
            print(f"Check the output directory: {script_dir}/exp/whisper_medium_ami_baseline/")
            print("Results and predictions are saved there.")
        else:
            print(f"❌ Base Whisper medium evaluation failed with return code: {result.returncode}")
            print("Check the error messages above for details.")
        print("=" * 60)
        
        return result.returncode
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required executable: {e}")
        return 1
    except KeyboardInterrupt:
        print("\\n⚠️  Evaluation interrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: Unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)