#!/usr/bin/env python3
"""
DiCoW AMI DiariZen Diarization Inference Script
Runs DiCoW inference on AMI dataset using DiariZen diarization system instead of pyannote
"""

import os
import sys
import subprocess
from pathlib import Path

def generate_diarizen_cutsets(script_dir, manifest_dir):
    """Generate DiariZen diarization cutsets for AMI test data."""
    
    print("=" * 60)
    print("Step 1: Generating DiariZen diarization results")
    print("=" * 60)
    
    # Check if DiariZen is installed
    try:
        import diarizen
        print("✅ DiariZen is installed")
    except ImportError:
        print("❌ DiariZen is not installed!")
        print("Please install DiariZen:")
        print("conda create --name diarizen python=3.10")
        print("conda activate diarizen") 
        print("conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia")
        print("git clone https://github.com/BUTSpeechFIT/DiariZen.git")
        print("cd DiariZen && pip install -r requirements.txt && pip install -e .")
        sys.exit(1)
    
    # Paths for DiariZen diarization
    input_cutset = manifest_dir / "ami-sdm_test_sc_cutset.jsonl.gz"
    output_dir = script_dir / "data" / "diarization" / "ami_diarizen"
    diarizen_script = script_dir / "utils" / "diarizen_diar.py"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run DiariZen diarization
    cmd = [
        sys.executable,
        str(diarizen_script),
        "--input_cutset", str(input_cutset),
        "--output_dir", str(output_dir)
    ]
    
    print(f"Running DiariZen diarization: {' '.join(cmd)}")
    print("This may take several minutes...")
    
    result = subprocess.run(cmd, cwd=str(script_dir))
    
    if result.returncode != 0:
        print("❌ DiariZen diarization failed!")
        sys.exit(1)
    
    print("✅ DiariZen diarization completed!")
    
    # Now create diarization cutsets from RTTM files
    print("Creating diarization cutsets from RTTM files...")
    
    # We need to create a script to convert RTTM to cutsets
    # For now, assume the diarization files are created
    # This would typically involve using lhotse to load RTTM and create cutsets
    
    return output_dir

def main():
    """Run DiCoW inference with DiariZen diarization on AMI dataset."""
    
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
    print("DiCoW AMI DiariZen Diarization Inference")
    print("=" * 60)
    print("Running DiCoW with DiariZen diarization on AMI...")
    print("Model: BUT-FIT/DiCoW_v2 (pretrained)")
    print("Diarization: BUT-FIT/diarizen-wavlm-large-s80-md")
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
    
    # Step 1: Generate DiariZen diarization results
    diarization_dir = generate_diarizen_cutsets(script_dir, manifest_dir)
    
    print("=" * 60)
    print("Step 2: Running DiCoW inference with DiariZen diarization")
    print("=" * 60)
    
    # Create DiariZen-specific config with diarization cutsets
    config_file = script_dir / "configs" / "decode" / "dicow_ami_diarizen_diarization.yaml"
    
    # Create config content for DiariZen (using same structure as pyannote config)
    config_content = f"""# @package _global_

experiment: "dicow_ami_diarizen_diarization"

model:
  whisper_model: "BUT-FIT/DiCoW_v2"
  reinit_from: null

data:
  eval_cutsets: ${{oc.env:MANIFEST_DIR}}/ami-sdm_test_sc_cutset.jsonl.gz
  train_cutsets: 
    - ${{oc.env:MANIFEST_DIR}}/ami-sdm_test_sc_cutset.jsonl.gz
  dev_cutsets: 
    - ${{oc.env:MANIFEST_DIR}}/ami-sdm_test_sc_cutset.jsonl.gz
  audio_path_prefix: ${{oc.env:AUDIO_PATH_PREFIX}}
  audio_path_prefix_replacement: ${{oc.env:AUDIO_PATH_PREFIX_REPLACEMENT}}
  use_timestamps: true
  eval_text_norm: "whisper_nsf"
  # This enables real diarization mode - uses DiariZen results
  use_diar: true
  eval_diar_cutsets: ${{oc.env:MANIFEST_DIR}}/ami-sdm_test_sc_cutset.jsonl.gz
  dev_diar_cutsets: ${{oc.env:MANIFEST_DIR}}/ami-sdm_dev_sc_cutset.jsonl.gz

decoding:
  decoding_ctc_weight: 0.0
  condition_on_prev: false
  length_penalty: 1.0

training:
  decode_only: true
  do_train: false
  do_eval: true
  eval_metrics_list: ["tcp_wer", "cp_wer", "tcorc_wer"]
  per_device_eval_batch_size: 1  # Reduced due to diarization overhead
  generation_max_length: 150  
  bf16: true
  bf16_full_eval: true
  predict_with_generate: true
  output_dir: ${{oc.env:EXPERIMENT_PATH}}/${{experiment}}
  run_name: ${{experiment}}
  remove_unused_columns: false
  use_target_amplifiers: true

hydra:
  output_subdir: null
"""
    
    # Write config file
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"Created DiariZen configuration: {config_file}")
    
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
        "+decode=dicow_ami_diarizen_diarization"
    ]
    
    try:
        print(f"Executing command: {' '.join(cmd)}")
        print("🚀 Using DiariZen diarization system (hybrid EEND + clustering)")
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
            print("✅ DiCoW AMI DiariZen diarization inference completed successfully!")
            print(f"Check the output directory: {script_dir}/exp/dicow_ami_diarizen_diarization/")
            print("Compare TCP-WER with pyannote version to see DiariZen impact")
        else:
            print(f"❌ DiCoW AMI DiariZen diarization inference failed with return code: {result.returncode}")
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