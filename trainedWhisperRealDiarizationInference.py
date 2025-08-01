#!/usr/bin/env python3
"""
Trained Whisper Medium Real Diarization Inference Script
Runs inference using your trained checkpoint with real diarization from pyannote
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run trained Whisper medium inference with real diarization on LibriMix dataset."""
    
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
    print("Trained Whisper Medium Real Diarization Inference")
    print("=" * 60)
    print("Running YOUR trained checkpoint with REAL diarization (pyannote)...")
    print("Model: Your trained LibriMix medium checkpoint (step 6000)")
    print("Diarization: pyannote/speaker-diarization-3.1")
    print(f"Repository root: {script_dir}")
    print("=" * 60)
    
    # Check if required files exist
    test_file = "libri2mix_mix_clean_sc_test_cutset.jsonl.gz"
    manifest_dir = script_dir / "data" / "manifests"
    checkpoint_path = script_dir / "exp" / "librimix_medium_training" / "checkpoint-6000"
    
    if not (manifest_dir / test_file).exists():
        print(f"ERROR: Required test manifest not found: {manifest_dir / test_file}")
        print("Please run the data preparation script first: scripts/data/prepare.sh")
        sys.exit(1)
    
    if not checkpoint_path.exists():
        print(f"ERROR: Trained checkpoint not found: {checkpoint_path}")
        print("Please run librimixMedium.py first to train the model")
        sys.exit(1)
    
    # Create decode config directory if it doesn't exist
    config_dir = script_dir / "configs" / "decode"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "trained_whisper_real_diar_librimix.yaml"
    
    # Create inference configuration for trained checkpoint with real diarization
    config_content = f"""# @package _global_

experiment: "trained_whisper_real_diar_librimix"

model:
  whisper_model: "openai/whisper-medium"
  reinit_from: "{checkpoint_path}"
  ctc_weight: 0.3
  use_qk_biasing: false
  shift_pos_embeds: false
  pretrained_encoder: null
  reinit_encoder_from: null
  target_amp_is_diagonal: true
  target_amp_bias_only: false
  target_amp_use_silence: true
  target_amp_use_target: true
  target_amp_use_overlap: true
  target_amp_use_non_target: true
  apply_target_amp_to_n_layers: -1
  target_amp_init: "disparagement"

data:
  eval_cutsets: ${{oc.env:MANIFEST_DIR}}/libri2mix_mix_clean_sc_test_cutset.jsonl.gz
  train_cutsets: 
    - ${{oc.env:MANIFEST_DIR}}/libri2mix_mix_clean_sc_test_cutset.jsonl.gz
  dev_cutsets: 
    - ${{oc.env:MANIFEST_DIR}}/libri2mix_mix_clean_sc_test_cutset.jsonl.gz
  audio_path_prefix: ${{oc.env:AUDIO_PATH_PREFIX}}
  audio_path_prefix_replacement: ${{oc.env:AUDIO_PATH_PREFIX_REPLACEMENT}}
  use_timestamps: true
  eval_text_norm: "whisper_nsf"
  # This enables real diarization mode - uses pyannote during inference
  use_diar: true

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
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    print(f"Created inference configuration: {config_file}")
    print(f"Using checkpoint: {checkpoint_path}")
    
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
        "+decode=trained_whisper_real_diar_librimix"
    ]
    
    try:
        print(f"Executing command: {' '.join(cmd)}")
        print("=" * 60)
        print("🚀 Starting trained Whisper medium inference with REAL diarization...")
        print("⚠️  Using your LibriMix-trained checkpoint from step 6000")
        print("⚠️  This will use pyannote for diarization during inference")
        print("⚠️  Expect significantly longer runtime due to diarization")
        print("⚠️  Results will show your model's performance with real diarization")
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
            print("✅ Trained Whisper medium real diarization inference completed!")
            print(f"Check results: {script_dir}/exp/trained_whisper_real_diar_librimix/")
            print("Compare with ground truth results to see diarization impact")
        else:
            print(f"❌ Trained Whisper medium real diarization inference failed: {result.returncode}")
            print("Check error messages above for details.")
        print("=" * 60)
        
        return result.returncode
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required executable: {e}")
        return 1
    except KeyboardInterrupt:
        print("\\n⚠️  Inference interrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: Unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)