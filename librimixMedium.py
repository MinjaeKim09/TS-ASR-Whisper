#!/usr/bin/env python3
"""
LibriMix Medium Training Script
Trains Whisper medium model using the libri2mix dataset for target speaker ASR
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Train Whisper medium model on LibriMix dataset using target amplifiers."""
    
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
    os.environ["WANDB_PROJECT"] = "librimix_medium_training"
    os.environ["WANDB_ENTITY"] = ""
    os.environ["HF_TOKEN"] = ""
    os.environ["HF_HOME"] = ""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["AUDIO_PATH_PREFIX"] = ""
    os.environ["AUDIO_PATH_PREFIX_REPLACEMENT"] = ""
    os.environ["PRETRAINED_CTC_MODELS_PATH"] = ""
    
    # Print status
    print("=" * 60)
    print("LibriMix Medium Training Script")
    print("=" * 60)
    print("Training Whisper medium model on LibriMix dataset...")
    print("Using target amplifier approach for speaker conditioning")
    print(f"Repository root: {script_dir}")
    print(f"Manifest directory: {os.environ['MANIFEST_DIR']}")
    print("=" * 60)
    
    # Check if required files exist
    train_files = [
        "libri2mix_clean_100_train_sc_cutset_30s.jsonl.gz",
        "libri2mix_both_100_train_sc_cutset_30s.jsonl.gz"
    ]
    
    dev_file = "libri2mix_mix_clean_sc_dev_cutset.jsonl.gz"
    test_file = "libri2mix_mix_clean_sc_test_cutset.jsonl.gz"
    
    manifest_dir = script_dir / "data" / "manifests"
    
    for train_file in train_files:
        if not (manifest_dir / train_file).exists():
            print(f"ERROR: Required training manifest not found: {manifest_dir / train_file}")
            print("Please run the data preparation script first: scripts/data/prepare.sh")
            sys.exit(1)
    
    if not (manifest_dir / dev_file).exists():
        print(f"ERROR: Required dev manifest not found: {manifest_dir / dev_file}")
        sys.exit(1)
        
    if not (manifest_dir / test_file).exists():
        print(f"ERROR: Required test manifest not found: {manifest_dir / test_file}")
        sys.exit(1)
    
    # Create custom config file if it doesn't exist
    config_dir = script_dir / "configs" / "train"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "librimix_medium.yaml"
    
    if not config_file.exists():
        config_content = """# @package _global_

experiment: "librimix_medium_training"

model:
  whisper_model: "openai/whisper-medium"
  ctc_weight: 0.3
  use_qk_biasing: false
  shift_pos_embeds: false
  pretrained_encoder: null
  reinit_encoder_from: null
  reinit_from: null
  target_amp_is_diagonal: true
  target_amp_bias_only: false
  target_amp_use_silence: true
  target_amp_use_target: true
  target_amp_use_overlap: true
  target_amp_use_non_target: true
  apply_target_amp_to_n_layers: -1
  target_amp_init: "disparagement"
  prefixes_to_preheat: ['model.encoder.additional_layer', 'model.encoder.additional_self_attention_layer', 'model.encoder.lm_head', 'model.encoder.subsample_conv1', 'model.encoder.subsample_conv2', 'model.encoder.target_amplifiers', 'model.encoder.blank_projection', 'model.encoder.modifiers','model.encoder.target_embeddings_proj']

data:
  use_libri: false
  libri_train_cached_path: ${oc.env:LIBRI_TRAIN_CACHED_PATH}
  libri_dev_cached_path: ${oc.env:LIBRI_DEV_CACHED_PATH}
  train_cutsets:
    - ${oc.env:MANIFEST_DIR}/libri2mix_clean_100_train_sc_cutset_30s.jsonl.gz
    - ${oc.env:MANIFEST_DIR}/libri2mix_both_100_train_sc_cutset_30s.jsonl.gz
  dev_cutsets: ${oc.env:MANIFEST_DIR}/libri2mix_mix_clean_sc_dev_cutset.jsonl.gz
  eval_cutsets: ${oc.env:MANIFEST_DIR}/libri2mix_mix_clean_sc_test_cutset.jsonl.gz
  use_timestamps: true
  dev_decoding_samples: 500
  train_with_diar_outputs: null
  train_text_norm: "whisper_nsf"
  eval_text_norm: "whisper_nsf"
  empty_transcripts_ratio: 0.0
  do_augment: true
  audio_path_prefix: ${oc.env:AUDIO_PATH_PREFIX}
  audio_path_prefix_replacement: ${oc.env:AUDIO_PATH_PREFIX_REPLACEMENT}
  use_random_segmentation: false
  mask_inputs: false
  random_sentence_l_crop_p: 0.0
  random_sentence_r_crop_p: 0.0
  max_l_crop: 0
  max_r_crop: 0
  vad_from_alignments: false
  cache_features_for_dev: false
  dataset_weights: null

decoding:
  decoding_ctc_weight: 0.0
  condition_on_prev: false
  length_penalty: 1.0

training:
  auto_find_batch_size: true
  bf16: true
  bf16_full_eval: true
  dataloader_num_workers: 4
  dataloader_prefetch_factor: 2
  dataloader_pin_memory: true
  overall_batch_size: 32
  decode_only: false
  use_custom_optimizer: true
  use_amplifiers_only_n_epochs: 1
  remove_timestamps_from_ctc: false
  target_amp_lr_multiplier: 100.0
  use_target_amplifiers: true
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 2
  max_steps: 20000
  num_train_epochs: 5
  early_stopping_patience: 3
  gradient_accumulation_steps: 1
  learning_rate: 2e-6
  warmup_steps: 1000
  weight_decay: 0.0
  greater_is_better: false
  ddp_find_unused_parameters: false
  generation_max_length: 225
  predict_with_generate: true
  
  eval_strategy: "steps"
  save_strategy: "steps"
  eval_steps: 500
  save_steps: 500
  
  metric_for_best_model: eval_tcp_wer
  train_metrics_list: ["tcp_wer", "cp_wer"]
  eval_metrics_list: ["tcp_wer", "cp_wer"]
  
  do_train: true
  load_best_model_at_end: true
  logging_steps: 10
  eval_delay: 1
  
  output_dir: ${oc.env:EXPERIMENT_PATH}/${experiment}
  run_name: ${experiment}
  
  remove_unused_columns: false

hydra:
  output_subdir: null

wandb:
  project: "librimix_medium_training"
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"Created training configuration: {config_file}")
    
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
        "--config-path", str(config_dir),
        "--config-name", "librimix_medium"
    ]
    
    try:
        print(f"Executing command: {' '.join(cmd)}")
        print("=" * 60)
        print("🚀 Starting LibriMix medium model training...")
        print("This will take several hours to complete.")
        print("=" * 60)
        
        # Run the training
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            check=False,  # Don't raise exception on non-zero exit
            text=True
        )
        
        print("=" * 60)
        if result.returncode == 0:
            print("✅ LibriMix medium training completed successfully!")
            print(f"Check the output directory: {script_dir}/exp/librimix_medium_training/")
            print("Model checkpoints and logs are saved there.")
        else:
            print(f"❌ LibriMix medium training failed with return code: {result.returncode}")
            print("Check the error messages above for details.")
        print("=" * 60)
        
        return result.returncode
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required executable: {e}")
        return 1
    except KeyboardInterrupt:
        print("\\n⚠️  Training interrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: Unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)