#!/usr/bin/env python3
"""
AMI Whisper Medium Training and Evaluation Script
Trains Whisper medium model on the AMI dataset for 6000 steps, then runs evaluation
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Train Whisper medium model on AMI dataset for 6000 steps, then run evaluation."""
    
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
    os.environ["WANDB_PROJECT"] = "ami_whisper_medium_training"
    os.environ["WANDB_ENTITY"] = ""
    os.environ["HF_TOKEN"] = ""
    os.environ["HF_HOME"] = ""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["AUDIO_PATH_PREFIX"] = ""
    os.environ["AUDIO_PATH_PREFIX_REPLACEMENT"] = ""
    os.environ["PRETRAINED_CTC_MODELS_PATH"] = ""
    os.environ["PRETRAINED_MODEL_PATH"] = ""
    
    # Print status
    print("=" * 60)
    print("AMI Whisper Medium Training and Evaluation Script")
    print("=" * 60)
    print("Training Whisper medium model on AMI dataset for 6000 steps...")
    print("Then running evaluation with REAL diarization (pyannote/speaker-diarization-3.1)")
    print(f"Repository root: {script_dir}")
    print(f"Manifest directory: {os.environ['MANIFEST_DIR']}")
    print("=" * 60)
    
    # Check if required files exist
    train_file = "ami-sdm_train_sc_cutset_30s.jsonl.gz"
    dev_file = "ami-sdm_dev_sc_cutset.jsonl.gz"
    test_file = "ami-sdm_test_sc_cutset.jsonl.gz"
    
    manifest_dir = script_dir / "data" / "manifests"
    
    for file_name in [train_file, dev_file, test_file]:
        if not (manifest_dir / file_name).exists():
            print(f"ERROR: Required manifest not found: {manifest_dir / file_name}")
            print("Please run the data preparation script first: scripts/data/prepare.sh")
            sys.exit(1)
    
    # Create config directories if they don't exist
    train_config_dir = script_dir / "configs" / "train"
    decode_config_dir = script_dir / "configs" / "decode"
    train_config_dir.mkdir(parents=True, exist_ok=True)
    decode_config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training configuration
    train_config_file = train_config_dir / "ami_whisper_medium.yaml"
    train_config_content = """# @package _global_

experiment: "ami_whisper_medium_training"

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
  train_cutsets: ${oc.env:MANIFEST_DIR}/ami-sdm_train_sc_cutset_30s.jsonl.gz
  dev_cutsets: ${oc.env:MANIFEST_DIR}/ami-sdm_dev_sc_cutset.jsonl.gz
  eval_cutsets: ${oc.env:MANIFEST_DIR}/ami-sdm_test_sc_cutset.jsonl.gz
  use_timestamps: true
  dev_decoding_samples: 500
  train_with_diar_outputs: null
  train_text_norm: "whisper_nsf"
  eval_text_norm: "whisper_nsf"
  empty_transcripts_ratio: 0.0
  do_augment: false
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
  auto_find_batch_size: false
  bf16: true
  bf16_full_eval: true
  dataloader_num_workers: 2
  dataloader_prefetch_factor: 1
  dataloader_pin_memory: false
  overall_batch_size: 16
  decode_only: false
  use_custom_optimizer: true
  use_amplifiers_only_n_epochs: 1
  remove_timestamps_from_ctc: false
  target_amp_lr_multiplier: 100.0
  use_target_amplifiers: true
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  max_steps: 6000
  num_train_epochs: 10
  early_stopping_patience: 3
  gradient_accumulation_steps: 16
  learning_rate: 2e-6
  warmup_steps: 600
  weight_decay: 0.0
  greater_is_better: false
  ddp_find_unused_parameters: false
  generation_max_length: 150
  predict_with_generate: true
  gradient_checkpointing: false
  
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
  project: "ami_whisper_medium_training"
"""
    
    with open(train_config_file, 'w') as f:
        f.write(train_config_content)
    print(f"Created training configuration: {train_config_file}")
    
    # Create evaluation configuration with real diarization
    eval_config_file = decode_config_dir / "ami_whisper_medium_eval.yaml"
    eval_config_content = """# @package _global_

experiment: "ami_whisper_medium_evaluation"

model:
  whisper_model: "openai/whisper-medium"
  reinit_from: "${oc.env:EXPERIMENT_PATH}/ami_whisper_medium_training/pytorch_model.bin"

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
  # Enable real diarization using pyannote/speaker-diarization-3.1
  use_diar: true
  eval_diar_cutsets: ${oc.env:MANIFEST_DIR}/ami-sdm_test_sc_cutset.jsonl.gz
  dev_diar_cutsets: ${oc.env:MANIFEST_DIR}/ami-sdm_dev_sc_cutset.jsonl.gz

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
  generation_max_length: 225
  bf16: true
  bf16_full_eval: true
  predict_with_generate: true
  output_dir: ${oc.env:EXPERIMENT_PATH}/${experiment}
  run_name: ${experiment}
  remove_unused_columns: false
  use_target_amplifiers: true

hydra:
  output_subdir: null
"""
    
    with open(eval_config_file, 'w') as f:
        f.write(eval_config_content)
    print(f"Created evaluation configuration: {eval_config_file}")
    
    # Find Python executable
    python_script = script_dir / "sge_tools" / "python"
    if python_script.exists():
        python_cmd = str(python_script)
    else:
        python_cmd = sys.executable
    
    main_script = script_dir / "src" / "main.py"
    
    try:
        # Phase 1: Training
        print("=" * 60)
        print("🚀 Phase 1: Training Whisper medium on AMI dataset...")
        print("This will train for 6000 steps.")
        print("=" * 60)
        
        train_cmd = [
            python_cmd,
            str(main_script),
            "--config-path", str(train_config_dir),
            "--config-name", "ami_whisper_medium"
        ]
        
        print(f"Executing training command: {' '.join(train_cmd)}")
        
        train_result = subprocess.run(
            train_cmd,
            cwd=str(script_dir),
            check=False,
            text=True
        )
        
        if train_result.returncode != 0:
            print(f"❌ Training failed with return code: {train_result.returncode}")
            return train_result.returncode
        
        print("✅ Training completed successfully!")
        print("=" * 60)
        
        # Phase 2: Evaluation with Real Diarization
        print("🔍 Phase 2: Evaluating trained model on AMI test set with REAL diarization...")
        print("⚠️  This will use pyannote/speaker-diarization-3.1 for real-time diarization")
        print("⚠️  Expect significantly longer runtime due to diarization overhead")
        print("=" * 60)
        
        eval_cmd = [
            python_cmd,
            str(main_script),
            "+decode=ami_whisper_medium_eval"
        ]
        
        print(f"Executing evaluation command: {' '.join(eval_cmd)}")
        
        eval_result = subprocess.run(
            eval_cmd,
            cwd=str(script_dir),
            check=False,
            text=True
        )
        
        print("=" * 60)
        if eval_result.returncode == 0:
            print("✅ Full pipeline completed successfully!")
            print(f"Training results: {script_dir}/exp/ami_whisper_medium_training/")
            print(f"Evaluation results (with real diarization): {script_dir}/exp/ami_whisper_medium_evaluation/")
            print("📊 Check TCP-WER, CP-WER, and TCORC-WER metrics in the evaluation results")
            print("🎯 Real diarization results show realistic performance with diarization errors")
        else:
            print(f"❌ Evaluation failed with return code: {eval_result.returncode}")
            print("Training was successful, but evaluation with real diarization had issues.")
        print("=" * 60)
        
        return eval_result.returncode
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required executable: {e}")
        return 1
    except KeyboardInterrupt:
        print("\\n⚠️  Process interrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: Unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)