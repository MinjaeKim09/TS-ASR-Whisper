#!/usr/bin/env python3
"""
Trained Whisper Medium Real Diarization Pipeline
Uses pyannote for diarization + your trained checkpoint for ASR
"""

import os
import sys
import json
import gzip
from pathlib import Path
from tqdm import tqdm
import torch
from pyannote.audio import Pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def setup_models():
    """Initialize trained Whisper pipeline and pyannote diarization."""
    print("Loading models...")
    
    script_dir = Path(__file__).parent.absolute()
    checkpoint_path = script_dir / "exp" / "librimix_medium_training" / "checkpoint-6000"
    
    if not checkpoint_path.exists():
        print(f"ERROR: Trained checkpoint not found: {checkpoint_path}")
        print("Please run librimixMedium.py first to train the model")
        sys.exit(1)
    
    # Load trained model
    print(f"Loading trained checkpoint from: {checkpoint_path}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    
    # Create pipeline with trained model
    trained_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Load pyannote diarization pipeline
    try:
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HF_TOKEN")
        )
        if torch.cuda.is_available():
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))
    except Exception as e:
        print(f"ERROR: Failed to load pyannote diarization: {e}")
        print("Make sure you have set HF_TOKEN environment variable")
        sys.exit(1)
    
    print("✅ Models loaded successfully")
    return trained_pipeline, diarization_pipeline

def process_audio_with_real_diarization(audio_path, trained_pipeline, diarization_pipeline):
    """Process single audio file with real diarization + trained Whisper."""
    try:
        # Get diarization
        diarization = diarization_pipeline(audio_path)
        
        # Convert diarization to speaker segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        if not speaker_segments:
            return {"error": "No speakers detected by diarization"}
        
        # Process with trained model
        # Note: This processes the full audio - ideally we'd segment by speaker
        result = trained_pipeline(audio_path)
        
        # For now, we'll assign the transcription to the dominant speaker
        # In a full implementation, you'd segment the audio by speaker first
        dominant_speaker = max(speaker_segments, key=lambda x: x["end"] - x["start"])
        
        return {
            "success": True,
            "transcription": result["text"] if isinstance(result, dict) else str(result),
            "dominant_speaker": dominant_speaker["speaker"],
            "speaker_segments": speaker_segments,
            "num_speakers": len(speaker_segments)
        }
    
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

def main():
    """Run trained Whisper inference with real diarization on LibriMix test set."""
    
    script_dir = Path(__file__).parent.absolute()
    
    print("=" * 60)
    print("Trained Whisper Medium Real Diarization Pipeline")
    print("=" * 60)
    print("Processing LibriMix test set with:")
    print("• Pyannote speaker-diarization-3.1 for real diarization")
    print("• YOUR trained LibriMix checkpoint (step 6000)")
    print("=" * 60)
    
    # Check HF token
    if not os.environ.get("HF_TOKEN"):
        print("⚠️  WARNING: HF_TOKEN not set. You may encounter authentication issues.")
        print("Set it with: export HF_TOKEN=your_huggingface_token")
        print()
    
    # Load test manifest
    test_manifest = script_dir / "data" / "manifests" / "libri2mix_mix_clean_sc_test_cutset.jsonl.gz"
    if not test_manifest.exists():
        print(f"ERROR: Test manifest not found: {test_manifest}")
        print("Please run the data preparation script first: scripts/data/prepare.sh")
        sys.exit(1)
    
    # Setup output directory
    output_dir = script_dir / "exp" / "trained_whisper_real_diar_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    trained_pipeline, diarization_pipeline = setup_models()
    
    # Process test set
    print("🚀 Starting inference...")
    results = []
    errors = []
    
    # Read manifest
    with gzip.open(test_manifest, 'rt') as f:
        test_cuts = [json.loads(line) for line in f]
    
    print(f"Processing {len(test_cuts)} test files...")
    
    # Process subset for testing (first 10 files)
    test_subset = test_cuts[:10]  # Start with small subset
    print(f"⚠️  Processing subset of {len(test_subset)} files for testing")
    
    for cut in tqdm(test_subset, desc="Processing audio files"):
        cut_id = cut["id"]
        audio_path = cut["recording"]["sources"][0]["source"]
        
        # Check if audio file exists
        if not Path(audio_path).exists():
            error_msg = f"Audio file not found: {audio_path}"
            errors.append({"cut_id": cut_id, "error": error_msg})
            continue
        
        # Process with real diarization
        result = process_audio_with_real_diarization(audio_path, trained_pipeline, diarization_pipeline)
        result["cut_id"] = cut_id
        result["audio_path"] = audio_path
        
        if "error" in result:
            errors.append(result)
        else:
            results.append(result)
    
    # Save results
    results_file = output_dir / "trained_real_diarization_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "results": results,
            "errors": errors,
            "summary": {
                "total_processed": len(test_subset),
                "successful": len(results),
                "failed": len(errors),
                "checkpoint_used": "librimix_medium_training/checkpoint-6000"
            }
        }, f, indent=2)
    
    print("=" * 60)
    print("✅ Processing completed!")
    print(f"Results saved to: {results_file}")
    print(f"Successfully processed: {len(results)}/{len(test_subset)} files")
    if errors:
        print(f"⚠️  Errors: {len(errors)} files failed")
        print("Check the results file for error details")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\\n⚠️  Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)