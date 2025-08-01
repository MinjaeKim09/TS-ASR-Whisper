#!/usr/bin/env python3
"""
Generate Real Diarization for LibriMix Test Set
Uses pyannote/speaker-diarization-3.1 to create realistic diarization outputs
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Generate real diarization for LibriMix test set using pyannote."""
    
    # Get the script directory (repository root)
    script_dir = Path(__file__).parent.absolute()
    
    # Paths
    input_cutset = script_dir / "data" / "manifests" / "libri2mix_mix_clean_sc_test_cutset.jsonl.gz"
    output_dir = script_dir / "data" / "diarization" / "librimix_pyannote"
    pyannote_script = script_dir / "utils" / "pyannote_diar.py"
    
    if not input_cutset.exists():
        print(f"ERROR: Input cutset not found: {input_cutset}")
        print("Please run the data preparation script first: scripts/data/prepare.sh")
        sys.exit(1)
    
    if not pyannote_script.exists():
        print(f"ERROR: Pyannote script not found: {pyannote_script}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Real Diarization for LibriMix")
    print("=" * 60)
    print(f"Input cutset: {input_cutset}")
    print(f"Output directory: {output_dir}")
    print("Using pyannote/speaker-diarization-3.1...")
    print("=" * 60)
    print("⚠️  This will take several hours for 3000 test files!")
    print("⚠️  Requires HuggingFace token for pyannote access")
    print("=" * 60)
    
    # Check if HuggingFace token is available
    if not os.environ.get("HF_TOKEN"):
        print("WARNING: HF_TOKEN environment variable not set.")
        print("You may need to set it for pyannote access:")
        print("export HF_TOKEN=your_huggingface_token")
        print()
    
    # Construct the command
    cmd = [
        sys.executable,
        str(pyannote_script),
        "--input_cutset", str(input_cutset),
        "--output_dir", str(output_dir)
    ]
    
    try:
        print(f"Executing command: {' '.join(cmd)}")
        print("🚀 Starting diarization generation...")
        print("This will process all 3000 LibriMix test files.")
        
        # Run the diarization
        result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            check=False,
            text=True
        )
        
        print("=" * 60)
        if result.returncode == 0:
            print("✅ Diarization generation completed successfully!")
            print(f"RTTM files saved to: {output_dir}")
            print()
            print("Next steps:")
            print("1. Convert RTTM to cutset format")
            print("2. Use the diarization cutset for inference")
        else:
            print(f"❌ Diarization generation failed with return code: {result.returncode}")
        print("=" * 60)
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\\n⚠️  Diarization generation interrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)