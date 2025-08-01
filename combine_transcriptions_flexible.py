#!/usr/bin/env python3
"""
Combine transcription results from any experiment directory
"""

import json
import glob
import sys
from pathlib import Path

def combine_transcriptions(experiment_path, output_name):
    """Combine all individual transcription files into one master file."""
    
    base_path = Path(experiment_path)
    
    if not base_path.exists():
        print(f"❌ Error: Path does not exist: {base_path}")
        return False
    
    # Find all transcription files recursively
    transcription_files = glob.glob(str(base_path / "**/tcp_wer_hyp.json"), recursive=True)
    
    if not transcription_files:
        print(f"❌ No transcription files found in {base_path}")
        return False
    
    print(f"Found {len(transcription_files)} transcription files in {base_path}")
    
    all_transcriptions = []
    
    # Process each file
    for file_path in sorted(transcription_files):
        session_name = Path(file_path).parent.name
        print(f"Processing: {session_name}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_transcriptions.extend(data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not all_transcriptions:
        print("❌ No transcription data found")
        return False
    
    # Sort by session_id and start_time
    all_transcriptions.sort(key=lambda x: (x.get('session_id', ''), x.get('start_time', 0)))
    
    # Determine output directory - use the same directory as input
    output_dir = base_path
    
    # Save combined JSON file
    json_output = output_dir / f"{output_name}_combined_transcriptions.json"
    with open(json_output, 'w') as f:
        json.dump(all_transcriptions, f, indent=2)
    
    print(f"✅ Combined {len(all_transcriptions)} segments into: {json_output}")
    
    # Create a readable text version
    text_output = output_dir / f"{output_name}_combined_transcriptions.txt"
    with open(text_output, 'w') as f:
        current_session = None
        
        for segment in all_transcriptions:
            # Session header
            session_id = segment.get('session_id', 'UNKNOWN')
            if session_id != current_session:
                current_session = session_id
                f.write(f"\n{'='*60}\n")
                f.write(f"SESSION: {current_session}\n")
                f.write(f"{'='*60}\n\n")
            
            # Format: [start-end] SPEAKER: text
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            speaker = segment.get('speaker', 'UNKNOWN')
            words = segment.get('words', '')
            
            f.write(f"[{start_time:7.2f}-{end_time:7.2f}] {speaker}: {words}\n")
    
    print(f"✅ Created readable text version: {text_output}")
    
    # Summary statistics
    sessions = set(seg.get('session_id', 'UNKNOWN') for seg in all_transcriptions)
    speakers = set(seg.get('speaker', 'UNKNOWN') for seg in all_transcriptions if seg.get('speaker'))
    total_duration = max((seg.get('end_time', 0) for seg in all_transcriptions), default=0)
    
    print(f"\n📊 Summary:")
    print(f"   - {len(sessions)} meetings")
    print(f"   - {len(speakers)} unique speakers")
    print(f"   - {len(all_transcriptions)} total segments")
    print(f"   - ~{total_duration:.1f}s total audio")
    
    return True

def main():
    """Main function to combine transcriptions."""
    
    if len(sys.argv) > 1:
        # Use provided path
        experiment_path = sys.argv[1]
        output_name = Path(experiment_path).name
    else:
        # Default: DiCoW AMI inference
        experiment_path = "/home/minjaekim/hailMary/TS-ASR-Whisper/exp/dicow_ami_inference"
        output_name = "dicow_ami"
    
    print(f"🎯 Combining transcriptions from: {experiment_path}")
    print(f"📝 Output name prefix: {output_name}")
    print()
    
    success = combine_transcriptions(experiment_path, output_name)
    
    if success:
        print(f"\n🎉 Successfully combined transcriptions!")
        print(f"📁 Files saved in: {experiment_path}")
    else:
        print(f"\n❌ Failed to combine transcriptions")
        sys.exit(1)

if __name__ == "__main__":
    main()