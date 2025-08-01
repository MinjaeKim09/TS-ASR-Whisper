#!/usr/bin/env python3
"""
Combine all AMI transcription results into a single file
"""

import json
import glob
from pathlib import Path

def combine_transcriptions():
    """Combine all individual transcription files into one master file."""
    
    # Path to transcription files
    base_path = Path("/home/minjaekim/hailMary/TS-ASR-Whisper/exp/ami_whisper_medium_training/test/6000/wer")
    
    # Find all transcription files
    transcription_files = glob.glob(str(base_path / "*/tcp_wer_hyp.json"))
    
    print(f"Found {len(transcription_files)} transcription files")
    
    all_transcriptions = []
    
    # Process each file
    for file_path in sorted(transcription_files):
        print(f"Processing: {Path(file_path).parent.name}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_transcriptions.extend(data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Sort by session_id and start_time
    all_transcriptions.sort(key=lambda x: (x['session_id'], x['start_time']))
    
    # Save combined file
    output_file = base_path.parent / "combined_transcriptions.json"
    with open(output_file, 'w') as f:
        json.dump(all_transcriptions, f, indent=2)
    
    print(f"✅ Combined {len(all_transcriptions)} segments into: {output_file}")
    
    # Create a readable text version
    text_output = base_path.parent / "combined_transcriptions.txt"
    with open(text_output, 'w') as f:
        current_session = None
        
        for segment in all_transcriptions:
            # Session header
            if segment['session_id'] != current_session:
                current_session = segment['session_id']
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
    sessions = set(seg['session_id'] for seg in all_transcriptions)
    speakers = set(seg['speaker'] for seg in all_transcriptions if seg.get('speaker'))
    total_duration = max(seg.get('end_time', 0) for seg in all_transcriptions)
    
    print(f"\n📊 Summary:")
    print(f"   - {len(sessions)} meetings")
    print(f"   - {len(speakers)} unique speakers")
    print(f"   - {len(all_transcriptions)} total segments")
    print(f"   - ~{total_duration:.1f}s total audio")

if __name__ == "__main__":
    combine_transcriptions()