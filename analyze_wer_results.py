#!/usr/bin/env python3
"""
Analyze WER results from Whisper baseline evaluation.
Calculates overall and per-session WER statistics for TCP and TC-ORC metrics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import editdistance


def load_json_file(file_path: str) -> List[Dict]:
    """Load JSON file containing utterances."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation."""
    # Convert to lowercase and split into words
    return text.lower().strip()


def calculate_wer(reference: str, hypothesis: str) -> Tuple[int, int, int, int]:
    """
    Calculate WER metrics using edit distance.
    Returns: (substitutions, deletions, insertions, reference_word_count)
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    ref_len = len(ref_words)
    if ref_len == 0:
        return 0, 0, len(hyp_words), 0
    
    # Calculate edit distance and get alignment
    distance = editdistance.eval(ref_words, hyp_words)
    
    # For detailed error analysis, we'd need alignment info
    # For now, we'll use the edit distance as total errors
    errors = distance
    
    # Approximate breakdown (this is simplified)
    hyp_len = len(hyp_words)
    if hyp_len > ref_len:
        insertions = hyp_len - ref_len
        substitutions = errors - insertions
        deletions = 0
    elif hyp_len < ref_len:
        deletions = ref_len - hyp_len
        substitutions = errors - deletions
        insertions = 0
    else:
        substitutions = errors
        deletions = 0
        insertions = 0
    
    # Ensure non-negative values
    substitutions = max(0, substitutions)
    deletions = max(0, deletions)
    insertions = max(0, insertions)
    
    return substitutions, deletions, insertions, ref_len


def process_session(session_dir: Path) -> Dict:
    """Process a single session directory and calculate WER metrics."""
    session_name = session_dir.name
    
    # Load files
    ref_file = session_dir / "ref.json"
    tcp_file = session_dir / "tcp_wer_hyp.json"
    tc_orc_file = session_dir / "tc_orc_wer_hyp.json"
    
    ref_data = load_json_file(str(ref_file))
    tcp_data = load_json_file(str(tcp_file))
    tc_orc_data = load_json_file(str(tc_orc_file))
    
    if not ref_data:
        return {
            'session': session_name,
            'error': 'No reference data',
            'tcp_wer': None,
            'tc_orc_wer': None
        }
    
    # Combine all utterances into single text strings for each condition
    ref_text = " ".join([normalize_text(utt.get('words', '')) for utt in ref_data])
    tcp_text = " ".join([normalize_text(utt.get('words', '')) for utt in tcp_data])
    tc_orc_text = " ".join([normalize_text(utt.get('words', '')) for utt in tc_orc_data])
    
    # Calculate WER for TCP
    tcp_subs, tcp_dels, tcp_ins, tcp_ref_len = calculate_wer(ref_text, tcp_text)
    tcp_errors = tcp_subs + tcp_dels + tcp_ins
    tcp_wer = (tcp_errors / tcp_ref_len * 100) if tcp_ref_len > 0 else float('inf')
    
    # Calculate WER for TC-ORC
    tc_orc_subs, tc_orc_dels, tc_orc_ins, tc_orc_ref_len = calculate_wer(ref_text, tc_orc_text)
    tc_orc_errors = tc_orc_subs + tc_orc_dels + tc_orc_ins
    tc_orc_wer = (tc_orc_errors / tc_orc_ref_len * 100) if tc_orc_ref_len > 0 else float('inf')
    
    return {
        'session': session_name,
        'ref_words': tcp_ref_len,  # Should be same for both
        'tcp': {
            'wer': tcp_wer,
            'errors': tcp_errors,
            'substitutions': tcp_subs,
            'deletions': tcp_dels,
            'insertions': tcp_ins,
            'hyp_words': len(tcp_text.split())
        },
        'tc_orc': {
            'wer': tc_orc_wer,
            'errors': tc_orc_errors,
            'substitutions': tc_orc_subs,
            'deletions': tc_orc_dels,
            'insertions': tc_orc_ins,
            'hyp_words': len(tc_orc_text.split())
        }
    }


def main():
    """Main function to analyze all WER results."""
    wer_dir = Path("/home/minjaekim/hailMary/TS-ASR-Whisper/exp/whisper_medium_ami_baseline/test/0/wer")
    
    if not wer_dir.exists():
        print(f"Directory not found: {wer_dir}")
        return
    
    # Get all session directories
    session_dirs = [d for d in wer_dir.iterdir() if d.is_dir()]
    session_dirs.sort()
    
    print(f"Found {len(session_dirs)} sessions")
    print("=" * 80)
    
    results = []
    total_tcp_errors = 0
    total_tc_orc_errors = 0
    total_ref_words = 0
    
    # Process each session
    for session_dir in session_dirs:
        result = process_session(session_dir)
        results.append(result)
        
        if 'error' not in result:
            total_ref_words += result['ref_words']
            total_tcp_errors += result['tcp']['errors']
            total_tc_orc_errors += result['tc_orc']['errors']
    
    # Calculate overall WER
    overall_tcp_wer = (total_tcp_errors / total_ref_words * 100) if total_ref_words > 0 else 0
    overall_tc_orc_wer = (total_tc_orc_errors / total_ref_words * 100) if total_ref_words > 0 else 0
    
    # Print results
    print("WHISPER MEDIUM AMI BASELINE - WER ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\nOVERALL RESULTS:")
    print(f"Total reference words: {total_ref_words:,}")
    print(f"TCP WER: {overall_tcp_wer:.2f}% ({total_tcp_errors:,} errors)")
    print(f"TC-ORC WER: {overall_tc_orc_wer:.2f}% ({total_tc_orc_errors:,} errors)")
    
    print(f"\nPER-SESSION BREAKDOWN:")
    print(f"{'Session':<10} {'Ref Words':<10} {'TCP WER %':<10} {'TC-ORC WER %':<12} {'TCP Errors':<11} {'TC-ORC Errors'}")
    print("-" * 80)
    
    for result in results:
        if 'error' in result:
            print(f"{result['session']:<10} ERROR: {result['error']}")
        else:
            session = result['session']
            ref_words = result['ref_words']
            tcp_wer = result['tcp']['wer']
            tc_orc_wer = result['tc_orc']['wer']
            tcp_errors = result['tcp']['errors']
            tc_orc_errors = result['tc_orc']['errors']
            
            print(f"{session:<10} {ref_words:<10,} {tcp_wer:<10.2f} {tc_orc_wer:<12.2f} {tcp_errors:<11,} {tc_orc_errors:,}")
    
    # Additional statistics
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        tcp_wers = [r['tcp']['wer'] for r in valid_results]
        tc_orc_wers = [r['tc_orc']['wer'] for r in valid_results]
        
        print(f"\nADDITIONAL STATISTICS:")
        print(f"Number of sessions: {len(valid_results)}")
        print(f"TCP WER - Min: {min(tcp_wers):.2f}%, Max: {max(tcp_wers):.2f}%, Avg: {sum(tcp_wers)/len(tcp_wers):.2f}%")
        print(f"TC-ORC WER - Min: {min(tc_orc_wers):.2f}%, Max: {max(tc_orc_wers):.2f}%, Avg: {sum(tc_orc_wers)/len(tc_orc_wers):.2f}%")


if __name__ == "__main__":
    main()