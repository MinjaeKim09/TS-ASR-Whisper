from argparse import ArgumentParser
import os

# DiariZen imports
from diarizen.pipelines.inference import DiariZenPipeline
from lhotse import load_manifest
import torch
from tqdm import tqdm


class DiariZenWrapper:
    """Wrapper to make DiariZen compatible with pyannote Pipeline interface"""
    
    def __init__(self, model_name="BUT-FIT/diarizen-wavlm-large-s80-md"):
        self.pipeline = DiariZenPipeline.from_pretrained(model_name)
        
    def __call__(self, audio_path):
        """Apply diarization to audio file and return pyannote-compatible result"""
        return self.pipeline(audio_path)
    
    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        """Create DiariZen wrapper from pretrained model"""
        return cls(model_name)
    
    def to(self, device):
        """Move pipeline to device (for compatibility)"""
        # DiariZen handles device internally
        return self


def main(cset_path, output_path):
    # Load DiariZen pipeline with pre-trained model
    pipeline = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md")
    
    cset = load_manifest(cset_path)

    for r in tqdm(cset):
        path = r.recording.sources[0].source
        fname = os.path.basename(path).split('.')[0]

        # Apply DiariZen diarization
        diarization = pipeline(path)

        # dump the diarization output to disk using RTTM format
        with open(f"{output_path}/{r.id}.rttm", "w") as rttm:
            diarization.write_rttm(rttm)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_cutset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    cset = load_manifest(args.input_cutset)

    os.makedirs(args.output_dir, exist_ok=True)

    main(args.input_cutset, args.output_dir)