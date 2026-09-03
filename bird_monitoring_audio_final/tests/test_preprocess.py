import numpy as np
from src.audio.preprocess import prepare_fixed_segment

def test_prepare_fixed_segment_padding():
    waveform = np.ones(10, dtype=np.float32)
    result = prepare_fixed_segment(waveform, sample_rate=10, segment_seconds=2)
    assert len(result) == 20

def test_prepare_fixed_segment_truncation():
    waveform = np.ones(30, dtype=np.float32)
    result = prepare_fixed_segment(waveform, sample_rate=10, segment_seconds=2)
    assert len(result) == 20
