from __future__ import annotations
from pathlib import Path
import librosa
import numpy as np
from config import AUDIO_TARGET_SAMPLE_RATE

def validate_audio(path: Path) -> tuple[bool, str, dict]:
    try:
        waveform, sample_rate = librosa.load(path, sr=None, mono=False)
        if waveform.size == 0:
            return False, "empty_audio", {}
        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=0)
        duration = len(waveform) / sample_rate
        if duration <= 0:
            return False, "invalid_duration", {}
        return True, "ok", {"sample_rate": int(sample_rate), "duration": float(duration), "samples": int(len(waveform))}
    except Exception as error:
        return False, f"unreadable:{error}", {}

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = librosa.load(path, sr=None, mono=True)
    if sample_rate != AUDIO_TARGET_SAMPLE_RATE:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=AUDIO_TARGET_SAMPLE_RATE)
        sample_rate = AUDIO_TARGET_SAMPLE_RATE
    waveform = waveform.astype(np.float32)
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    return waveform, sample_rate

def prepare_fixed_segment(waveform: np.ndarray, sample_rate: int, segment_seconds: float = 5.0) -> np.ndarray:
    target_length = int(segment_seconds * sample_rate)
    if len(waveform) == target_length:
        return waveform
    if len(waveform) > target_length:
        return waveform[:target_length]
    return np.pad(waveform, (0, target_length - len(waveform)))
