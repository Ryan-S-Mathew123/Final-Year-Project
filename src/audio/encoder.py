from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor
from config import AUDIO_MODEL, DEVICE
from src.audio.preprocess import load_audio, prepare_fixed_segment
from src.utils.helpers import get_logger, l2_normalize

logger = get_logger(__name__)

class AudioEncoder:
    def __init__(self, model_name: str = AUDIO_MODEL, device: str = DEVICE) -> None:
        self.device = torch.device(device)
        logger.info("Loading audio model: %s", model_name)
        logger.info("Device: %s", self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, path: Path) -> np.ndarray:
        waveform, sample_rate = load_audio(path)
        waveform = prepare_fixed_segment(waveform, sample_rate)
        inputs = self.processor(audio=waveform, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        embedding = self.model.get_audio_features(**inputs).pooler_output.squeeze(0).cpu().numpy()
        return l2_normalize(embedding)
