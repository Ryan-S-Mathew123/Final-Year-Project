from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import ensure_directories
from src.audio.generate_embeddings import generate_audio_embeddings

if __name__ == "__main__":
    ensure_directories()
    generate_audio_embeddings()
