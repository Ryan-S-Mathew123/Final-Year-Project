from pathlib import Path
import json, sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.classifier import AudioClassifier

def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage:\npython scripts/predict_audio.py path/to/audio.wav")
    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")
    classifier = AudioClassifier()
    result = classifier.predict(audio_path)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
