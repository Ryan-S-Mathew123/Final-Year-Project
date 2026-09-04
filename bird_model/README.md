# Bird Monitoring: Audio Bird Species Recognition

This project identifies bird species from uploaded or recorded audio. It contains two recognition workflows:

- A browser application using MFCC features and a Random Forest classifier.
- A CLAP audio-embedding workflow for command-line prediction and prototype-based search.

The project currently contains 12 species: Spotted Dove, Rose-ringed Parakeet, Asian Koel, Common Cuckoo, White-naped Woodpecker, Spotted Owlet, Rock Eagle-Owl, Rock Pigeon, Indian Peafowl, Indian Grey Hornbill, Grey Junglefowl, and Peregrine Falcon.

## Project location

Run all commands from the project directory:

```powershell
cd "C:\Btech\Final Year Project\Final-Year-Project"
```

## Installation (Windows PowerShell)

Python 3.10 or newer is recommended.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The repository already includes `venv` on some machines, but creating a fresh environment avoids dependency mismatches.

## Browser application

The web app loads `model.pkl` and `label_encoder.pkl` from the project root. Start it with:

```powershell
python train.py
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in a browser and upload an audio file. Supported formats include WAV, MP3, OGG, FLAC, and M4A.

Run `python train.py` again whenever the folders in `dataset/` change. The training script searches `dataset/` first and writes the updated model files to the project root.

## Embedding workflow

The embedding workflow uses `datasets/audio/` and stores generated vectors in `embeddings/` and metadata in `metadata/`.
It additionally requires PyTorch and Hugging Face Transformers:

```powershell
python -m pip install torch transformers
```

Validate the dataset:

```powershell
python scripts/validate_dataset.py
```

Generate audio embeddings:

```powershell
python scripts/generate_audio_embeddings.py
```

Predict from an audio file:

```powershell
python scripts/predict_audio.py "C:\path\to\test.wav"
```

Calibrate the unknown-species threshold:

```powershell
python scripts/calibrate_thresholds.py
```

Add a species, place its recordings in the generated folder, and rebuild embeddings:

```powershell
python scripts/add_species.py --species oriental_magpie_robin
python scripts/generate_audio_embeddings.py
```

Use `--generate` with `add_species.py` to perform both steps when the recordings are already present.

## Embedding settings

Edit `config.py` to change the search mode:

- `individual`: compare against every reference recording.
- `prototype`: compare against one mean normalized embedding per species.

If the maximum cosine similarity is below `AUDIO_UNKNOWN_THRESHOLD`, the classifier returns `UNKNOWN`.

## Tests

Run the available tests from the project directory:

```powershell
python -m pytest
```

## Main folders

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI web application |
| `train.py` | MFCC and Random Forest training |
| `dataset/` | Input audio for the browser model |
| `datasets/audio/` | Input audio for the embedding workflow |
| `scripts/` | Dataset, embedding, and prediction commands |
| `src/` | Embedding workflow implementation |
| `templates/` | Web interface |
| `model.pkl` | Trained Random Forest model |
| `label_encoder.pkl` | Saved species label encoder |
