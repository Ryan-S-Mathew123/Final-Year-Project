# Bird Monitoring: Audio Bird Species Recognition

This project identifies bird species from uploaded or recorded audio. It contains two recognition workflows:

- A comparison workflow using MFCC features and four shared, regularized scikit-learn classifier families.
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

## Shared classifier comparison

Train the comparison classifiers from the repository root:

```powershell
python models/audio_model/train.py
```

The script extracts the same 40 averaged MFCC features for every clip, makes a deterministic stratified 70/15/15 train/validation/test split, and trains Random Forest, SVM, K-Nearest Neighbors, and Gradient Boosting. Training accuracy is retained as an overfitting check; model comparisons use validation and held-out test accuracy and macro F1. The dashboard reads the resulting artifacts and test metrics.

Start the Flask comparison dashboard from the repository root:

```powershell
python models/flask_frontend/app.py
```

Open `http://127.0.0.1:5000` in a browser and upload an audio file. Supported formats include WAV, MP3, OGG, FLAC, and M4A.

For image and video comparisons, also run `python models/image_model/train_classifiers.py`; it requires the ConvNeXt checkpoint and image train/validation/test folders. The separate audio embedding FastAPI application (`models/audio_model/app.py`) remains available for the CLAP/prototype workflow described below.

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
| `app.py` | Audio embedding FastAPI application |
| `train.py` | MFCC feature extraction and shared classifier training |
| `dataset/` | Input audio for the browser model |
| `datasets/audio/` | Input audio for the embedding workflow |
| `scripts/` | Dataset, embedding, and prediction commands |
| `src/` | Embedding workflow implementation |
| `templates/` | Web interface |
| `comparison_models/` | Trained shared-family model artifacts |
| `results/classifier_comparison.json` | Train/validation/test model metrics |
