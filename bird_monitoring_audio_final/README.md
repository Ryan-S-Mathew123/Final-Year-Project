# Bird Monitoring — Audio Embedding Recognition

Audio-only embedding-based bird species recognition for 12 initial species.

## Dataset

Each species currently contains 60 recordings of approximately 5 seconds:

- spotted_dove
- rose_ringed_parakeet
- asian_koel
- common_cuckoo
- white_naped_woodpecker
- spotted_owlet
- rock_eagle_owl
- rock_pigeon
- indian_peafowl
- indian_grey_hornbill
- grey_junglefowl
- peregrine_falcon

Expected total: 720 recordings.

## Setup

Python 3.10+:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Validate:

```bash
python scripts/validate_dataset.py
```

Generate embeddings:

```bash
python scripts/generate_audio_embeddings.py
```

Predict:

```bash
python scripts/predict_audio.py path/to/test.wav
```

Calibrate unknown threshold:

```bash
python scripts/calibrate_thresholds.py
```

Add a species without retraining:

```bash
python scripts/add_species.py --species oriental_magpie_robin
```

Place recordings into the created folder, then regenerate embeddings:

```bash
python scripts/generate_audio_embeddings.py
```

Or:

```bash
python scripts/add_species.py --species oriental_magpie_robin --generate
```

The encoder is never retrained. New species are added by generating embeddings and rebuilding species prototypes.

## Search modes

Set in `config.py`:

- `individual`: search against every reference recording.
- `prototype`: search against one mean normalized embedding per species.

## Unknown detection

If maximum cosine similarity is below `AUDIO_UNKNOWN_THRESHOLD`, prediction returns `UNKNOWN`.
