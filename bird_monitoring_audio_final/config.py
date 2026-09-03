from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
AUDIO_DATASET_DIR = DATASETS_DIR / "audio"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
AUDIO_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "audio_embeddings"
METADATA_DIR = PROJECT_ROOT / "metadata"

AUDIO_MODEL = "laion/larger_clap_general"
AUDIO_EMBEDDING_DIMENSION = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

AUDIO_TARGET_SAMPLE_RATE = 48_000
AUDIO_SEGMENT_SECONDS = 5.0
MIN_AUDIO_DURATION_SECONDS = 4.5
MAX_AUDIO_DURATION_SECONDS = 5.5

TOP_K = 5
AUDIO_SEARCH_MODE = "individual"
AUDIO_UNKNOWN_THRESHOLD = 0.35

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

SPECIES = {
    "spotted_dove": {"common_name": "Spotted Dove", "scientific_name": "Spilopelia chinensis"},
    "rose_ringed_parakeet": {"common_name": "Rose-ringed Parakeet", "scientific_name": "Psittacula krameri"},
    "asian_koel": {"common_name": "Asian Koel", "scientific_name": "Eudynamys scolopaceus"},
    "common_cuckoo": {"common_name": "Common Cuckoo", "scientific_name": "Cuculus canorus"},
    "white_naped_woodpecker": {"common_name": "White-naped Woodpecker", "scientific_name": "Chrysocolaptes festivus"},
    "spotted_owlet": {"common_name": "Spotted Owlet", "scientific_name": "Athene brama"},
    "rock_eagle_owl": {"common_name": "Rock Eagle-Owl", "scientific_name": "Bubo bengalensis"},
    "rock_pigeon": {"common_name": "Rock Pigeon", "scientific_name": "Columba livia"},
    "indian_peafowl": {"common_name": "Indian Peafowl", "scientific_name": "Pavo cristatus"},
    "indian_grey_hornbill": {"common_name": "Indian Grey Hornbill", "scientific_name": "Ocyceros birostris"},
    "grey_junglefowl": {"common_name": "Grey Junglefowl", "scientific_name": "Gallus sonneratii"},
    "peregrine_falcon": {"common_name": "Peregrine Falcon", "scientific_name": "Falco peregrinus"},
}

def ensure_directories() -> None:
    for directory in [AUDIO_DATASET_DIR, AUDIO_EMBEDDINGS_DIR, METADATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    for species_id in SPECIES:
        (AUDIO_DATASET_DIR / species_id).mkdir(parents=True, exist_ok=True)
