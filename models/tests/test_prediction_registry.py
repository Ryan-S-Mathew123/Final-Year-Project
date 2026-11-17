import numpy as np

from models.prediction_registry import extract_audio_features
from models.species import normalize_species_id


def test_audio_labels_map_to_image_species_ids():
    aliases = {
        "Grey Indian HornBill": "indian_grey_hornbill",
        "Grey Junglefowl Clips": "grey_junglefowl",
        "Indian Peafoul": "indian_peafowl",
        "Peregrine Falcon  Clips": "peregrine_falcon",
        "Rock Eagle": "rock_eagle_owl",
        "Spotted Dove_": "spotted_dove",
    }

    for label, expected in aliases.items():
        assert normalize_species_id(label) == expected


def test_audio_features_accept_mobile_sample_rate_and_match_training_shape():
    sample_rate = 44_100
    time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.1 * np.sin(2 * np.pi * 440 * time_axis)

    features = extract_audio_features(audio, sample_rate)

    assert features.shape == (1, 40)
    assert np.isfinite(features).all()
