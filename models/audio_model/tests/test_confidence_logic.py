from app import should_mark_low_confidence


def test_high_confidence_known_prediction_is_not_low_confidence():
    assert not should_mark_low_confidence([0.92, 0.08], threshold=0.85, min_margin=0.10)


def test_low_confidence_prediction_is_rejected():
    assert should_mark_low_confidence([0.72, 0.26], threshold=0.85, min_margin=0.10)


def test_close_class_scores_are_rejected():
    assert should_mark_low_confidence([0.86, 0.82], threshold=0.85, min_margin=0.10)
