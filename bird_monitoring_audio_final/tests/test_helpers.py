import numpy as np
from src.utils.helpers import l2_normalize

def test_l2_normalize():
    vector = np.array([3.0, 4.0], dtype=np.float32)
    normalized = l2_normalize(vector)
    assert np.isclose(np.linalg.norm(normalized), 1.0)
