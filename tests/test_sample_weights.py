import numpy as np

from src.sample_weights import label_uniqueness, sample_weights_from_uniqueness


def test_uniqueness_lower_when_labels_overlap():
    # Two fully overlapping intervals → uniqueness 0.5 each
    bar_id = np.array([0, 0])
    t1 = np.array([2, 2])
    uniq = label_uniqueness(bar_id, t1)
    assert np.allclose(uniq, [0.5, 0.5])
    w = sample_weights_from_uniqueness(uniq)
    assert np.allclose(w, [1.0, 1.0])


def test_uniqueness_one_when_disjoint():
    bar_id = np.array([0, 5])
    t1 = np.array([1, 6])
    uniq = label_uniqueness(bar_id, t1)
    assert np.allclose(uniq, [1.0, 1.0])
