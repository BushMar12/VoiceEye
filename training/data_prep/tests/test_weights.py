import pytest
from training.data_prep.compute_class_weights import (
    inverse_frequency_weights,
    count_instances_per_class,
)


def test_equal_counts_gives_equal_weights():
    counts = {i: 100 for i in range(48)}
    w = inverse_frequency_weights(counts, nc=48)
    for v in w.values():
        assert v == pytest.approx(1.0)


def test_rare_class_gets_higher_weight():
    counts = {i: 1000 for i in range(48)}
    counts[0] = 10  # class 0 is 100x rarer
    w = inverse_frequency_weights(counts, nc=48)
    assert w[0] > w[1]
    # Mean should be 1.0
    assert sum(w.values()) / 48 == pytest.approx(1.0)


def test_zero_count_does_not_divide_by_zero():
    counts = {i: 100 for i in range(48)}
    counts[5] = 0
    w = inverse_frequency_weights(counts, nc=48)
    # Weight clamped: treated as count=1 internally
    assert w[5] > w[0]  # class 5 rarest
    assert all(v > 0 for v in w.values())


def test_count_instances_per_class(tmp_path):
    # Two label files: 3 class-0, 1 class-2, 0 elsewhere
    (tmp_path / 'a.txt').write_text('0 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.1 0.1')
    (tmp_path / 'b.txt').write_text('0 0.2 0.2 0.1 0.1\n2 0.7 0.7 0.2 0.2')
    counts = count_instances_per_class(tmp_path, nc=48)
    assert counts[0] == 3
    assert counts[2] == 1
    assert counts[1] == 0
