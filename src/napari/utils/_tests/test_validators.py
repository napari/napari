import pytest

from napari.utils import validators


def test_sequence_validator():
    correct_float_source = [1.0, 2.0]
    correct_int_source = [1, 2]
    incorrect_sources = [
        [1, None],
        [None, 2.0],
    ]

    assert validators.check_sequence(correct_float_source, 2)
    assert validators.check_sequence(correct_int_source, 2)
    for source in incorrect_sources:
        assert not validators.check_sequence(source, 2)


def test_pairwise():
    source = [1, 2, 3]
    output = validators._pairwise(source)
    assert list(output) == [(1, 2), (2, 3)]


def test_validate_increasing():
    valid_source = [1, 2, 3]
    validators.validate_increasing(valid_source)

    invalid_sources = [[3, 2, 1], [1, 1, 2]]
    for source in invalid_sources:
        with pytest.raises(
            ValueError, match='must be monotonically increasing'
        ):
            validators.validate_increasing(source)
