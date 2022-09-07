import pytest

from .. import validators


def test_sequence_validator():
    validate = validators.validate_n_seq(2, int)

    # this should work
    validate([4, 5])

    with pytest.raises(TypeError):
        validate(8)  # raises TypeError

    with pytest.raises(ValueError):
        validate([1, 2, 3])  # raises ValueError

    with pytest.raises(TypeError):
        validate([1.4, 5])  # raises TypeError


def test_pairwise():
    source = [1, 2, 3]
    output = validators.pairwise(source)
    assert list(output) == [(1, 2), (2, 3)]


def test_validate_increasing():
    valid_source = [1, 2, 3]
    validators.validate_increasing(valid_source)

    invalid_sources = [[3, 2, 1], [1, 1, 2]]
    for source in invalid_sources:
        with pytest.raises(ValueError):
            validators.validate_increasing(source)
