from .. import validators
import pytest


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
