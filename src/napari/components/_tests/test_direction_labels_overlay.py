import pytest
from pydantic import ValidationError

from napari.components.overlays.direction_labels import DirectionLabelsOverlay


def test_defaults():
    overlay = DirectionLabelsOverlay()
    assert overlay.labels == ()
    assert overlay.visible is False  # inherited Overlay default
    assert overlay.box is False  # free overlay: no background box
    # position is a non-CanvasPosition sentinel so the overlay is not tiled.
    assert overlay.position not in ('bottom_right', 'top_left')


def test_labels_accepts_pairs_and_partial_none():
    overlay = DirectionLabelsOverlay(
        labels=[('I', 'S'), (None, 'P'), ('R', None), None]
    )
    assert overlay.labels == (('I', 'S'), (None, 'P'), ('R', None), None)


def test_labels_coerces_lists_to_tuples():
    overlay = DirectionLabelsOverlay(labels=[['A', 'P'], ['R', 'L']])
    assert overlay.labels == (('A', 'P'), ('R', 'L'))


def test_labels_none_means_no_labels():
    overlay = DirectionLabelsOverlay(labels=None)
    assert overlay.labels == ()


def test_labels_are_not_normalized_to_any_ndim():
    # Unlike a Dims field, the overlay stores labels verbatim (a suffix frame);
    # reconciliation to ndim happens at render time, not on the model.
    overlay = DirectionLabelsOverlay(labels=[('A', 'P'), ('R', 'L')])
    assert overlay.labels == (('A', 'P'), ('R', 'L'))


def test_string_entry_rejected():
    with pytest.raises(ValidationError, match='negative, positive'):
        DirectionLabelsOverlay(labels=['RL', ('A', 'P')])


def test_wrong_length_pair_rejected():
    with pytest.raises(ValidationError, match='negative, positive'):
        DirectionLabelsOverlay(labels=[('R', 'L', 'x')])


def test_non_string_label_rejected():
    with pytest.raises(ValidationError, match='must be a string'):
        DirectionLabelsOverlay(labels=[(1, 2)])


def test_non_iterable_labels_rejected():
    with pytest.raises(ValidationError, match='sequence'):
        DirectionLabelsOverlay(labels=1)
