from napari.components.overlays.text import TextOverlay


def test_text_overlay():
    label = TextOverlay()
    assert label is not None
