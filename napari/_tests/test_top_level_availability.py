import napari


def test_top_level_availability(make_napari_viewer):
    """Current viewer should be available at napari.current_viewer."""
    viewer = make_napari_viewer()
    assert viewer == napari.current_viewer()
