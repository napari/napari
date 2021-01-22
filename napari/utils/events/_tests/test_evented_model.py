from napari.utils.events import EventedModel


def test_creating_evented_model():
    """Test creating an evented pydantic model."""
    model = EventedModel()
    assert model is not None
    assert model.events is not None
