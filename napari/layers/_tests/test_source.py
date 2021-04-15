from magicgui import magicgui

from napari.layers import Points
from napari.layers._source import Source, current_source, layer_source


def test_layer_source():
    """Test basic layer source assignment mechanism"""
    with layer_source(path='some_path', reader_plugin='builtins'):
        points = Points()

    assert points.source == Source(path='some_path', reader_plugin='builtins')


def test_source_context():
    """Test nested contexts, overrides, and resets."""

    @magicgui
    def widget(x=1):
        ...

    assert current_source() == Source()
    # everything created within this context will have this sample source
    with layer_source(sample=('samp', 'name')):
        assert current_source() == Source(sample=('samp', 'name'))
        # nested contexts override previous ones
        with layer_source(path='a', reader_plugin='plug'):
            assert current_source() == Source(
                path='a', reader_plugin='plug', sample=('samp', 'name')
            )
            # note the new path now...
            with layer_source(path='b'):
                assert current_source() == Source(
                    path='b', reader_plugin='plug', sample=('samp', 'name')
                )
                with layer_source(widget=widget):
                    assert current_source() == Source(
                        path='b',
                        reader_plugin='plug',
                        sample=('samp', 'name'),
                        widget=widget,
                    )
                # as we exit the contexts, they should undo their assignments
                assert current_source() == Source(
                    path='b', reader_plugin='plug', sample=('samp', 'name')
                )
            assert current_source() == Source(
                path='a', reader_plugin='plug', sample=('samp', 'name')
            )
        assert current_source() == Source(sample=('samp', 'name'))
    assert current_source() == Source()
