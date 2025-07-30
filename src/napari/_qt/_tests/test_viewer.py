"""Tests of the Viewer class that interact directly with the Qt code"""

import textwrap
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QUrl


@pytest.mark.usefixtures('builtins')
def test_drop_python_file(make_napari_viewer, tmp_path):
    """Test dropping a python file on to the viewer."""
    viewer = make_napari_viewer()
    filename = tmp_path / 'image_to_drop.py'
    file_content = textwrap.dedent("""
    import numpy as np
    from napari import Viewer

    data = np.zeros((10, 10))
    viewer = Viewer()
    viewer.add_image(data, name='Dropped Image')
    """)
    filename.write_text(file_content)

    # Simulate dropping the file
    mock_event = MagicMock()
    mock_event.mimeData.return_value.urls.return_value = [
        QUrl(filename.as_uri())
    ]
    viewer.window._qt_viewer.dropEvent(mock_event)

    # Check that the file was executed
    assert viewer.layers[0].name == 'Dropped Image'


@pytest.mark.usefixtures('builtins')
def test_drop_python_file_3d(make_napari_viewer, tmp_path):
    """Test that dropping a python file using a 3D image on the viewer works."""
    viewer = make_napari_viewer()
    filename = tmp_path / 'image_to_drop_3d.py'
    file_content = textwrap.dedent("""
    import numpy as np
    from napari import Viewer

    data = np.zeros((2, 10, 10))
    viewer = Viewer(ndisplay=3)
    viewer.add_image(data, name='Dropped Image')
    """)
    filename.write_text(file_content)

    # Simulate dropping the file
    mock_event = MagicMock()
    mock_event.mimeData.return_value.urls.return_value = [
        QUrl(filename.as_uri())
    ]
    viewer.window._qt_viewer.dropEvent(mock_event)

    # Check that the file was executed
    assert viewer.layers[0].name == 'Dropped Image'
    assert viewer.dims.ndim == 3


@pytest.mark.usefixtures('builtins')
def test_drop_python_file_double_viewer(make_napari_viewer, tmp_path):
    """Test that dropping a python file on the viewer works."""
    viewer = make_napari_viewer()
    filename = tmp_path / 'test.py'
    file_content = textwrap.dedent("""
    import numpy as np
    from napari import Viewer

    data = np.zeros((10, 10))
    viewer1 = Viewer()
    viewer1.add_image(data, name='Dropped Image')
    viewer2 = Viewer(title="text")
    viewer2.add_image(data, name='Dropped Image 2')
    """)
    filename.write_text(file_content)

    # Simulate dropping the file
    mock_event = MagicMock()
    mock_event.mimeData.return_value.urls.return_value = [
        QUrl(filename.as_uri())
    ]
    viewer.window._qt_viewer.dropEvent(mock_event)

    # Check that the file was executed
    assert viewer.layers[0].name == 'Dropped Image'
    assert len(viewer._instances) == 2  # Two viewers should be created
    instances = list(viewer._instances)
    idx = 0 if instances[1] == viewer else 1
    assert instances[idx].title == 'text'  # Check the second viewer's name
    instances[idx].close()  # Close the second viewer
