from typing import List

import pytest
from napari import Viewer


@pytest.fixture(scope="function")
def viewer_factory(qapp):
    viewers: List[Viewer] = []

    def actual_factory(*model_args, **model_kwargs):
        viewer = Viewer(*model_args, **model_kwargs)
        viewers.append(viewer)
        return viewer.window.qt_viewer, viewer

    yield actual_factory

    for viewer in viewers:
        viewer.close()
