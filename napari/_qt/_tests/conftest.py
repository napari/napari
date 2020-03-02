from typing import List

import pytest

from napari._qt.qt_viewer import QtViewer
from napari.components import ViewerModel


@pytest.fixture(scope="function")
def viewermodel_factory(qtbot):
    views: List[QtViewer] = []

    def actual_factory(*model_args, **model_kwargs):
        viewer = ViewerModel(*model_args, **model_kwargs)
        view = QtViewer(viewer)
        views.append(view)
        qtbot.add_widget(view)

        return view, viewer

    yield actual_factory

    for view in views:
        view.shutdown()
