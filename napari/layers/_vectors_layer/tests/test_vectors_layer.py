from ..model import InvalidDataFormatError

import numpy as np

from napari import ViewerApp
from napari.util import app_context


def test_vectorData_datatype0():
    """
    test data of image-like shape
    :return:
    """
    with app_context():
        # create the viewer and window
        viewer = ViewerApp()

        N = 10
        M = 5
        pos = np.zeros(shape=(N, M, 2), dtype=np.float32)
        try:
            viewer.add_vectors(pos)
            assert(True)
        except Exception as ex:
            print("exception thrown when creating coord-like vector layer: "+str(ex))


def test_vectorData_datatype1():
    """
    test data of coord-like shape
    :return:
    """
    with app_context():
        # create the viewer and window
        viewer = ViewerApp()

        N = 10
        pos = np.zeros(shape=(N, 4), dtype=np.float32)
        try:
            viewer.add_vectors(pos)
        except Exception as ex:
            print("exception thrown when creating coord-like vector layer: "+str(ex))


def test_vectorData_datatype_notimpl():
    """
    test data of improper shape
    :return:
    """
    with app_context():
        # create the viewer and window
        viewer = ViewerApp()

        N = 10
        M = 5
        pos = np.zeros(shape=(N, M, 4), dtype=np.float32)
        try:
            viewer.add_vectors(pos)
        except InvalidDataFormatError:
            print('invalid data format')
        except Exception as ex:
            print("exception thrown when creating not implemented vector layer: "+str(ex))


def test_vectorData_datatype_notimpl2(self):
    """
    test data of vispy-coordinate shape (not allowed)
    :return:
    """
    with app_context():
        # create the viewer and window
        viewer = ViewerApp()

        N = 10
        pos = np.zeros(shape=(N, 2), dtype=np.float32)
        try:
            viewer.add_vectors(pos)
        except InvalidDataFormatError:
            print('invalid data format')
        except Exception as ex:
            print("exception thrown when creating not implemented vector layer: "+str(ex))


def test_vectorData_image_assignment(self):
    """
    test replacing vector data after layer construction
    :return:
    """
    with app_context():
        # create the viewer and window
        viewer = ViewerApp()

        N = 10
        pos = np.zeros(shape=(N, 4), dtype=np.float32)
        pos2 = np.zeros(shape=(N, N, 2), dtype=np.float32)
        try:
            layer = viewer.add_vectors(pos)
            layer.vectors = pos2
        except Exception as ex:
            print("exception thrown when : "+str(ex))
