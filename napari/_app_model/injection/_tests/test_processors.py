from typing import Union
from unittest.mock import MagicMock

import numpy as np
import pytest

from napari._app_model.injection._processors import _add_layer_data_to_viewer
from napari.types import ImageData, LabelsData


def test_add_layer_data_to_viewer():
    v = MagicMock()
    with pytest.raises(TypeError, match="napari supports only Optional"):
        _add_layer_data_to_viewer(
            data=np.zeros((10, 10)),
            return_type=Union[ImageData, LabelsData],
            viewer=v,
        )
