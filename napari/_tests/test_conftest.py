from unittest.mock import Mock, patch

import pytest
from superqt.utils import qthrottled


@pytest.mark.usefixtures("disable_throttling")
@patch("qtpy.QtCore.QTimer.start")
def test_disable_throttle(start_mock):
    mock = Mock()

    @qthrottled(timeout=5)
    def f() -> str:
        mock()

    f()
    start_mock.assert_not_called()
    mock.assert_called_once()
