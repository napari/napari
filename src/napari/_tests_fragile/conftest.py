import sys
from pathlib import Path

import pytest


def pytest_ignore_collect(
    collection_path: Path, config: pytest.Config
) -> bool:
    return all('napari/_tests_fragile' not in arg for arg in sys.argv)
