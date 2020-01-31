import sys
import os
from napari.plugins.manager import find_module_by_prefix


def test_naming_convention_discovery():
    path = os.path.join(os.path.dirname(__file__), 'fixtures')
    sys.path.append(path)
    module_names = [m.__name__ for m in find_module_by_prefix()]
    assert 'napari_test_plugin' in module_names
    sys.path.pop(sys.path.index(path))
