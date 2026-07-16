import shutil

from napari.resources._icons import ICON_PATH, ICONS
from napari.utils.misc import dir_hash, paths_hash


def test_icon_hash_equality():
    if (_themes := ICON_PATH / '_themes').exists():
        shutil.rmtree(_themes)
    dir_hash_result = dir_hash(ICON_PATH)
    paths_hash_result = paths_hash(ICONS.values())
    assert dir_hash_result == paths_hash_result
