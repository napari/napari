from napari.resources._icons import ICON_PATH
from napari.utils.misc import dir_hash, paths_hash


def test_icon_hash_equality():
    icons = {x.stem: str(x) for x in ICON_PATH.iterdir() if x.suffix == '.svg'}
    dir_hash_result = dir_hash(ICON_PATH)
    paths_hash_result = paths_hash(icons)
    assert dir_hash_result == paths_hash_result
