import pytest

GOOD_PLUGIN = """
from napari_plugin_engine import HookImplementationMarker

@HookImplementationMarker("test")
def napari_get_reader(path):
    return True
"""


@pytest.fixture
def entrypoint_plugin(tmp_path):
    """An example plugin that uses entry points."""
    (tmp_path / "entrypoint_plugin.py").write_text(GOOD_PLUGIN)
    distinfo = tmp_path / "entrypoint_plugin-1.2.3.dist-info"
    distinfo.mkdir()
    (distinfo / "top_level.txt").write_text('entrypoint_plugin')
    (distinfo / "entry_points.txt").write_text(
        "[app.plugin]\na_plugin = entrypoint_plugin"
    )
    (distinfo / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: a_plugin\n"
        "Version: 1.2.3\n"
        "Author-Email: example@example.com\n"
        "Home-Page: https://www.example.com\n"
        "Requires-Python: >=3.7\n"
    )
    return tmp_path
