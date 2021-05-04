import subprocess
import sys


def test_plugin_discovery_is_delayed():
    """Test that plugins are not getting discovered at napari import time."""
    cmd = [
        sys.executable,
        '-c',
        'import sys; from napari.plugins import plugin_manager; '
        'sys.exit(int("svg" in plugin_manager.plugins))',
    ]
    # will fail if plugin discovery happened at import
    proc = subprocess.run(cmd, capture_output=True)
    assert not proc.returncode, 'Plugins were discovered at import time!'

    cmd = [
        sys.executable,
        '-c',
        'import sys; '
        'from napari.plugins import plugin_manager; '
        'plugin_manager.discover(); '
        'sys.exit(not int("svg" in plugin_manager.plugins))',
    ]
    # will fail if napari-svg is not in the environment and test needs fixing
    proc = subprocess.run(cmd, capture_output=True)
    assert not proc.returncode, 'napari-svg unavailable, this test is broken!'
