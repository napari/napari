import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari.plugins._plugin_manager import NapariPluginManager


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


def test_plugin_events(napari_plugin_manager):
    """Test event emission by plugin manager."""
    tnpm: NapariPluginManager = napari_plugin_manager

    register_events = []
    enable_events = []
    disable_events = []

    tnpm.events.registered.connect(lambda e: register_events.append(e))
    tnpm.events.enabled.connect(lambda e: enable_events.append(e))
    tnpm.events.disabled.connect(lambda e: disable_events.append(e))

    class Plugin:
        pass

    tnpm.register(Plugin, name='Plugin')
    assert 'Plugin' in tnpm.plugins
    assert len(register_events) == 1
    assert register_events[0].value == 'Plugin'
    assert not enable_events
    assert not disable_events

    tnpm.set_blocked('Plugin')
    assert len(disable_events) == 1
    assert disable_events[0].value == 'Plugin'
    assert not enable_events
    assert 'Plugin' not in tnpm.plugins
    # blocked from registering
    assert tnpm.is_blocked('Plugin')
    tnpm.register(Plugin, name='Plugin')
    assert 'Plugin' not in tnpm.plugins
    assert len(register_events) == 1

    tnpm.set_blocked('Plugin', False)
    assert not tnpm.is_blocked('Plugin')
    assert len(enable_events) == 1
    assert enable_events[0].value == 'Plugin'
    # note: it doesn't immediately re-register it
    assert 'Plugin' not in tnpm.plugins
    # but we can now re-register it
    tnpm.register(Plugin, name='Plugin')
    assert len(register_events) == 2
