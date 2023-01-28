"""Test _get_loader_configs() function."""
import pytest

from napari.components.experimental.chunk._pool_group import (
    _get_loader_configs,
)


def test_get_loader_config_error():
    """Test that defaults are required."""
    with pytest.raises(KeyError):
        _get_loader_configs({})


@pytest.mark.async_only
def test_get_loader_config_defaults():
    """Test config that has defaults but no octree loaders."""
    config = {
        "loader_defaults": {
            "force_synchronous": False,
            "num_workers": 10,
            "delay_queue_ms": 0,
        },
        "octree": {},
    }
    configs = _get_loader_configs(config)
    assert len(configs) == 1
    assert configs[0]['num_workers'] == 10
    assert configs[0]['delay_queue_ms'] == 0


TEST_CONFIG = {
    "loader_defaults": {
        "force_synchronous": False,
        "num_workers": 10,
        "delay_queue_ms": 0,
    },
    "octree": {
        "loaders": {
            0: {"num_workers": 10, "delay_queue_ms": 100},
            3: {"num_workers": 5, "delay_queue_ms": 0},
        },
    },
}


@pytest.mark.async_only
def test_get_loader_config_override():
    """Test two loaders that override the defaults."""
    configs = _get_loader_configs(TEST_CONFIG)

    # Check each config overrode the defaults.
    assert len(configs) == 2
    assert configs[0]['num_workers'] == 10
    assert configs[3]['num_workers'] == 5
    assert configs[0]['delay_queue_ms'] == 100
    assert configs[3]['delay_queue_ms'] == 0

    # Check the defaults are still there.
    assert configs[0]['force_synchronous'] is False
    assert configs[3]['force_synchronous'] is False


@pytest.mark.async_only
def test_loader_pool_group():
    from napari.components.experimental.chunk._pool_group import (
        LoaderPoolGroup,
    )

    group = LoaderPoolGroup(TEST_CONFIG)

    # Test _get_loader_priority() returns the priority of the pool we
    # should use. The one at or below the priority we give it.
    assert group._get_loader_priority(0) == 0
    assert group._get_loader_priority(1) == 0
    assert group._get_loader_priority(2) == 0
    assert group._get_loader_priority(3) == 3
    assert group._get_loader_priority(4) == 3
    assert group._get_loader_priority(5) == 3
