from unittest import mock
from urllib import error

from napari.plugins import hub

# Mock data
# ----------------------------------------------------------------------------
HUB_REPLY = b"""{"authors": [{"email": "sofroniewn@gmail.com", "name": "Nicholas Sofroniew"}],
"development_status": ["Development Status :: 4 - Beta"],
"license": "BSD-3-Clause",
"name": "napari-svg",
"project_site": "https://github.com/napari/napari-svg",
"summary": "A plugin",
"version": "0.1.6",
"visibility": "public"}"""
ANACONDA_REPLY_DIFFERENT_PYPI = b'{"versions": ["0.1.5"]}'
ANACONDA_REPLY_SAME_PYPI = b'{"versions": ["0.1.5", "0.1.6"]}'
ANACONDA_REPLY_EMPTY = b'{"versions": []}'


# Mocks
# ----------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, *, data: bytes, _error=None) -> None:
        self.data = data
        self._error = _error

    def read(self):
        if self._error:
            raise self._error

        return self.data

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return


def mocked_urlopen_valid_different(*args, **kwargs):
    if "https://api.anaconda.org" in args[0]:
        return FakeResponse(data=ANACONDA_REPLY_DIFFERENT_PYPI)
    return FakeResponse(data=HUB_REPLY)


def mocked_urlopen_valid_same(*args, **kwargs):
    if "https://api.anaconda.org" in args[0]:
        return FakeResponse(data=ANACONDA_REPLY_SAME_PYPI)
    return FakeResponse(data=HUB_REPLY)


def mocked_urlopen_valid_empty(*args, **kwargs):
    if "https://api.anaconda.org" in args[0]:
        return FakeResponse(data=ANACONDA_REPLY_EMPTY)
    return FakeResponse(data=HUB_REPLY)


def mocked_urlopen_valid_not_in_forge(*args, **kwargs):
    if "https://api.anaconda.org" in args[0]:
        return FakeResponse(
            data=ANACONDA_REPLY_EMPTY,
            _error=error.HTTPError('', 1, '', '', None),
        )
    return FakeResponse(data=HUB_REPLY)


# Tests
# ----------------------------------------------------------------------------
@mock.patch('urllib.request.urlopen', new=mocked_urlopen_valid_different)
def test_hub_plugin_info_different_pypi():
    hub.hub_plugin_info.cache_clear()
    info, is_available_in_conda_forge = hub.hub_plugin_info(
        'napari-SVG', conda_forge=True
    )
    assert is_available_in_conda_forge
    assert info.name == 'napari-svg'
    assert info.version == '0.1.5'


@mock.patch('urllib.request.urlopen', new=mocked_urlopen_valid_same)
def test_hub_plugin_info_same_as_pypi():
    hub.hub_plugin_info.cache_clear()
    info, is_available_in_conda_forge = hub.hub_plugin_info(
        'napari-SVG', conda_forge=True
    )
    assert is_available_in_conda_forge
    assert info.version == '0.1.6'


@mock.patch('urllib.request.urlopen', new=mocked_urlopen_valid_empty)
def test_hub_plugin_info_empty():
    hub.hub_plugin_info.cache_clear()
    info, is_available_in_conda_forge = hub.hub_plugin_info(
        'napari-SVG', conda_forge=True
    )
    assert not is_available_in_conda_forge
    assert info.version == '0.1.6'


@mock.patch('urllib.request.urlopen', new=mocked_urlopen_valid_empty)
def test_hub_plugin_info_forge_false():
    hub.hub_plugin_info.cache_clear()
    info, is_available_in_conda_forge = hub.hub_plugin_info(
        'napari-SVG', conda_forge=False
    )
    assert is_available_in_conda_forge
    assert info.version == '0.1.6'


@mock.patch('urllib.request.urlopen', new=mocked_urlopen_valid_not_in_forge)
def test_hub_plugin_info_not_in_forge():
    hub.hub_plugin_info.cache_clear()
    info, is_available_in_conda_forge = hub.hub_plugin_info(
        'napari-SVG', conda_forge=True
    )
    assert not is_available_in_conda_forge
    assert info.version == '0.1.6'
