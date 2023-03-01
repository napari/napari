import sys

import pytest
from napari_plugin_engine import PluginError

from napari.plugins import exceptions


# monkeypatch fixture is from pytest
@pytest.mark.parametrize('as_html', (True, False), ids=['as_html', 'as_text'])
@pytest.mark.parametrize('cgitb', (True, False), ids=['cgitb', 'ipython'])
def test_format_exceptions(cgitb, as_html, monkeypatch):
    if cgitb:
        monkeypatch.setitem(sys.modules, 'IPython.core.ultratb', None)
    monkeypatch.setattr(
        exceptions,
        'standard_metadata',
        lambda x: {'package': 'test-package', 'version': '0.1.0'},
    )

    # we make sure to actually raise the exceptions,
    # otherwise they will miss the __traceback__ attributes.
    try:
        try:
            raise ValueError('cause')  # noqa TRY301
        except ValueError as e:
            raise PluginError(
                'some error',
                plugin_name='test_plugin',
                plugin="mock",
                cause=e,
            ) from e
    except PluginError:
        pass

    formatted = exceptions.format_exceptions('test_plugin', as_html=as_html)
    assert "some error" in formatted
    assert "version: 0.1.0" in formatted
    assert "plugin package: test-package" in formatted

    assert exceptions.format_exceptions('nonexistent', as_html=as_html) == ''
