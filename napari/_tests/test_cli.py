from napari import __main__
import sys
import pytest
from contextlib import contextmanager


def test_cli_works(monkeypatch, capsys):
    """Test the cli runs and shows help"""
    monkeypatch.setattr(sys, 'argv', ['napari', '-h'])
    with pytest.raises(SystemExit):
        __main__.main()
    assert 'napari command line viewer.' in str(capsys.readouterr())


def test_cli_shows_plugins(monkeypatch, capsys):
    """Test the cli --info runs and shows plugins"""
    monkeypatch.setattr(sys, 'argv', ['napari', '--info'])
    with pytest.raises(SystemExit):
        __main__.main()
    assert 'svg' in str(capsys.readouterr())


def test_cli_parses_unknowns(monkeypatch):
    """test that we can parse layer keyword arg variants"""

    def assert_kwargs(*args, **kwargs):
        assert args == (["file"],)
        assert kwargs['contrast_limits'] == (0, 1)

    @contextmanager
    def gui_qt(**kwargs):
        yield

    # testing all the variants of literal_evals
    monkeypatch.setattr(__main__, 'view_path', assert_kwargs)
    monkeypatch.setattr(__main__, 'gui_qt', gui_qt)
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits', '(0, 1)'])
        __main__.main()
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits', '(0,1)'])
        __main__.main()
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits=(0, 1)'])
        __main__.main()
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits=(0,1)'])
        __main__.main()


def test_cli_raises(monkeypatch):
    """test that unknown kwargs raise the correct errors."""
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file', '--nonsense'])
        with pytest.raises(SystemExit) as e:
            __main__.main()
        assert str(e.value) == 'error: unrecognized arguments: --nonsense'

    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file', '--gamma'])
        with pytest.raises(SystemExit) as e:
            __main__.main()
        assert str(e.value) == 'error: argument --gamma expected one argument'
