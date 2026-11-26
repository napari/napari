from __future__ import annotations

import gc
import os
import sys
from typing import TYPE_CHECKING
from unittest import mock

import pytest

import napari
from napari import __main__
from napari._main import main as console_main

if TYPE_CHECKING:
    from napari.utils._testsupport import ViewerFactory


@pytest.fixture
def mock_run():
    """mock to prevent starting the event loop."""
    with (
        mock.patch('napari.run'),
    ):
        yield napari.run


def test_cli_works(monkeypatch, capsys):
    """Test the cli runs and shows help"""
    monkeypatch.setattr(sys, 'argv', ['napari', '-h'])
    with pytest.raises(SystemExit):
        __main__._run()
    assert 'napari command line viewer.' in str(capsys.readouterr())


def test_cli_shows_plugins(monkeypatch, capsys, tmp_plugin):
    """Test the cli --info runs and shows plugins"""
    monkeypatch.setattr(sys, 'argv', ['napari', '--info'])
    with pytest.raises(SystemExit):
        __main__._run()
    assert tmp_plugin.name in str(capsys.readouterr())


@pytest.mark.parametrize('console_script', [False, True])
def test_cli_starts_without_qtreload(
    console_script: bool,
    monkeypatch: pytest.MonkeyPatch,
    make_napari_viewer: ViewerFactory,
    mock_run: mock.MagicMock,
) -> None:
    monkeypatch.delenv('NAPARI_DEV', raising=False)
    monkeypatch.setitem(sys.modules, 'qtreload', None)
    monkeypatch.setattr(sys, 'argv', ['napari'])
    viewer = make_napari_viewer()
    monkeypatch.setattr(__main__, 'Viewer', lambda: viewer)
    if console_script:
        with (
            mock.patch('napari._qt.qt_event_loop.run', mock_run),
            pytest.raises(SystemExit) as exc,
        ):
            console_main()
        assert exc.value.code == 0
    else:
        __main__.main()
    mock_run.assert_called_once_with(gui_exceptions=True)


def test_cli_dev_modules_with_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('NAPARI_DEV', '0')
    monkeypatch.setenv('NAPARI_DEV_MODULES', 'previous_module')
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'napari',
            '--dev',
            '--dev-module',
            'first_plugin',
            '--dev_module',
            'second_plugin',
            'image.tif',
        ],
    )
    viewer = mock.Mock()
    monkeypatch.setattr(__main__, 'Viewer', lambda: viewer)

    with mock.patch('napari._hot_reload.version', return_value='0.3.0'):
        assert __main__._build_viewer() is viewer

    assert os.environ['NAPARI_DEV'] == '1'
    assert os.environ['NAPARI_DEV_MODULES'] == 'first_plugin,second_plugin'
    viewer._window._qt_viewer._qt_open.assert_called_once_with(
        ['image.tif'], stack=[], plugin=None, layer_type=None
    )


@pytest.mark.parametrize('installed_version', [None, '0.2.3'])
def test_cli_dev_requires_qtreload(
    installed_version: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from importlib.metadata import PackageNotFoundError

    from napari import _hot_reload

    def version(name: str) -> str:
        if installed_version is None:
            raise PackageNotFoundError(name)
        return installed_version

    monkeypatch.setattr(_hot_reload, 'version', version)
    monkeypatch.setattr(sys, 'argv', ['napari', '--dev'])

    with pytest.raises(SystemExit) as exc:
        __main__.main()

    assert exc.value.code == 2
    assert 'napari[reload]' in capsys.readouterr().err


def test_cli_parses_unknowns(mock_run, monkeypatch, make_napari_viewer):
    """test that we can parse layer keyword arg variants"""
    mocked_viewer = (
        make_napari_viewer()
    )  # our mock view_path will return this object

    def assert_kwargs(*args, **kwargs):
        assert ['file'] in args
        assert kwargs['contrast_limits'] == (0, 1)

    # testing all the variants of literal_evals
    with mock.patch('napari.__main__.Viewer', return_value=mocked_viewer):
        monkeypatch.setattr(
            napari.components.viewer_model.ViewerModel, 'open', assert_kwargs
        )
        with monkeypatch.context() as m:
            m.setattr(
                sys, 'argv', ['n', 'file', '--contrast-limits', '(0, 1)']
            )
            __main__._run()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits', '(0,1)'])
            __main__._run()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits=(0, 1)'])
            __main__._run()
        with monkeypatch.context() as m:
            m.setattr(sys, 'argv', ['n', 'file', '--contrast-limits=(0,1)'])
            __main__._run()


def test_cli_raises(monkeypatch):
    """test that unknown kwargs raise the correct errors."""
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file', '--nonsense'])
        with pytest.raises(SystemExit) as e:
            __main__._run()
        assert str(e.value) == 'error: unrecognized argument: --nonsense'

    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file', '--gamma'])
        with pytest.raises(SystemExit) as e:
            __main__._run()
        assert str(e.value) == 'error: argument --gamma expected one argument'


@pytest.mark.usefixtures('builtins')
def test_cli_runscript(monkeypatch, tmp_path, make_napari_viewer):
    """Test that running napari script.py runs a script"""
    v = make_napari_viewer()
    script = tmp_path / 'test.py'
    script.write_text('import napari; v = napari.Viewer(); v.add_points([])')

    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', str(script)])
        m.setattr(__main__, 'Viewer', lambda: v)
        m.setattr(
            'qtpy.QtWidgets.QApplication.exec', lambda *_: None
        )  # revent event loop if run this test standalone
        __main__._run()

    assert len(v.layers) == 1


def test_cli_passes_kwargs(
    mock_run, monkeypatch, make_napari_viewer, mock_qt_method
):
    """test that we can parse layer keyword arg variants"""
    qt_open = mock_qt_method('napari._qt.qt_viewer.QtViewer._qt_open')
    v = make_napari_viewer()

    with (
        mock.patch('napari.__main__.Viewer', return_value=v),
        monkeypatch.context() as m,
    ):
        m.setattr(sys, 'argv', ['n', 'file', '--name', 'some name'])
        __main__._run()

    qt_open.assert_called_once_with(
        ['file'],
        stack=[],
        plugin=None,
        layer_type=None,
        name='some name',
    )
    mock_run.assert_called_once_with(gui_exceptions=True)


def test_cli_passes_kwargs_stack(
    mock_run, monkeypatch, make_napari_viewer, mock_qt_method
):
    """test that we can parse layer keyword arg variants"""
    qt_open = mock_qt_method('napari._qt.qt_viewer.QtViewer._qt_open')
    v = make_napari_viewer()

    with (
        mock.patch('napari.__main__.Viewer', return_value=v),
        monkeypatch.context() as m,
    ):
        m.setattr(
            sys,
            'argv',
            [
                'n',
                'file',
                '--stack',
                'file1',
                'file2',
                '--stack',
                'file3',
                'file4',
                '--name',
                'some name',
            ],
        )
        __main__._run()

    qt_open.assert_called_once_with(
        ['file'],
        stack=[['file1', 'file2'], ['file3', 'file4']],
        plugin=None,
        layer_type=None,
        name='some name',
    )
    mock_run.assert_called_once_with(gui_exceptions=True)


def test_cli_retains_viewer_ref(mock_run, monkeypatch, make_napari_viewer):
    """Test that napari.__main__ is retaining a reference to the viewer."""
    mocked_viewer = (
        make_napari_viewer()
    )  # our mock view_path will return this object
    ref_count = None  # counter that will be updated before __main__._run()

    def _check_refs(**kwargs):
        # when run() is called in napari.__main__, we will call this function
        # it forces garbage collection, and then makes sure that at least one
        # additional reference to our viewer exists.
        gc.collect()
        if sys.getrefcount(mocked_viewer) <= ref_count:  # pragma: no cover
            raise AssertionError(
                'Reference to napari.viewer has been lost by '
                'the time the event loop started in napari.__main__'
            )

    mock_run.side_effect = _check_refs
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['napari', 'path/to/file.tif'])
        # return our local v
        with mock.patch(
            'napari.__main__.Viewer', return_value=mocked_viewer
        ) as mock_viewer:
            ref_count = sys.getrefcount(
                mocked_viewer
            )  # count current references
            # mock gui open so we're not opening dialogs/throwing errors on fake path
            with mock.patch(
                'napari._qt.qt_viewer.QtViewer._qt_open', return_value=None
            ) as mock_viewer_open:
                __main__._run()
            mock_viewer.assert_called_once()
            mock_viewer_open.assert_called_once()


def test_cli_plugin_info(monkeypatch):
    """--plugin-info delegates to npe2.cli.list_ and exits."""
    monkeypatch.setattr(sys, 'argv', ['napari', '--plugin-info'])
    with (
        mock.patch('npe2.cli.list_') as mock_list,
        pytest.raises(SystemExit),
    ):
        __main__._run()
    mock_list.assert_called_once()
