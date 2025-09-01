import gc
import sys
from unittest import mock

import pytest

import napari
from napari import __main__


@pytest.fixture
def mock_run():
    """mock to prevent starting the event loop."""
    with (
        mock.patch('napari._qt.widgets.qt_splash_screen.NapariSplashScreen'),
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
            'qtpy.QtWidgets.QApplication.exec_', lambda *_: None
        )  # revent event loop if run this test standalone
        __main__._run()

    assert len(v.layers) == 1


@mock.patch('napari._qt.qt_viewer.QtViewer._qt_open')
def test_cli_passes_kwargs(qt_open, mock_run, monkeypatch, make_napari_viewer):
    """test that we can parse layer keyword arg variants"""
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


@mock.patch('napari._qt.qt_viewer.QtViewer._qt_open')
def test_cli_passes_kwargs_stack(
    qt_open, mock_run, monkeypatch, make_napari_viewer
):
    """test that we can parse layer keyword arg variants"""
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
