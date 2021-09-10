from pathlib import Path

from napari.utils.history import (
    get_open_history,
    get_save_history,
    update_open_history,
    update_save_history,
)


def test_open_history():
    open_history = get_open_history()
    assert len(open_history) == 1
    assert str(Path.home()) in open_history


def test_update_open_history(tmpdir):
    new_folder = Path(tmpdir) / "some-file.svg"
    update_open_history(new_folder)
    assert str(new_folder.parent) in get_open_history()


def test_save_history():
    save_history = get_save_history()
    assert len(save_history) == 1
    assert str(Path.home()) in save_history


def test_update_save_history(tmpdir):
    new_folder = Path(tmpdir) / "some-file.svg"
    update_save_history(new_folder)
    assert str(new_folder.parent) in get_save_history()
