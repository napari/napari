import os
import platform

import qtpy

from napari.resources import build_pyqt_resources, import_resources


def test_resources():
    """Test that we can build icons and resources."""
    out = import_resources(version='test')
    os.remove(out)


def test_path_pollution(tmp_path, monkeypatch, capfd):
    if qtpy.API == "pyqt5":
        file_name = "pyside2-rcc"
    else:
        file_name = "pyrcc5"
    (tmp_path / "path").mkdir()
    with open(tmp_path / "path" / file_name, "w") as f_p:
        f_p.write("#! /usr/bin/env bash\n\n")
        f_p.write(f"echo Fake {file_name} executable\n")
        f_p.write("exit 10")

    (tmp_path / "path" / file_name).chmod(0o777)

    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(tmp_path / "path"), os.environ["PATH"]])
    )
    build_pyqt_resources(str(tmp_path / "resourcses.py"))
    assert (tmp_path / "resourcses.py").exists()

    if qtpy.API == "pyside2" and platform.system() != "Windows":
        captured = capfd.readouterr()
        assert f"Fake {file_name} executable" in captured.out
