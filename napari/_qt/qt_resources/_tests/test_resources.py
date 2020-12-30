import os
import platform
import shutil
import sys

import pytest
import qtpy

from napari._qt.qt_resources import build_pyqt_resources, import_resources


def test_resources():
    """Test that we can build icons and resources."""
    path, loader = import_resources(version='test')
    loader()
    os.remove(path)


@pytest.mark.skipif(
    sys.platform.startswith('win'), reason='Test not supported on windows'
)
def test_path_pollution(tmp_path, monkeypatch, capfd):
    """
    This test add fake binaries in path to check if second one is chosen.
    Its create fake binaries for ``pyqt5`` which ends with error and ``pyside2-rcc``
    which call proper executable. Because fake executable are bash scripts then
    this test may gave fake pass on Windows.
    """
    if qtpy.API == "pyqt5":
        executable_name = "pyrcc5"
    else:
        executable_name = "pyside2-rcc"

    executable_path = shutil.which(executable_name)
    (tmp_path / "path").mkdir()
    with open(tmp_path / "path" / "pyrcc5", "w") as f_p:
        f_p.write("#! /usr/bin/env python\n\n")
        f_p.write("import sys\nprint('Fake pyrcc5 executable')\n")
        f_p.write("sys.exit(10)")

    with open(tmp_path / "path" / "pyside2-rcc", "w") as f_p:
        f_p.write("#! /usr/bin/env python\n\n")
        f_p.write(
            "import sys, subprocess\n"
            f"subprocess.check_call(['{executable_path}'] + sys.argv[1:])\n"
        )

    with open(tmp_path / "path" / "python", "w") as f_p:
        f_p.write("#! /usr/bin/env bash\n\nset +x\n")
        f_p.write(f"{sys.executable} $@\n")

    (tmp_path / "path" / "pyrcc5").chmod(0o777)
    (tmp_path / "path" / "pyside2-rcc").chmod(0o777)
    (tmp_path / "path" / "python").chmod(0o777)

    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(tmp_path / "path"), os.environ["PATH"]])
    )
    monkeypatch.setattr(sys, "executable", str(tmp_path / "path" / "python"))
    build_pyqt_resources(str(tmp_path / "resources.py"))
    assert (tmp_path / "resources.py").exists()

    if platform.system() != "Windows":
        captured = capfd.readouterr()
        assert "Fake pyrcc5 executable" in captured.out
