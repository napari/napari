import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from napari._qt.dialogs.qt_installer import CondaInstaller, PipInstaller

if TYPE_CHECKING:
    from virtualenv.run import Session


@pytest.fixture
def tmp_virtualenv(tmp_path) -> 'Session':
    virtualenv = pytest.importorskip('virtualenv')

    cmd = [str(tmp_path), '--no-setuptools', '--no-wheel', '--activators', '']
    return virtualenv.cli_run(cmd)


@pytest.fixture
def tmp_conda_env(tmp_path):
    import subprocess

    subprocess.check_call(
        [
            'conda',
            'create',
            '-y',
            '-p',
            str(tmp_path),
            f'python={sys.version_info.major}.{sys.version_info.minor}',
        ]
    )
    return tmp_path


def test_pip_installer(qtbot, tmp_virtualenv: 'Session'):
    installer = PipInstaller(python_interpreter=tmp_virtualenv.creator.exe)
    with qtbot.waitSignal(installer.allFinished, timeout=20000):
        installer.install(['pip-install-test'])
        installer.install(['typing-extensions'])
        job_id = installer.install(['requests'])
        assert isinstance(job_id, int)
        installer.cancel(job_id)

    assert not installer.hasJobs()

    pkgs = 0
    for pth in tmp_virtualenv.creator.libs:
        if (pth / 'pip_install_test').exists():
            pkgs += 1
        if (pth / 'typing_extensions.py').exists():
            pkgs += 1
        if (pth / 'requests').exists():
            raise AssertionError('requests got installed')

    if pkgs < 2:
        raise AssertionError('package was not installed')

    with qtbot.waitSignal(installer.allFinished, timeout=10000):
        job_id = installer.uninstall(['pip-install-test'])

    for pth in tmp_virtualenv.creator.libs:
        if (pth / 'pip_install_test').exists():
            raise AssertionError('pip_install_test still installed')

    assert not installer.hasJobs()


def test_conda_installer(qtbot, tmp_conda_env: Path):
    installer = CondaInstaller()
    with qtbot.waitSignal(installer.allFinished, timeout=10000):
        installer.install(['typing-extensions'], prefix=tmp_conda_env)
        installer.waitForFinished()

    assert not installer.hasJobs()

    with qtbot.waitSignal(installer.allFinished, timeout=10000):
        installer.uninstall(['typing-extensions'], prefix=tmp_conda_env)
        installer.waitForFinished()

    assert not installer.hasJobs()
