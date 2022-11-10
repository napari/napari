import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from napari._qt.dialogs.qt_package_installer import InstallerQueue, InstallerTools

if TYPE_CHECKING:
    from virtualenv.run import Session


@pytest.fixture
def tmp_virtualenv(tmp_path) -> 'Session':
    virtualenv = pytest.importorskip('virtualenv')

    cmd = [str(tmp_path), '--no-setuptools', '--no-wheel', '--activators', '']
    return virtualenv.cli_run(cmd)


def conda_exe():
    if conda_exe := os.environ.get('CONDA_EXE', ''):
        pass  # in an active conda env, this is set and we take it
    elif conda_dir := os.environ.get('CONDA'):
        # $CONDA is usually defined in GHA, pointing to their bundled conda root
        conda_exe = os.path.join(conda_dir, 'condabin', 'conda')
        if os.name == 'nt':
            conda_exe += '.bat'
    if not os.path.isfile(conda_exe):
        conda_exe = 'conda.bat ' if os.name == 'nt' else 'conda'
    return conda_exe


@pytest.fixture
def tmp_conda_env(tmp_path):
    import subprocess

    try:
        subprocess.check_output(
            [
                conda_exe(),
                'create',
                '-yq',
                '-p',
                str(tmp_path),
                '--override-channels',
                '-c',
                'conda-forge',
                f'python={sys.version_info.major}.{sys.version_info.minor}',
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.output)
        raise

    return tmp_path


def test_pip_installer_tasks(qtbot, tmp_virtualenv: 'Session'):
    installer = InstallerQueue()
    with qtbot.waitSignal(installer.allFinished, timeout=20000):
        installer.install(
            tool=InstallerTools.pip,
            pkgs=['pip-install-test'],
            _executable=tmp_virtualenv.creator.exe,
        )
        installer.install(
            tool=InstallerTools.pip,
            pkgs=['typing-extensions'],
            _executable=tmp_virtualenv.creator.exe,
        )
        job_id = installer.install(
            tool=InstallerTools.pip,
            pkgs=['requests'],
            _executable=tmp_virtualenv.creator.exe,
        )
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
        job_id = installer.uninstall(
            tool=InstallerTools.pip,
            pkgs=['pip-install-test'],
            _executable=tmp_virtualenv.creator.exe,
        )

    for pth in tmp_virtualenv.creator.libs:
        if (pth / 'pip_install_test').exists():
            raise AssertionError('pip_install_test still installed')

    assert not installer.hasJobs()


def test_conda_installer(qtbot, tmp_conda_env: Path):
    conda_executable = conda_exe()
    installer = InstallerQueue()
    with qtbot.waitSignal(installer.allFinished, timeout=600_000):
        installer.install(
            tool=InstallerTools.conda,
            pkgs=['typing-extensions'],
            prefix=tmp_conda_env,
            _executable=conda_executable,
        )

    conda_meta = tmp_conda_env / "conda-meta"
    glob_pat = "typing-extensions-*.json"

    assert not installer.hasJobs()
    assert list(conda_meta.glob(glob_pat))

    with qtbot.waitSignal(installer.allFinished, timeout=600_000):
        installer.uninstall(
            tool=InstallerTools.conda,
            pkgs=['typing-extensions'],
            prefix=tmp_conda_env,
            _executable=conda_executable,
        )

    assert not installer.hasJobs()
    assert not list(conda_meta.glob(glob_pat))
