import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from qtpy.QtCore import QProcessEnvironment

from napari._qt.dialogs.qt_package_installer import (
    AbstractInstallerTool,
    CondaInstallerTool,
    InstallerQueue,
    InstallerTools,
    PipInstallerTool,
)

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

    try:
        subprocess.check_output(
            [
                CondaInstallerTool.executable(),
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


def test_pip_installer_tasks(qtbot, tmp_virtualenv: 'Session', monkeypatch):
    installer = InstallerQueue()
    monkeypatch.setattr(
        PipInstallerTool, "executable", lambda *a: tmp_virtualenv.creator.exe
    )
    with qtbot.waitSignal(installer.allFinished, timeout=20000):
        installer.install(
            tool=InstallerTools.pip,
            pkgs=['pip-install-test'],
        )
        installer.install(
            tool=InstallerTools.pip,
            pkgs=['typing-extensions'],
        )
        job_id = installer.install(
            tool=InstallerTools.pip,
            pkgs=['requests'],
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

    assert pkgs >= 2, 'package was not installed'

    with qtbot.waitSignal(installer.allFinished, timeout=10000):
        job_id = installer.uninstall(
            tool=InstallerTools.pip,
            pkgs=['pip-install-test'],
        )

    for pth in tmp_virtualenv.creator.libs:
        assert not (
            pth / 'pip_install_test'
        ).exists(), 'pip_install_test still installed'

    assert not installer.hasJobs()


def _assert_exit_code_not_zero(
    self, exit_code=None, exit_status=None, error=None
):
    errors = []
    if exit_code == 0:
        errors.append("- 'exit_code' should have been non-zero!")
    if error is not None:
        errors.append("- 'error' should have been None!")
    if errors:
        raise Exception("\n".join(errors))
    return self._on_process_done_original(exit_code, exit_status, error)


class _NonExistingTool(AbstractInstallerTool):
    def executable(self):
        return f"this-tool-does-not-exist-{hash(time.time())}"

    def arguments(self):
        return ()

    def environment(self, env=None):
        return QProcessEnvironment.systemEnvironment()


def _assert_error_used(self, exit_code=None, exit_status=None, error=None):
    errors = []
    if error is None:
        errors.append("- 'error' should have been populated!")
    if exit_code is not None:
        errors.append("- 'exit_code' should not have been populated!")
    if errors:
        raise Exception("\n".join(errors))
    return self._on_process_done_original(exit_code, exit_status, error)


def test_installer_failures(qtbot, tmp_virtualenv: 'Session', monkeypatch):
    installer = InstallerQueue()
    monkeypatch.setattr(
        PipInstallerTool, "executable", lambda *a: tmp_virtualenv.creator.exe
    )

    # CHECK 1) Errors should trigger finished and allFinished too
    with qtbot.waitSignal(installer.allFinished, timeout=10000):
        installer.install(
            tool=InstallerTools.pip,
            pkgs=[f'this-package-does-not-exist-{hash(time.time())}'],
        )

    # Keep a reference before we monkey patch stuff
    installer._on_process_done_original = installer._on_process_done

    # CHECK 2) Non-existing packages should return non-zero
    monkeypatch.setattr(
        installer, "_on_process_done", _assert_exit_code_not_zero
    )
    with qtbot.waitSignal(installer.allFinished, timeout=10000):
        installer.install(
            tool=InstallerTools.pip,
            pkgs=[f'this-package-does-not-exist-{hash(time.time())}'],
        )

    # CHECK 3) Non-existing tools should fail to start
    monkeypatch.setattr(installer, "_on_process_done", _assert_error_used)
    monkeypatch.setattr(installer, "_get_tool", lambda *a: _NonExistingTool)
    with qtbot.waitSignal(installer.allFinished, timeout=10000):
        installer.install(
            tool=_NonExistingTool,
            pkgs=[f'this-package-does-not-exist-{hash(time.time())}'],
        )


@pytest.mark.skipif(
    not CondaInstallerTool.available(), reason="Conda is not available."
)
def test_conda_installer(qtbot, tmp_conda_env: Path):
    installer = InstallerQueue()

    with qtbot.waitSignal(installer.allFinished, timeout=600_000):
        installer.install(
            tool=InstallerTools.conda,
            pkgs=['typing-extensions'],
            prefix=tmp_conda_env,
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
        )

    assert not installer.hasJobs()
    assert not list(conda_meta.glob(glob_pat))


def test_constraints_are_in_sync():
    conda_constraints = sorted(CondaInstallerTool.constraints())
    pip_constraints = sorted(PipInstallerTool.constraints())

    assert len(conda_constraints) == len(pip_constraints)

    name_re = re.compile(r"([a-z0-9_\-]+).*")
    for conda_constraint, pip_constraint in zip(
        conda_constraints, pip_constraints
    ):
        conda_name = name_re.match(conda_constraint).group(1)
        pip_name = name_re.match(pip_constraint).group(1)
        assert conda_name == pip_name
