import contextlib
import shutil
import sys
from pathlib import Path
from typing import Deque, Optional, Sequence, Tuple

from qtpy.QtCore import QObject, QProcess, QProcessEnvironment

from ...utils._appdirs import user_plugin_dir, user_site_packages
from ...utils.misc import running_as_bundled_app

JobId = int


class InstallerProcess(QProcess):
    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        self.setProcessEnvironment(env)
        self._queue: Deque[Tuple[str, ...]] = Deque()
        self.finished.connect(self._on_process_finished)

    def install(
        self, pkg_list: Sequence[str], *, prefix: Optional[str] = None
    ) -> JobId:
        return self._queue_args(self._get_install_args(pkg_list, prefix))

    def uninstall(
        self, pkg_list: Sequence[str], *, prefix: Optional[str] = None
    ) -> JobId:
        return self._queue_args(self._get_uninstall_args(pkg_list))

    def _queue_args(self, args) -> JobId:
        self._queue.append(args)
        self._process_queue()
        return hash(args)

    def cancel(self, job_id: JobId):
        for i, args in enumerate(self._queue):
            if hash(args) == job_id:
                self.terminate() if i == 0 else self._queue.remove(args)
                return
        raise ValueError(f"No job with id {job_id}")

    def cancelAll(self):
        self._queue.clear()
        self.terminate()

    def hasJobs(self) -> bool:
        return bool(self._queue)

    def _get_install_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        return tuple(pkg_list)

    def _get_uninstall_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        return tuple(pkg_list)

    def _process_queue(self):
        if not self._queue:
            return
        self.setArguments(list(self._queue[0]))
        self.start()

    def _on_process_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus
    ):
        with contextlib.suppress(IndexError):
            self._queue.popleft()
        self._process_queue()

    def waitForFinished(self, msecs: int = 10000) -> bool:
        while self.hasJobs():
            super().waitForFinished(msecs)


class PipInstaller(InstallerProcess):
    def __init__(
        self, parent: Optional[QObject] = None, python_interpreter: str = ''
    ) -> None:
        super().__init__(parent)
        self.setProgram(str(python_interpreter or _get_python_exe()))

    def _get_install_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        cmd = ['-m', 'pip', 'install', '--upgrade']
        if prefix is not None:
            cmd.extend(['--prefix', str(prefix)])
        if running_as_bundled_app() and sys.platform.startswith('linux'):
            cmd += ['--no-warn-script-location', '--prefix', user_plugin_dir()]
        return tuple(cmd + list(pkg_list))

    def _get_uninstall_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        return tuple(['-m', 'pip', 'uninstall', '-y'] + list(pkg_list))


class CondaInstaller(InstallerProcess):
    def __init__(
        self, parent: Optional[QObject] = None, use_mamba: bool = True
    ) -> None:
        super().__init__(parent)
        program = 'mamba' if use_mamba and shutil.which('mamba') else 'conda'
        self.setProgram(program)
        self.channels = ('conda-forge',)
        self._default_prefix = (
            sys.prefix if (Path(sys.prefix) / "conda-meta").is_dir() else None
        )

    def _get_install_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        return self._get_args('install', prefix, pkg_list)

    def _get_uninstall_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        return self._get_args('remove', prefix, pkg_list)

    # TODO Rename this here and in `_get_install_args` and `_get_uninstall_args`
    def _get_args(self, arg0, prefix, pkg_list):
        cmd = [arg0, '-y']
        prefix = prefix or self._default_prefix
        if prefix is not None:
            cmd.extend(['--prefix', str(prefix)])
        for channel in self.channels:
            cmd.extend(["-c", channel])
        return tuple(cmd + list(pkg_list))


def _get_python_exe():
    if sys.executable.endswith("napari") and sys.platform == 'darwin':
        # sys.prefix should be <napari.app>/Contents/Resources/Support/Python/Resources
        if (python := Path(sys.prefix) / "bin" / "python3").is_file():
            return str(python)
    return sys.executable


if __name__ == '__main__':

    i = InstallerProcess()
    i.setProgram(sys.executable)
    i.readyReadStandardOutput.connect(lambda: print(i.readAllStandardOutput()))
