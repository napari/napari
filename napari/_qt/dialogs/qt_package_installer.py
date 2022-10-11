import contextlib
import os
import shutil
import sys
from logging import getLogger
from pathlib import Path
from tempfile import gettempdir
from typing import Deque, Optional, Sequence, Tuple

from qtpy.QtCore import QObject, QProcess, QProcessEnvironment, Signal
from qtpy.QtWidgets import QTextEdit

from ...plugins.pypi import _user_agent
from ...utils._appdirs import user_plugin_dir, user_site_packages
from ...utils.misc import running_as_bundled_app

JobId = int
log = getLogger(__name__)


class AbstractInstaller(QProcess):
    """Abstract base class for package installers (pip, conda, etc)."""

    # emitted when all jobs are finished
    # not to be confused with finished, which is emitted when each job is finished
    allFinished = Signal()

    # abstract method
    def _modify_env(self, env: QProcessEnvironment):
        raise NotImplementedError()

    # abstract method
    def _get_install_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        raise NotImplementedError()

    # abstract method
    def _get_uninstall_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        raise NotImplementedError()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._queue: Deque[Tuple[str, ...]] = Deque()
        self._output_widget = None

        self.setProcessChannelMode(QProcess.MergedChannels)
        self.readyReadStandardOutput.connect(self._on_stdout_ready)
        self.readyReadStandardError.connect(self._on_stderr_ready)

        env = QProcessEnvironment.systemEnvironment()
        self._modify_env(env)
        self.setProcessEnvironment(env)

        self.finished.connect(self._on_process_finished)
        self.errorOccurred.connect(self._on_error_occurred)

    # -------------------------- Public API ------------------------------
    def install(
        self, pkg_list: Sequence[str], *, prefix: Optional[str] = None
    ) -> JobId:
        """Install packages in `pkg_list` into `prefix`.

        Parameters
        ----------
        pkg_list : Sequence[str]
            List of packages to install.
        prefix : Optional[str], optional
            Optional prefix to install packages into.

        Returns
        -------
        JobId : int
            ID that can be used to cancel the process.
        """
        return self._queue_args(self._get_install_args(pkg_list, prefix))

    def uninstall(
        self, pkg_list: Sequence[str], *, prefix: Optional[str] = None
    ) -> JobId:
        """Uninstall packages in `pkg_list` from `prefix`.

        Parameters
        ----------
        pkg_list : Sequence[str]
            List of packages to uninstall.
        prefix : Optional[str], optional
            Optional prefix from which to uninstall packages.

        Returns
        -------
        JobId : int
            ID that can be used to cancel the process.
        """
        return self._queue_args(self._get_uninstall_args(pkg_list, prefix))

    def cancel(self, job_id: Optional[JobId] = None):
        """Cancel `job_id` if it is running.

        Parameters
        ----------
        job_id : Optional[JobId], optional
            Job ID to cancel.  If not provided, cancel all jobs.
        """

        def end_process():
            if os.name == 'nt':
                self.kill()
            else:
                self.terminate()
            if self._output_widget:
                self._output_widget.append("\nTask was cancelled by the user.")

        if job_id is None:
            # cancel all jobs
            self._queue.clear()
            end_process()
            return

        for i, args in enumerate(self._queue):
            if hash(args) == job_id:
                if i == 0:  # first in queue, currently running
                    end_process()
                else:  # still pending, just remove from queue
                    self._queue.remove(args)
                return
        msg = f"No job with id {job_id}. Current queue:\n - "
        msg += "\n - ".join(
            [f"{hash(args)} -> {args}" for args in self._queue]
        )
        raise ValueError(msg)

    def waitForFinished(self, msecs: int = 10000) -> bool:
        """Block and wait for all jobs to finish.

        Parameters
        ----------
        msecs : int, optional
            Time to wait, by default 10000
        """
        while self.hasJobs():
            super().waitForFinished(msecs)
        return True

    def hasJobs(self) -> bool:
        """True if there are jobs remaining in the queue."""
        return bool(self._queue)

    def set_output_widget(self, output_widget: QTextEdit):
        if output_widget:
            self._output_widget = output_widget

    # -------------------------- Private methods ------------------------------
    def _log(self, msg: str):
        log.debug(msg)
        if self._output_widget:
            self._output_widget.append(msg)

    def _queue_args(self, args) -> JobId:
        self._queue.append(args)
        self._process_queue()
        return hash(args)

    def _process_queue(self):
        if not self._queue:
            self.allFinished.emit()
            return
        self.setArguments(list(self._queue[0]))
        # this might throw a warning because the same process
        # was already running but it's ok
        self._log(f"Starting '{self.program()}' with args {self.arguments()}")
        self.start()

    def _on_process_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus
    ):
        self._on_process_done(exit_code=exit_code, exit_status=exit_status)

    def _on_error_occurred(self, error: QProcess.ProcessError):
        self._on_process_done(error=error)

    def _on_process_done(
        self,
        exit_code: Optional[int] = None,
        exit_status: Optional[QProcess.ExitStatus] = None,
        error: Optional[QProcess.ProcessError] = None,
    ):
        with contextlib.suppress(IndexError):
            self._queue.popleft()
        msg = "Task finished with "
        if error:
            msg += f"with errors! Code: {error}."
        else:
            msg += f"exit code {exit_code} with status {exit_status}."
        self._log(msg)
        self._process_queue()

    def _on_stdout_ready(self):
        text = self.readAllStandardOutput().data().decode()
        if text:
            self._log(text)

    def _on_stderr_ready(self):
        text = self.readAllStandardError().data().decode()
        if text:
            self._log(text)


class PipInstaller(AbstractInstaller):
    def __init__(
        self, parent: Optional[QObject] = None, python_interpreter: str = ''
    ) -> None:
        super().__init__(parent)
        self.setProgram(str(python_interpreter or _get_python_exe()))

    def _modify_env(self, env: QProcessEnvironment):
        # patch process path
        combined_paths = os.pathsep.join(
            [
                user_site_packages(),
                env.systemEnvironment().value("PYTHONPATH"),
            ]
        )
        env.insert("PYTHONPATH", combined_paths)
        env.insert("PIP_USER_AGENT_USER_DATA", _user_agent())

    def _get_install_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        cmd = ['-m', 'pip', 'install', '--upgrade']
        if prefix is not None:
            cmd.extend(['--prefix', str(prefix)])
        elif running_as_bundled_app(False) and sys.platform.startswith(
            'linux'
        ):
            cmd.extend(
                ['--no-warn-script-location', '--prefix', user_plugin_dir()]
            )
        return tuple(cmd + list(pkg_list))

    def _get_uninstall_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        return tuple(['-m', 'pip', 'uninstall', '-y'] + list(pkg_list))


class CondaInstaller(AbstractInstaller):
    default_channels = ('conda-forge',)

    def __init__(
        self,
        parent: Optional[QObject] = None,
        use_mamba: bool = True,
        channels: Optional[Sequence[str]] = None,
        _conda_exe: Optional[os.PathLike] = None,
    ) -> None:
        _bat = ".bat" if os.name == "nt" else ""
        if _conda_exe:
            self._bin = _conda_exe
        else:
            self._bin = (
                f'mamba{_bat}'
                if use_mamba and shutil.which(f'mamba{_bat}')
                else f'conda{_bat}'
            )
        super().__init__(parent)
        self.setProgram(self._bin)
        # TODO: make configurable per install once plugins can request it
        self.channels = channels or self.default_channels
        self._default_prefix = (
            sys.prefix if (Path(sys.prefix) / "conda-meta").is_dir() else None
        )

    def _napari_pin(self):
        from ..._version import version, version_tuple

        if "rc" in version or "dev" in version:
            # dev or rc versions might not be available in public channels
            # but only installed locally - if we try to pin those, mamba
            # will fail to pin it because there's no record of that version
            # in the remote index, only locally; to work around this bug
            # we will have to pin to e.g. 0.4.* instead of 0.4.17.* for now
            return ".".join([str(part) for part in version_tuple[:2]])
        return ".".join([str(part) for part in version_tuple[:3]])

    def _modify_env(self, env: QProcessEnvironment):
        PINNED = 'CONDA_PINNED_PACKAGES'
        system_pins = f"&{env.value(PINNED)}" if env.contains(PINNED) else ""
        env.insert(PINNED, f"napari={self._napari_pin()}{system_pins}")
        env.insert('CONDA_VERBOSITY', '3')

        if os.name == "nt":
            if not env.contains("TEMP"):
                temp = gettempdir()
                env.insert("TMP", temp)
                env.insert("TEMP", temp)
            if not env.contains("USERPROFILE"):
                env.insert("HOME", os.path.expanduser("~"))
                env.insert("USERPROFILE", os.path.expanduser("~"))

    def _get_install_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        return self._get_args('install', pkg_list, prefix)

    def _get_uninstall_args(
        self, pkg_list: Sequence[str], prefix: Optional[str] = None
    ) -> Tuple[str, ...]:
        return self._get_args('remove', pkg_list, prefix)

    def _get_args(self, arg0, pkg_list: Sequence[str], prefix: Optional[str]):
        cmd = [arg0, '-y', '--override-channels']
        if prefix := str(prefix or self._default_prefix):
            cmd.extend(['--prefix', prefix])
        for channel in self.channels:
            cmd.extend(["-c", channel])
        return tuple(cmd + list(pkg_list))


def _get_python_exe():
    # Note: is_bundled_app() returns False even if using a Briefcase bundle...
    # Workaround: see if sys.executable is set to something something napari on Mac
    if sys.executable.endswith("napari") and sys.platform == 'darwin':
        # sys.prefix should be <napari.app>/Contents/Resources/Support/Python/Resources
        if (python := Path(sys.prefix) / "bin" / "python3").is_file():
            return str(python)
    return sys.executable
