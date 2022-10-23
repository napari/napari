import contextlib
import os
import shutil
import sys
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from tempfile import gettempdir
from typing import Deque, Literal, Optional, Sequence, Tuple, Union

from npe2 import PluginManager
from qtpy.QtCore import QObject, QProcess, QProcessEnvironment, Signal
from qtpy.QtWidgets import QTextEdit

from ...plugins import plugin_manager
from ...plugins.pypi import _user_agent
from ...utils._appdirs import user_plugin_dir, user_site_packages
from ...utils.misc import running_as_bundled_app

JobId = int
ActionType = Union[Literal["install"], Literal["uninstall"]]
TaskHandler = Union[Literal["pip"], Literal["conda"]]
log = getLogger(__name__)


@dataclass(frozen=True)
class AbstractInstallerTask:
    action: ActionType
    pkgs: Tuple[str, ...]
    origins: Tuple[str, ...] = ()
    prefix: Optional[str] = None
    _executable: Optional[str] = None

    @property
    def ident(self):
        return hash((self.action, *self.pkgs, *self.origins, self.prefix))

    # abstract method
    def executable(self):
        raise NotImplementedError()

    # abstract method
    def arguments(
        self,
        action: ActionType,
        pkgs: Tuple[str, ...],
        prefix: Optional[str] = None,
    ):
        raise NotImplementedError()

    # abstract method
    def environment(
        self, env: QProcessEnvironment = None
    ) -> QProcessEnvironment:
        raise NotImplementedError()


class PipInstallerTask(AbstractInstallerTask):
    def executable(self):
        if self._executable:
            return str(self._executable)
        return str(_get_python_exe())

    def arguments(self) -> Tuple[str, ...]:
        args = ['-m', 'pip']
        if self.action == "install":
            args += ['install', '--upgrade']
            for origin in self.origins:
                args += ['--extra-index-url', origin]
        else:
            args += ['uninstall', '-y']
        if 10 <= log.getEffectiveLevel() < 30:  # DEBUG level
            args.append('-vvv')
        if self.prefix is not None:
            args.extend(['--prefix', str(self.prefix)])
        elif running_as_bundled_app(False) and sys.platform.startswith(
            'linux'
        ):
            args += [
                '--no-warn-script-location',
                '--prefix',
                user_plugin_dir(),
            ]
        return (*args, *self.pkgs)

    def environment(
        self, env: QProcessEnvironment = None
    ) -> QProcessEnvironment:
        if env is None:
            env = QProcessEnvironment.systemEnvironment()
        combined_paths = os.pathsep.join(
            [
                user_site_packages(),
                env.systemEnvironment().value("PYTHONPATH"),
            ]
        )
        env.insert("PYTHONPATH", combined_paths)
        env.insert("PIP_USER_AGENT_USER_DATA", _user_agent())
        return env


class CondaInstallerTask(AbstractInstallerTask):
    def executable(self):
        if self._executable:
            return str(self._executable)
        _bat = ".bat" if os.name == "nt" else ""
        if exe := os.environ.get("MAMBA_EXE", shutil.which(f'mamba{_bat}')):
            return exe
        _exe = ".exe" if os.name == "nt" else ""
        if exe := os.environ.get("CONDA_EXE", shutil.which(f'conda{_exe}')):
            return exe
        return 'conda'  # cross our fingers

    def arguments(self) -> Tuple[str, ...]:
        prefix = self.prefix or self._default_prefix()
        args = [self.action, '-y', '--prefix', prefix]
        args.append('--override-channels')
        for channel in (*self.origins, *self._default_channels()):
            args.extend(["-c", channel])
        return (*args, *self.pkgs)

    def environment(
        self, env: QProcessEnvironment = None
    ) -> QProcessEnvironment:
        if env is None:
            env = QProcessEnvironment.systemEnvironment()
        PINNED = 'CONDA_PINNED_PACKAGES'
        system_pins = f"&{env.value(PINNED)}" if env.contains(PINNED) else ""
        env.insert(PINNED, f"napari={self._napari_pin()}{system_pins}")
        if 10 <= log.getEffectiveLevel() < 30:  # DEBUG level
            env.insert('CONDA_VERBOSITY', '3')
        if os.name == "nt":
            if not env.contains("TEMP"):
                temp = gettempdir()
                env.insert("TMP", temp)
                env.insert("TEMP", temp)
            if not env.contains("USERPROFILE"):
                env.insert("HOME", os.path.expanduser("~"))
                env.insert("USERPROFILE", os.path.expanduser("~"))
        return env

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

    def _default_channels(self):
        return ('conda-forge',)

    def _default_prefix(self):
        if (Path(sys.prefix) / "conda-meta").is_dir():
            return sys.prefix
        raise ValueError("prefix has not been specified!")


class InstallerQueue(QProcess):
    """Queue for installation and uninstallation tasks in the plugin manager."""

    # emitted when all jobs are finished
    # not to be confused with finished, which is emitted when each job is finished
    allFinished = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._queue: Deque[AbstractInstallerTask] = Deque()
        self._output_widget = None

        self.setProcessChannelMode(QProcess.MergedChannels)
        self.readyReadStandardOutput.connect(self._on_stdout_ready)
        self.readyReadStandardError.connect(self._on_stderr_ready)

        self.finished.connect(self._on_process_finished)
        self.errorOccurred.connect(self._on_error_occurred)

    # -------------------------- Public API ------------------------------
    def install(
        self,
        task_handler: TaskHandler,
        pkgs: Sequence[str],
        *,
        prefix: Optional[str] = None,
        origins: Sequence[str] = (),
        **kwargs,
    ) -> JobId:
        """Install packages in `pkgs` into `prefix` using `task_handler` tool with additional
        `origins` as source for `pkgs`.

        Parameters
        ----------
        task_handler : str
            Which type of installation task_handler to use: pip or conda
        pkgs : Sequence[str]
            List of packages to install.
        prefix : Optional[str], optional
            Optional prefix to install packages into.
        origins : Optional[Sequence[str]], optional
            Additional sources for packages to be downloaded from.

        Returns
        -------
        JobId : int
            ID that can be used to cancel the process.
        """
        item = self._build_queue_item(
            task_handler=task_handler,
            action="install",
            pkgs=pkgs,
            prefix=prefix,
            origins=origins,
            **kwargs,
        )
        return self._queue_item(item)

    def uninstall(
        self,
        task_handler: TaskHandler,
        pkgs: Sequence[str],
        *,
        prefix: Optional[str] = None,
        **kwargs,
    ) -> JobId:
        """Uninstall packages in `pkgs` from `prefix` using `task_handler` tool.

        Parameters
        ----------
        task_handler : str
            Which type of installation task_handler to use: pip or conda
        pkgs : Sequence[str]
            List of packages to uninstall.
        prefix : Optional[str], optional
            Optional prefix from which to uninstall packages.

        Returns
        -------
        JobId : int
            ID that can be used to cancel the process.
        """
        item = self._build_queue_item(
            task_handler=task_handler,
            action="uninstall",
            pkgs=pkgs,
            prefix=prefix,
            **kwargs,
        )
        return self._queue_item(item)

    def cancel(self, job_id: Optional[JobId] = None):
        """Cancel `job_id` if it is running.

        Parameters
        ----------
        job_id : Optional[JobId], optional
            Job ID to cancel.  If not provided, cancel all jobs.
        """

        def end_process():
            if os.name == 'nt':
                self.kill()  # this might be too agressive and won't allow rollbacks!
            else:
                self.terminate()
            if self._output_widget:
                self._output_widget.append("\nTask was cancelled by the user.")

        if job_id is None:
            # cancel all jobs
            self._queue.clear()
            end_process()
            return

        for i, item in enumerate(self._queue):
            if item.ident == job_id:
                if i == 0:  # first in queue, currently running
                    end_process()
                else:  # still pending, just remove from queue
                    self._queue.remove(item)
                return
        msg = f"No job with id {job_id}. Current queue:\n - "
        msg += "\n - ".join(
            [
                f"{item.ident} -> {item.executable()} {item.arguments()}"
                for item in self._queue
            ]
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

    def _build_queue_item(
        self,
        task_handler: TaskHandler,
        action: ActionType,
        pkgs: Sequence[str],
        prefix: Optional[str] = None,
        origins: Sequence[str] = (),
        **kwargs,
    ) -> AbstractInstallerTask:
        if task_handler == "pip":
            InstallerAction = PipInstallerTask
        elif task_handler == "conda":
            InstallerAction = CondaInstallerTask
        else:
            raise ValueError(f"Handler {task_handler} not recognized!")
        return InstallerAction(
            pkgs=pkgs, action=action, origins=origins, prefix=prefix, **kwargs
        )

    def _queue_item(self, item: AbstractInstallerTask) -> JobId:
        self._queue.append(item)
        self._process_queue()
        return item.ident

    def _process_queue(self):
        if not self._queue:
            self.allFinished.emit()
            return
        task = self._queue[0]
        self.setProgram(str(task.executable()))
        self.setProcessEnvironment(task.environment())
        self.setArguments([str(arg) for arg in task.arguments()])
        # this might throw a warning because the same process
        # was already running but it's ok
        self._log(f"Starting '{self.program()}' with args {self.arguments()}")
        self.start()

    def _on_process_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus
    ):
        try:
            current = self._queue[0]
        except IndexError:
            current = None
        if (
            current
            and current.action == "uninstall"
            and exit_status == QProcess.ExitStatus.NormalExit
            and exit_code == 0
        ):
            pm2 = PluginManager.instance()
            npe1_plugins = set(plugin_manager.iter_available())
            for pkg in current.pkgs:
                if pkg in pm2:
                    pm2.unregister(pkg)
                elif pkg in npe1_plugins:
                    plugin_manager.unregister(pkg)
                else:
                    log.warning(
                        'Cannot unregister %s, not a known napari plugin.', pkg
                    )
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


def _get_python_exe():
    # Note: is_bundled_app() returns False even if using a Briefcase bundle...
    # Workaround: see if sys.executable is set to something something napari on Mac
    if sys.executable.endswith("napari") and sys.platform == 'darwin':
        # sys.prefix should be <napari.app>/Contents/Resources/Support/Python/Resources
        if (python := Path(sys.prefix) / "bin" / "python3").is_file():
            return str(python)
    return sys.executable
