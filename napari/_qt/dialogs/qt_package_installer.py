import contextlib
import os
import sys
from pathlib import Path
from tempfile import gettempdir
from typing import (
    Callable,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

from qtpy.QtCore import QObject, QProcess, QProcessEnvironment, Signal
from qtpy.QtWidgets import QTextEdit

from ...plugins import plugin_manager
from ...utils._appdirs import user_plugin_dir, user_site_packages
from ...utils.misc import running_as_bundled_app

InstallerTypes = Literal['pip', 'mamba']

# set_output_widget
# cancel
# install
# uninstall


# TODO: add error icon and handle pip install errors
class Installer(QObject):
    started = Signal()
    finished = Signal(int)
    readyReadStandardOutput = Signal()

    def __init__(
        self,
        output_widget: QTextEdit = None,
    ):
        super().__init__()
        self._queue: List[Tuple[Tuple[str, ...], Callable[[], QProcess]]] = []
        self._processes: Dict[Tuple[str, ...], QProcess] = {}
        self._exit_code = 0
        self._output_widget = output_widget
        self.process = None

    def _create_process(self):
        process = QProcess()
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda process=process: self._on_stdout_ready(process)
        )
        process.readyReadStandardOutput.connect(self.readyReadStandardOutput)

        env = QProcessEnvironment.systemEnvironment()
        process.setProcessEnvironment(env)

        self._set_program(process)
        self.set_output_widget(self._output_widget)
        process.finished.connect(self._on_process_finished)
        return process

    def _set_program(self, process: QProcess):
        raise NotImplementedError()

    def _sys_executable_or_bundled_python(self):
        # Note: is_bundled_app() returns False even if using a Briefcase bundle...
        # Workaround: see if sys.executable is set to something something napari on Mac
        if sys.executable.endswith("napari") and sys.platform == 'darwin':
            # sys.prefix should be <napari.app>/Contents/Resources/Support/Python/Resources
            python = os.path.join(sys.prefix, "bin", "python3")
            if os.path.isfile(python):
                return python
        return sys.executable

    def set_output_widget(self, output_widget: QTextEdit):
        if output_widget:
            self._output_widget = output_widget

    def _on_process_finished(
        self, exit_code: int, exit_status: QProcess.ExitStatus
    ):
        if exit_code != 0:
            self._exit_code = 0

        process_to_terminate = [
            pkg_list
            for pkg_list, proc in self._processes.items()
            if proc == process
        ]

        for pkg_list in process_to_terminate:
            proc = self._processes.pop(pkg_list)
            proc.terminate()

        self._handle_action()

    def _on_stdout_ready(self, process):
        if self._output_widget:
            text = process.readAllStandardOutput().data().decode()
            self._output_widget.append(text)

    def _handle_action(self):
        if self._queue:
            pkg_list, func = self._queue.pop()
            self.started.emit()
            process = func()
            self._processes[pkg_list] = process

        if not self._processes:

            plugin_manager.discover()
            plugin_manager.prune()
            self.finished.emit(self._exit_code)

    def install(self, pkg_list: Sequence[str]):
        self._queue_job('install', pkg_list)

    def uninstall(self, pkg_list: Sequence[str]):
        self._queue_job('uninstall', pkg_list)

    def _queue_job(
        self, action: Literal['install', 'uninstall'], pkg_list: Sequence[str]
    ):
        def _go():
            if action == 'install':
                self._execute(self._install_args(pkg_list))
            else:
                self._execute(self._uninstall_args(pkg_list))
                for pkg in pkg_list:
                    plugin_manager.unregister(pkg)

        self._queue.insert(0, (tuple(pkg_list), _go))
        self._handle_action()

    def _install_args(self, pkg_list: Sequence[str]):
        raise NotImplementedError()

    def _uninstall_args(self, pkg_list: Sequence[str]):
        raise NotImplementedError()

    def _execute(self, args):
        process = self._create_process()
        process.setArguments(args)
        if self._output_widget and self._queue:
            self._output_widget.clear()
        process.start()
        return process

    def cancel(self, pkg_list: Sequence[str] = None):
        if pkg_list is None:
            for _, process in self._processes.items():
                process.terminate()

            self._processes = {}
        else:
            with contextlib.suppress(KeyError):
                process = self._processes.pop(tuple(pkg_list))
                process.terminate()


class PipInstaller(Installer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_program(self, process: QProcess):
        process.setProgram(self._sys_executable_or_bundled_python())
        # patch process path
        env = process.systemEnvironment()
        combined_paths = os.pathsep.join(
            [user_site_packages(), env.systemEnvironment().value("PYTHONPATH")]
        )
        env.insert("PYTHONPATH", combined_paths)

    def _install_args(self, pkg_list: Sequence[str]):
        cmd = ['-m', 'pip', 'install', '--upgrade']

        if running_as_bundled_app() and sys.platform.startswith('linux'):
            cmd += ['--no-warn-script-location', '--prefix', user_plugin_dir()]
        return cmd + list(pkg_list)

    def _uninstall_args(
        self,
        pkg_list: Sequence[str],
    ):
        args = ['-m', 'pip', 'uninstall', '-y']
        return args + list(pkg_list)


class CondaInstaller(Installer):
    def __init__(self, use_mamba: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bin = 'mamba' if use_mamba else 'conda'

        if (Path(sys.prefix) / "conda-meta").is_dir():
            self._conda_env_path = sys.prefix
        else:
            self._conda_env_path = None

    def _set_program(self, process: QProcess):
        process.setProgram(self._bin)

        if self._bin == "mamba":
            from ..._version import version_tuple

            env = process.systemEnvironment()

            # To avoid napari version changing when installing a plugin, we
            # add a pin to the current napari version, that way we can
            # restrict any changes to the actual napari application.
            # Conda/mamba also pin python by default, so we effectively
            # constrain python and napari versions from changing, when
            # installing plugins inside the constructor bundled application.
            # See: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#preventing-packages-from-updating-pinning
            napari_version = ".".join(str(v) for v in version_tuple[:3])
            if env.contains("CONDA_PINNED_PACKAGES"):
                # str delimiter is '&'
                system_pins = f"&{env.value('CONDA_PINNED_PACKAGES')}"
            else:
                system_pins = ""
            env.insert(
                "CONDA_PINNED_PACKAGES",
                f"napari={napari_version}{system_pins}",
            )
            if os.name == "nt":
                # workaround https://github.com/napari/napari/issues/4247, 4484
                if not env.contains("TEMP"):
                    temp = gettempdir()
                    env.insert("TMP", temp)
                    env.insert("TEMP", temp)
                if not env.contains("USERPROFILE"):
                    env.insert("HOME", os.path.expanduser("~"))
                    env.insert("USERPROFILE", os.path.expanduser("~"))

    def _install_args(
        self,
        pkg_list: Sequence[str],
        channels: Sequence[str] = ("conda-forge",),
    ):
        return self._exec('install', channels, pkg_list)

    def _uninstall_args(
        self,
        pkg_list: Sequence[str],
        channels: Sequence[str] = ("conda-forge",),
    ):
        return self._exec('remove', channels, pkg_list)

    def _exec(self, cmd_name, channels, pkg_list):
        cmd = [cmd_name, '-y', '--prefix', self._conda_env_path]
        for channel in channels:
            cmd.extend(["-c", channel])
        return cmd + list(pkg_list)
