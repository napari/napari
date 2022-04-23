import sys
import uuid
from pathlib import Path
from time import sleep

from qtpy.QtCore import QObject, QProcess, Signal

from ..utils.notifications import notification_manager
from ..utils.translations import trans
from .qthreading import create_worker


class UpdateManager(QObject):
    """Manager for handling the update process.

    Parameters
    ----------
    parent : QObject, optional
        Parent of the manager. Default is None.
    version: str, optional
        Default is ``None``.
    update : boo., optional
        Default is ``False``.
    """

    finished = Signal()

    def __init__(self, parent=None, version=None, update=False):
        super().__init__(parent=parent)
        self._version = version
        self._finished = False
        self._process = None
        self._worker_thread = None
        self._current_progress = [(0, 'update')]
        self._update_keys = [
            ('conda-forge/noarch', trans._('update - repodata')),
            ('Updating specs:', trans._('update - specs')),
            ('Total download:', trans._('update - downloading')),
            ('Preparing transaction:', trans._('update - files')),
            ('Verifying transaction:', trans._('update - certificates')),
            ('Executing transaction:', trans._('update - packages')),
            ('To activate this environment', trans._('update - shortcuts')),
        ]

        self._worker_thread = create_worker(
            self._dummy_process,
            _progress={'total': len(self._update_keys), 'desc': "update"},
        )
        self._worker_thread.yielded.connect(self._handle_yield)

        if update:
            self._update_napari()

    def _handle_yield(self, total):
        i, msg = self._current_progress[-1]
        pbar = self._worker_thread.pbar
        pbar.set_description(msg)
        pbar.total = (
            int(round(((total * len(self._update_keys)) / (i + 1)), 0)) + 1
        )

    def _dummy_process(self):
        i = 0
        while not self._finished:
            sleep(1)
            i += 1
            yield i

    def _envs_path(self):
        # Check if running from base, and adjust accordingly
        envs_path = Path(sys.prefix).parent.parent
        if envs_path.exists():
            return envs_path

    def _update_napari(self):
        """"""
        self._worker_thread.start()
        envs_path = Path(sys.prefix).parent.parent
        env_path = envs_path / "envs" / f"napari-{self._version}"
        if env_path.exists():
            old_path = str(env_path) + '-' + str(uuid.uuid4()) + "-broken"
            env_path.rename(old_path)
            # FIXME: Clean up at the end?
            # shutil.rmtree(old_path, ignore_errors=True)

        args = [
            "create",
            "-p",
            str(env_path),
            f"napari={self._version}=*pyside*",
            # f"napari-menu={self._version}",
            "--channel",
            "conda-forge",
            "--channel",  # For testing nightly builds
            "napari",  # For testing nightly builds
            # "--shortcuts-only=napari-menu",
            # "--json",
            "--yes",
        ]
        print(' '.join(args))
        conda_exec = str(envs_path / "condabin" / "conda")
        conda_exec = str(envs_path / "condabin" / "mamba")
        print(conda_exec)
        self._process = QProcess()
        self._process.setProgram(conda_exec)
        self._process.setArguments(args)
        self._process.setProcessChannelMode(QProcess.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self._process.finished.connect(self._finished_update)
        self._process.start()

    def _on_stdout_ready(self):
        """"""
        text = self._process.readAllStandardOutput().data().decode()
        print([text])

        # Handle common known messages
        for i, (key, value) in enumerate(self._update_keys):
            data = (i, value)
            if key in text and data not in self._current_progress:
                print(f'\n\nFound {key}\n\n')
                self._current_progress.append(data)
                break

    def _finished_update(self):
        """"""
        self._finished = True
        if self._process.exitCode() == 0:
            msg = trans._("Version updated successfully!")
        else:
            msg = trans._("Version could not be updated!")

        notification_manager.receive_info(trans._(msg))
        self.finished.emit()

    def stop(self):
        """Stop the installation process."""
        if self._worker_thread:
            self._worker_thread.quit()

        if self._process:
            self._process.kill()
