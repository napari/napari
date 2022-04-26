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
    """

    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._finished = False
        self._process = None
        self._processes = []
        self._worker_thread = None
        self._update_keys = [
            ('conda-forge/noarch', trans._('update - repodata')),
            ('Updating specs:', trans._('update - specs')),
            ('Total download:', trans._('update - downloading')),
            ('Preparing transaction:', trans._('update - files')),
            ('Verifying transaction:', trans._('update - certificates')),
            ('Executing transaction:', trans._('update - packages')),
            ('To activate this environment', trans._('update - shortcuts')),
        ]

    def _handle_yield(self, total):
        """Handle yielded progress for the dummy process."""
        i, msg = self._current_progress[-1]
        pbar = self._worker_thread.pbar
        pbar.set_description(msg)
        # FIXME: This is a hack to get the progress bar to update
        # ensure the bar is always larger and does not decrease due
        # to rounding
        pbar.total = (
            int(round(((total * len(self._update_keys)) / (i + 1)), 0)) + 1
        )

    def _dummy_process(self):
        """Dummy process to trigger the progress bars in the notification area."""
        total = 0
        while not self._finished:
            sleep(1)
            total += 1
            yield total

    def _envs_path(self):
        """Return the path to the conda environments."""
        # Check if running from base, and adjust accordingly
        envs_path = Path(sys.prefix).parent.parent
        if envs_path.exists():
            return envs_path

    def _create_process(self, conda=True):
        """Create a conda/mamba process.

        Parameters
        ----------
        conda : bool, optional
            If ``True`` use conda, otherwise use ``mamba``.
            Default is ``True``.

        Return
        ------
        process : QProcess or None
            Return the created conda/mambda process.
        """
        envs_path = self._envs_path()
        if conda:
            conda_exec = str(envs_path / "condabin" / "conda")
        else:
            conda_exec = str(envs_path / "condabin" / "mamba")

        process = QProcess()
        process.setProgram(conda_exec)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda process=process: self._on_stdout_ready(process)
        )
        process.finished.connect(
            lambda ec, es: self._on_process_finished(process, ec, es)
        )
        process.finished.connect(self._run_process)
        self._processes.append(process)
        return process

    def _on_stdout_ready(self, process):
        """Hanlde standard output from the process.

        Parameters
        ----------
        process : QProcess
            The process to handle.
        """
        text = process.readAllStandardOutput().data().decode()
        # print([text])

        # Handle common known messages
        for i, (key, value) in enumerate(self._update_keys):
            data = (i, value)
            if key in text and data not in self._current_progress:
                print(f'\n\nFound {key}\n\n')
                self._current_progress.append(data)
                break

    def _on_process_finished(self, process, exit_code, exit_status):
        """Handle process finish."""
        self._finished = True
        if exit_code == 0:
            msg = trans._("Version updated successfully!")
        else:
            msg = trans._("Version could not be updated!")

        notification_manager.receive_info(trans._(msg))
        self.finished.emit()

    def _run_process(self):
        """Run the process in the process queue."""
        if self._processes:
            self._process = self._processes.pop(0)
            self._process.start()

            # FIXME: Generalize to other commands
            self._current_progress = [(0, trans._('update'))]
            self._worker_thread = create_worker(
                self._dummy_process,
                _progress={
                    'total': len(self._update_keys),
                    'desc': trans._("update"),
                },
            )
            self._worker_thread.yielded.connect(self._handle_yield)
            self._worker_thread.start()

    def run_update(self, version, nightly=False):
        """"""
        env_path = self._envs_path() / "envs" / f"napari-{version}"
        if env_path.exists():
            old_path = str(env_path) + '-' + str(uuid.uuid4()) + "-broken"
            env_path.rename(old_path)
            # FIXME: Clean up at the end?
            # shutil.rmtree(old_path, ignore_errors=True)

        args = [
            "create",
            "-p",
            str(env_path),
            f"napari={version}=*pyside*",
            # f"napari-menu={version}",
            "--channel",
            "conda-forge",
            # "--shortcuts-only=napari-menu",
            "--yes",
        ]
        if nightly:
            args.extend(
                [
                    "--channel",
                    "napari",
                ]
            )

        process = self._create_process(conda=False)
        process.setArguments(args)
        self._run_process()

    def install(self, packages):
        """Install plugins in batch on the environment.

        Parameters
        ----------
        packages: List[str]
            Packages should be a list of tuples of the form.
        """
        env_path = self._envs_path() / "envs" / f"napari-{self._version}"
        args = [
            "install",
            "-p",
            str(env_path),
            "--channel",
            "conda-forge",
            "--yes",
        ] + packages
        process = self._create_process(conda=False)
        process.setArguments(args)
        self._run_process()

    def clean_cache(self):
        """Clean the cache."""
        args = ["clean", "--yes"]
        process = self._create_process(conda=True)
        process.setArguments(args)
        self._run_process()

    def stop(self):
        """Stop the running processes."""
        if self._worker_thread:
            self._worker_thread.quit()

        self._processes = []
        if self._process:
            self._process.kill()


_MANAGER = None


def get_update_manager(parent=None):
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = UpdateManager(parent)
    return _MANAGER
