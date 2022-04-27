import sys
import uuid
from pathlib import Path
from time import sleep

from qtpy.QtCore import QDir, QFile, QObject, QProcess, Signal

from ..utils.notifications import notification_manager
from ..utils.translations import trans
from .qthreading import create_worker


class ManagerActions:
    install = "install"
    update = "update"
    clear = "clear"
    clean = "clean"
    remove = "remove"


class UpdateManager(QObject):
    """Manager for handling the update process.

    Parameters
    ----------
    parent : QObject, optional
        Parent of the manager. Default is None.
    """

    started = Signal()
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._finished = True
        self._process = None
        self._processes = []
        self._worker_thread = None
        self._messages = []
        # FIXME: add keys per action
        self._update_keys = [
            ('conda-forge/noarch', trans._('update - downloading')),
            ('Updating specs:', trans._('update - downloading')),
            ('Total download:', trans._('update - downloading')),
            ('Preparing transaction:', trans._('update - installing')),
            ('Verifying transaction:', trans._('update - installing')),
            ('Executing transaction:', trans._('update - installing')),
            ('To activate this environment', trans._('update - installing')),
        ]

    def _handle_yield(self, total):
        """Handle yielded progress for the dummy process."""
        i, msg = self._current_progress[-1]
        pbar = self._worker_thread.pbar
        pbar.set_description(msg)
        # This is a hack to get the progress bar to update when runnning a
        # QProcess
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

    def _base_path(self):
        """Return the path to the conda environments."""
        # FIXME: Check if running from base, and adjust accordingly
        base_path = Path(sys.prefix).parent.parent
        if base_path.exists() and (base_path / "condabin").exists():
            return base_path

    def _envs_path(self):
        """Return the path to the conda environments."""
        # FIXME: Check if running from base, and adjust accordingly
        envs_path = Path(sys.prefix).parent.parent / "envs"
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
        base_path = self._base_path()
        if conda:
            conda_exec = str(base_path / "condabin" / "conda")
        else:
            conda_exec = str(base_path / "condabin" / "mamba")

        process = QProcess()
        process.setProgram(conda_exec)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._on_stdout_ready)
        process.finished.connect(self._on_process_finished)
        process.finished.connect(self._run_process)
        self._processes.append(process)
        self._messages = []
        return process

    def _on_stdout_ready(self):
        """Handle standard output from the process.

        Parameters
        ----------
        process : QProcess
            The process to handle.
        """
        text = self._process.readAllStandardOutput().data().decode()
        self._messages.append(text)
        print(text)

        # Handle common known messages
        for i, (key, value) in enumerate(self._update_keys):
            data = (i, value)
            if key in text and data not in self._current_progress:
                self._current_progress.append(data)
                break

    def _on_process_finished(self, exit_code, exit_status):
        """Handle process finish."""
        print("\n".join(self._messages))
        if exit_code == 0:
            if self._process._action == ManagerActions.update:
                notification_manager.receive_info(
                    trans._("Version updated successfully!")
                )
                # Add file to identify a successful update
                napari_file = (
                    Path(self._process._env_path) / "conda-meta" / "napari"
                )
                with open(str(napari_file), "w") as fh:
                    fh.write("")
            elif self._process._action == ManagerActions.install:
                pass
            elif self._process._action == ManagerActions.clean:
                pass
            elif self._process._action == ManagerActions.clear:
                pass
        else:
            if self._process._action == ManagerActions.update:
                # FIXME: Show dialog
                trans._("Version could not be updated!")
            elif self._process._action == ManagerActions.install:
                pass
            elif self._process._action == ManagerActions.clean:
                pass
            elif self._process._action == ManagerActions.clear:
                pass

    def _run_process(self, total=None, desc=""):
        """Run the process in the process queue."""
        if self._processes:
            self._process = self._processes.pop(0)
            self._process.start()
            self._current_progress = [(0, desc)]

            _progress = {}
            if desc:
                _progress['desc'] = desc
            if total:
                _progress['total'] = total

            self._worker_thread = create_worker(
                self._dummy_process,
                _progress=_progress,
            )
            self._worker_thread.yielded.connect(self._handle_yield)
            self._worker_thread.start()
            self._finished = False
            self.started.emit()
        else:
            self._finished = True
            self.finished.emit()

    @staticmethod
    def remove_dirs(dirNames):
        """Remove a list of directories.

        Parameters
        ----------
        dirNames : list
            List of directories to remove.
        """
        results = []
        for path in dirNames:
            results.append(UpdateManager.removeDir(path))
        return results

    @staticmethod
    def remove_dir(dirName):
        """Remove a directory."""
        result = True
        qdir = QDir(dirName)
        if qdir.exists(dirName):
            for info in qdir.entryInfoList(
                QDir.NoDotAndDotDot
                | QDir.System
                | QDir.Hidden
                | QDir.AllDirs
                | QDir.Files,
                QDir.DirsFirst,
            ):
                if info.isDir():
                    result = UpdateManager.remove_dir(info.absoluteFilePath())
                else:
                    result = QFile.remove(info.absoluteFilePath())

                if not result:
                    return result
            result = dir.rmdir(dirName)
        return result

    def run_update(self, version, nightly=False):
        """"""
        env_path = self._envs_path() / f"napari-{version}"
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
            # FIXME: When napari menu is updated with the pins?
            # Also we require the conda fork
            # f"napari-menu={version}",
            "--channel",
            "conda-forge",
            # FIXME: When napari menu is updated with the pins?
            # Also we require the conda fork
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
        process._action = ManagerActions.update
        process._env_path = env_path
        self._run_process(total=len(self._update_keys), desc=trans._("update"))

    def install(self, version, packages):
        """Install plugins in batch on the environment.

        Parameters
        ----------
        packages: List[str]
            Packages should be a list of tuples of the form.
        """
        env_path = self._envs_path() / "envs" / f"napari-{version}"
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
        process._action = ManagerActions.install
        self._run_process(total=0, desc=trans._("install"))

    def remove(self, version):
        """Remove older napari versions.

        Parameters
        ----------
        version: str
            Version to remove.
        """
        env_path = self._envs_path() / "envs" / f"napari-{version}"
        args = [
            "remove",
            "-p",
            str(env_path),
            "--all",
            "--yes",
        ]
        process = self._create_process(conda=True)
        process.setArguments(args)
        process._action = ManagerActions.remove
        self._run_process(total=0, desc=trans._("remove"))

    def clean(self):
        """Clean the cache."""
        args = ["clean", "--all", "--yes"]
        process = self._create_process(conda=True)
        process.setArguments(args)
        process._action = ManagerActions.clean
        self._run_process(total=0, desc=trans._("clean"))

    def clear(self):
        """Clear previous broken installations."""
        for path in self._envs_path().iterdir():
            parts = path.name.split("-")
            if (
                path.is_dir()
                and parts[0] == "napari"
                and parts[-1] == "broken"
            ):
                print(f"removing {str(path)}")

    def is_finished(self) -> bool:
        """Return whether the process is finished."""
        return self._finished

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
