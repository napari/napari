"""Application-owned lifecycle for managed plugin environments."""

from __future__ import annotations

import atexit
import hashlib
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from napari.plugins._environment_types import (
    BackendCanceled,
    BackendFailure,
    BackendProgress,
    BackendUnavailable,
    EnvironmentBackend,
    EnvironmentRecipe,
    LocalPackageRecipe,
    WorkerCommand,
)
from napari.plugins.environments import (
    PluginEnvironmentError,
    PluginEnvironmentProvisioningError,
    PluginEnvironmentUnavailableError,
    PluginTask,
    PluginTaskPhase,
    PluginWorkerError,
    PluginWorkerFailure,
)
from napari.utils._platformdirs import user_data_dir

if TYPE_CHECKING:
    from collections.abc import Callable

    from npe2.manifest import PluginManifest

    from napari.plugins._environment_types import BackendPool

logger = logging.getLogger(__name__)


@dataclass
class _PoolEntry:
    plugin: str
    environment_id: str
    physical_name: str
    pool: BackendPool
    active: int = 0
    retired: bool = False


def _safe_relative_path(base: Path, value: str, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f'{label} must be relative to the plugin manifest')
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(
            f'{label} must remain inside the plugin package'
        ) from error
    return resolved


def _manifest_source_directory(manifest: PluginManifest) -> Path:
    source_file = getattr(manifest, '_source_file', None)
    if source_file is None:
        raise ValueError(
            f'Plugin {manifest.name!r} has no manifest source path; managed '
            'environment paths cannot be resolved'
        )
    return Path(source_file).resolve().parent


def _environment_recipe(
    manifest: PluginManifest,
    environment: Any,
) -> EnvironmentRecipe:
    lock_path = getattr(environment, 'lockfile', None)
    local_requirements = tuple(environment.local_packages)
    base = (
        _manifest_source_directory(manifest)
        if local_requirements or lock_path is not None
        else None
    )
    if not local_requirements:
        local_packages: tuple[LocalPackageRecipe, ...] = ()
    else:
        assert base is not None
        local_packages = tuple(
            LocalPackageRecipe(
                _safe_relative_path(
                    base, str(package.path), 'Local package path'
                ),
                bool(package.editable),
                tuple(package.extras),
            )
            for package in local_requirements
        )
    if lock_path is None:
        lockfile = None
    else:
        assert base is not None
        lockfile = _safe_relative_path(
            base, str(lock_path), 'Lockfile path'
        ).read_bytes()
    return EnvironmentRecipe(
        plugin=manifest.name,
        plugin_version=manifest.package_version or '0+unknown',
        environment_id=str(environment.id),
        python=str(environment.python),
        conda=tuple(environment.conda),
        pypi=tuple(environment.pypi),
        channels=tuple(environment.channels),
        local_packages=local_packages,
        lockfile=lockfile,
    )


def _worker_commands(manifest: PluginManifest) -> tuple[WorkerCommand, ...]:
    environments = {
        str(environment.id): environment
        for environment in (
            getattr(manifest.contributions, 'environments', None) or ()
        )
    }
    commands: list[WorkerCommand] = []
    for command in manifest.contributions.commands or ():
        environment_id = getattr(command, 'environment', None)
        if environment_id is None:
            continue
        environment_id = str(environment_id)
        environment = environments.get(environment_id)
        if environment is None:
            raise ValueError(
                f'Worker command {command.id!r} references unknown environment '
                f'{environment_id!r}'
            )
        if command.python_name is None:
            raise ValueError(
                f'Worker command {command.id!r} requires python_name'
            )
        commands.append(
            WorkerCommand(
                plugin=manifest.name,
                environment_id=environment_id,
                command_id=command.id,
                target=str(command.python_name),
                accepts_context=bool(
                    getattr(command, 'accepts_worker_context', False)
                ),
                recipe=_environment_recipe(manifest, environment),
            )
        )
    return tuple(commands)


def _iter_manifests() -> tuple[PluginManifest, ...]:
    from npe2 import plugin_manager

    return tuple(plugin_manager.iter_manifests(disabled=False))


def _owner_for_contribution(contribution_id: str) -> str | None:
    owners = [
        manifest.name
        for manifest in _iter_manifests()
        if contribution_id.startswith(f'{manifest.name}.')
    ]
    return max(owners, key=len, default=None)


def _find_worker_command(command_id: str) -> WorkerCommand:
    for manifest in _iter_manifests():
        try:
            commands = _worker_commands(manifest)
        except (OSError, ValueError) as error:
            raise PluginEnvironmentProvisioningError(
                f'Invalid managed environment declaration for {manifest.name}',
                plugin=manifest.name,
                command=command_id,
                phase=PluginTaskPhase.PREPARING,
                details=str(error),
            ) from error
        for command in commands:
            if command.command_id == command_id:
                return command
    raise KeyError(f'No managed plugin worker command {command_id!r}')


def _find_environment(environment_id: str) -> EnvironmentRecipe:
    for manifest in _iter_manifests():
        for environment in (
            getattr(manifest.contributions, 'environments', None) or ()
        ):
            if str(environment.id) == environment_id:
                try:
                    return _environment_recipe(manifest, environment)
                except (OSError, ValueError) as error:
                    raise PluginEnvironmentProvisioningError(
                        f'Invalid managed environment declaration for '
                        f'{manifest.name}',
                        plugin=manifest.name,
                        environment=environment_id,
                        phase=PluginTaskPhase.PREPARING,
                        details=str(error),
                    ) from error
    raise KeyError(f'No managed plugin environment {environment_id!r}')


class PluginEnvironmentManager:
    """Own managed environment state for one napari application."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        backend_factory: Callable[[Path], EnvironmentBackend] | None = None,
        max_parallel_tasks: int = 4,
    ) -> None:
        self.root = root or Path(user_data_dir()) / 'plugin-environments'
        self._backend_factory = backend_factory
        self._backend: EnvironmentBackend | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=max_parallel_tasks,
            thread_name_prefix='napari-plugin-environment',
        )
        self._pool_entries: dict[str, _PoolEntry] = {}
        self._environment_locks: dict[str, threading.Lock] = {}
        self._tasks: set[PluginTask[Any]] = set()
        self._task_plugins: dict[PluginTask[Any], str] = {}
        self._closed = False
        self._lock = threading.RLock()

    def prepare(self, environment_id: str) -> PluginTask[None]:
        task: PluginTask[None] = PluginTask()
        if owner := _owner_for_contribution(environment_id):
            self._associate_task(task, owner)

        def run() -> None:
            recipe = _find_environment(environment_id)
            self._associate_task(task, recipe.plugin)
            self._prepare_recipe(task, recipe)
            task._set_result(None)

        self._submit(task, run)
        return task

    def execute(
        self,
        command_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> PluginTask[Any]:
        task: PluginTask[Any] = PluginTask()
        if owner := _owner_for_contribution(command_id):
            self._associate_task(task, owner)

        def run() -> None:
            command = _find_worker_command(command_id)
            self._associate_task(task, command.plugin)
            entry = self._pool_for(task, command.recipe)
            try:
                if task.cancellation_requested:
                    raise BackendCanceled
                task._report_progress(
                    PluginTaskPhase.EXECUTING,
                    f'Executing {command_id}',
                )
                try:
                    result = entry.pool.execute(
                        command.target,
                        args,
                        kwargs,
                        accepts_context=command.accepts_context,
                        progress=lambda update: self._report(task, update),
                        set_cancel_callback=task._set_cancel_callback,
                    )
                except BackendFailure as error:
                    if task.cancellation_requested:
                        raise BackendCanceled from error
                    diagnostics = error.diagnostics
                    failure = (
                        PluginWorkerFailure(
                            category=diagnostics.get('category'),
                            message=str(
                                diagnostics.get('message', str(error))
                            ),
                            target=diagnostics.get('target'),
                            traceback=diagnostics.get('traceback'),
                            remote_exception_type=diagnostics.get(
                                'remote_exception_type'
                            ),
                            remote_exception_message=diagnostics.get(
                                'remote_exception_message'
                            ),
                            worker_environment=diagnostics.get(
                                'worker_environment'
                            ),
                            worker_pid=diagnostics.get('worker_pid'),
                            exit_code=diagnostics.get('exit_code'),
                            signal=diagnostics.get('signal'),
                            timeout=diagnostics.get('timeout'),
                            elapsed=diagnostics.get('elapsed'),
                            serialization_context=diagnostics.get(
                                'serialization_context'
                            ),
                        )
                        if diagnostics is not None
                        else None
                    )
                    raise PluginWorkerError(
                        str(error),
                        plugin=command.plugin,
                        environment=command.environment_id,
                        command=command.command_id,
                        phase=PluginTaskPhase.EXECUTING,
                        details=error.details,
                        failure=failure,
                    ) from error
            finally:
                self._release_pool(entry)
            task._set_result(result)

        self._submit(task, run)
        return task

    def _submit(
        self,
        task: PluginTask[Any],
        runner: Callable[[], None],
    ) -> None:
        def execute() -> None:
            task._set_running(
                PluginTaskPhase.PREPARING,
                'Preparing managed plugin environment',
            )
            try:
                runner()
            except BackendCanceled:
                task._set_canceled()
            except PluginEnvironmentError as error:
                task._set_error(error)
            except BackendUnavailable as error:
                task._set_error(
                    PluginEnvironmentUnavailableError(
                        str(error),
                        phase=task.phase,
                        details=error.details,
                    )
                )
            except BackendFailure as error:
                task._set_error(
                    PluginEnvironmentProvisioningError(
                        str(error),
                        phase=task.phase,
                        details=error.details,
                    )
                )
            except Exception as error:  # noqa: BLE001
                task._set_error(
                    PluginEnvironmentError(
                        str(error),
                        phase=task.phase,
                    )
                )
            finally:
                with self._lock:
                    self._tasks.discard(task)
                    self._task_plugins.pop(task, None)

        with self._lock:
            if self._closed:
                self._task_plugins.pop(task, None)
                task._set_error(
                    PluginEnvironmentUnavailableError(
                        'Plugin environment manager is closed'
                    )
                )
                return
            self._tasks.add(task)
            try:
                self._executor.submit(execute)
            except RuntimeError as error:
                self._tasks.discard(task)
                self._task_plugins.pop(task, None)
                task._set_error(
                    PluginEnvironmentUnavailableError(
                        'Plugin environment manager could not submit the task',
                        details=str(error),
                    )
                )

    def _associate_task(self, task: PluginTask[Any], plugin: str) -> None:
        with self._lock:
            self._task_plugins[task] = plugin

    def _get_backend(self) -> EnvironmentBackend:
        with self._lock:
            if self._backend is None:
                factory: Callable[[Path], EnvironmentBackend]
                if self._backend_factory is None:
                    from napari.plugins._wetlands import WetlandsBackend

                    factory = WetlandsBackend
                else:
                    factory = self._backend_factory
                self._backend = factory(self.root)
            return self._backend

    @staticmethod
    def _report(task: PluginTask[Any], update: BackendProgress) -> None:
        task._report_progress(
            update.phase,
            update.message,
            update.current,
            update.total,
        )

    @staticmethod
    def _physical_prefix(recipe: EnvironmentRecipe) -> str:
        identity = hashlib.sha256(
            f'{recipe.plugin}\0{recipe.environment_id}'.encode()
        ).hexdigest()[:12]
        slug = re.sub(
            r'[^A-Za-z0-9_.-]+',
            '-',
            f'{recipe.plugin}-{recipe.environment_id}',
        )
        return f'{slug[:64]}-{identity}'

    def _physical_name(
        self,
        backend: EnvironmentBackend,
        recipe: EnvironmentRecipe,
    ) -> str:
        return (
            f'{self._physical_prefix(recipe)}-'
            f'{backend.fingerprint(recipe)[:16]}'
        )

    def _environment_identity(
        self,
        task: PluginTask[Any],
        recipe: EnvironmentRecipe,
    ) -> tuple[EnvironmentBackend, str]:
        try:
            backend = self._get_backend()
        except BackendUnavailable as error:
            raise PluginEnvironmentUnavailableError(
                str(error),
                plugin=recipe.plugin,
                environment=recipe.environment_id,
                phase=task.phase,
                details=error.details,
            ) from error
        except BackendFailure as error:
            raise PluginEnvironmentProvisioningError(
                str(error),
                plugin=recipe.plugin,
                environment=recipe.environment_id,
                phase=task.phase,
                details=error.details,
            ) from error
        except Exception as error:
            raise PluginEnvironmentUnavailableError(
                'Could not initialize the managed environment backend',
                plugin=recipe.plugin,
                environment=recipe.environment_id,
                phase=task.phase,
                details=str(error),
            ) from error
        try:
            physical_name = self._physical_name(backend, recipe)
        except BackendUnavailable as error:
            raise PluginEnvironmentUnavailableError(
                str(error),
                plugin=recipe.plugin,
                environment=recipe.environment_id,
                phase=task.phase,
                details=error.details,
            ) from error
        except BackendFailure as error:
            raise PluginEnvironmentProvisioningError(
                str(error),
                plugin=recipe.plugin,
                environment=recipe.environment_id,
                phase=task.phase,
                details=error.details,
            ) from error
        except Exception as error:
            raise PluginEnvironmentProvisioningError(
                f'Invalid recipe for {recipe.environment_id}',
                plugin=recipe.plugin,
                environment=recipe.environment_id,
                phase=task.phase,
                details=str(error),
            ) from error
        return backend, physical_name

    def _prepare_recipe(
        self,
        task: PluginTask[Any],
        recipe: EnvironmentRecipe,
        *,
        backend: EnvironmentBackend | None = None,
        physical_name: str | None = None,
    ) -> tuple[Any, str]:
        if backend is None or physical_name is None:
            backend, physical_name = self._environment_identity(task, recipe)
        with self._lock:
            environment_lock = self._environment_locks.setdefault(
                recipe.environment_id, threading.Lock()
            )
        with environment_lock:
            if task.cancellation_requested:
                raise BackendCanceled
            try:
                environment = backend.prepare_environment(
                    physical_name,
                    recipe,
                    progress=lambda update: self._report(task, update),
                    set_cancel_callback=task._set_cancel_callback,
                )
            except BackendUnavailable as error:
                raise PluginEnvironmentUnavailableError(
                    str(error),
                    plugin=recipe.plugin,
                    environment=recipe.environment_id,
                    phase=task.phase,
                    details=error.details,
                ) from error
            except BackendFailure as error:
                raise PluginEnvironmentProvisioningError(
                    str(error),
                    plugin=recipe.plugin,
                    environment=recipe.environment_id,
                    phase=task.phase,
                    details=error.details,
                ) from error
            self._retire_pool_generations(
                backend,
                recipe,
                keep=physical_name,
            )
            self._remove_stale_generations(backend, recipe, keep=physical_name)
        return environment, physical_name

    def _pool_for(
        self,
        task: PluginTask[Any],
        recipe: EnvironmentRecipe,
    ) -> _PoolEntry:
        backend, physical_name = self._environment_identity(task, recipe)
        with self._lock:
            entry = self._pool_entries.get(physical_name)
            if entry is not None and not entry.retired:
                entry.active += 1
                return entry

        environment, physical_name = self._prepare_recipe(
            task,
            recipe,
            backend=backend,
            physical_name=physical_name,
        )
        if task.cancellation_requested:
            raise BackendCanceled
        with self._lock:
            entry = self._pool_entries.get(physical_name)
            if entry is not None and not entry.retired:
                entry.active += 1
                return entry
        try:
            pool = backend.start_pool(
                environment,
                progress=lambda update: self._report(task, update),
            )
        except BackendFailure as error:
            raise PluginEnvironmentProvisioningError(
                str(error),
                plugin=recipe.plugin,
                environment=recipe.environment_id,
                phase=PluginTaskPhase.STARTING,
                details=error.details,
            ) from error
        with self._lock:
            existing = self._pool_entries.get(physical_name)
            created: _PoolEntry | None = None
            if existing is not None and not existing.retired:
                existing.active += 1
                canceled = False
            elif task.cancellation_requested:
                canceled = True
            else:
                canceled = False
                created = _PoolEntry(
                    recipe.plugin,
                    recipe.environment_id,
                    physical_name,
                    pool,
                    active=1,
                )
                self._pool_entries[physical_name] = created
        if existing is not None and not existing.retired:
            pool.close()
            return existing
        if canceled:
            pool.close()
            raise BackendCanceled
        assert created is not None
        return created

    def _retire_pool_generations(
        self,
        backend: EnvironmentBackend,
        recipe: EnvironmentRecipe,
        *,
        keep: str,
    ) -> None:
        retired: list[_PoolEntry] = []
        with self._lock:
            for name, entry in tuple(self._pool_entries.items()):
                if (
                    entry.plugin == recipe.plugin
                    and entry.environment_id == recipe.environment_id
                    and name != keep
                ):
                    entry.retired = True
                    if entry.active == 0:
                        self._pool_entries.pop(name, None)
                        retired.append(entry)
        for entry in retired:
            self._close_retired_pool(backend, entry)

    def _release_pool(self, entry: _PoolEntry) -> None:
        backend: EnvironmentBackend | None = None
        close = False
        with self._lock:
            entry.active -= 1
            if entry.active < 0:
                raise RuntimeError('Managed plugin pool lease underflow')
            if (
                entry.retired
                and entry.active == 0
                and self._pool_entries.get(entry.physical_name) is entry
            ):
                self._pool_entries.pop(entry.physical_name, None)
                backend = self._backend
                close = True
        if close and backend is not None:
            self._close_retired_pool(backend, entry)

    @staticmethod
    def _close_retired_pool(
        backend: EnvironmentBackend,
        entry: _PoolEntry,
    ) -> None:
        try:
            entry.pool.close()
        except Exception:
            logger.exception(
                'Failed to close retired plugin environment pool %s',
                entry.physical_name,
            )
            return
        backend.remove_environment(entry.physical_name)

    def _remove_stale_generations(
        self,
        backend: EnvironmentBackend,
        recipe: EnvironmentRecipe,
        *,
        keep: str,
    ) -> None:
        prefix = f'{self._physical_prefix(recipe)}-'
        with self._lock:
            active = set(self._pool_entries)
        try:
            names = backend.environment_names()
        except Exception:
            logger.exception(
                'Failed to inspect stale managed environments for %s',
                recipe.environment_id,
            )
            return
        for name in names:
            if name.startswith(prefix) and name != keep and name not in active:
                backend.remove_environment(name)

    def release_plugin(self, plugin: str) -> None:
        """Stop worker pools owned by a disabled or unregistered plugin."""

        with self._lock:
            tasks = [
                task
                for task, task_plugin in self._task_plugins.items()
                if task_plugin == plugin
            ]
        for task in tasks:
            task.cancel()
        with self._lock:
            entries = [
                (name, entry)
                for name, entry in self._pool_entries.items()
                if entry.plugin == plugin
            ]
            for name, _entry in entries:
                self._pool_entries.pop(name, None)
        for _name, entry in entries:
            try:
                entry.pool.close()
            except Exception:
                logger.exception(
                    'Failed to close plugin environment pool %s',
                    entry.physical_name,
                )

    def close(self) -> None:
        """Cancel active work and release all owned runtime resources."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            pools = tuple(self._pool_entries.values())
            backend = self._backend
        errors: list[BaseException] = []
        for entry in pools:
            try:
                entry.pool.close()
            except Exception as error:  # noqa: BLE001
                errors.append(error)
        if backend is not None:
            try:
                backend.close()
            except Exception as error:  # noqa: BLE001
                errors.append(error)
        with self._lock:
            self._pool_entries.clear()
            self._tasks.clear()
            self._task_plugins.clear()
        if errors:
            raise PluginEnvironmentError(
                'Plugin environment shutdown did not complete cleanly',
                details='\n'.join(map(str, errors)),
            )


_manager: PluginEnvironmentManager | None = None
_manager_lock = threading.Lock()


def get_plugin_environment_manager() -> PluginEnvironmentManager:
    """Return the application-owned plugin environment manager."""

    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = PluginEnvironmentManager()
        return _manager


def _set_plugin_environment_manager(
    manager: PluginEnvironmentManager | None,
) -> None:
    global _manager
    with _manager_lock:
        _manager = manager


def shutdown_plugin_environments() -> None:
    """Close the application-owned plugin environment manager."""

    global _manager
    with _manager_lock:
        manager, _manager = _manager, None
    if manager is not None:
        manager.close()


def release_plugin_environments(plugin: str) -> None:
    """Release managed resources if the application manager exists."""

    with _manager_lock:
        manager = _manager
    if manager is not None:
        manager.release_plugin(plugin)


def _shutdown_plugin_environments_at_exit() -> None:
    try:
        shutdown_plugin_environments()
    except Exception:
        logger.exception('Failed to shut down managed plugin environments')


atexit.register(_shutdown_plugin_environments_at_exit)
