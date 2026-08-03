"""Application-owned lifecycle for managed plugin environments."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
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
    PluginEnvironmentInfo,
    PluginEnvironmentOperation,
    PluginEnvironmentProvisioningError,
    PluginEnvironmentState,
    PluginEnvironmentUnavailableError,
    PluginTask,
    PluginTaskMetadata,
    PluginTaskPhase,
    PluginTaskState,
    PluginWorkerError,
    PluginWorkerFailure,
    PluginWorkerState,
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


@dataclass(frozen=True)
class _EnvironmentDeclaration:
    recipe: EnvironmentRecipe
    display_name: str
    provision: str


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
            _local_package_recipe(
                base,
                str(package.path),
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


def _local_package_recipe(base: Path, value: str) -> LocalPackageRecipe:
    path = _safe_relative_path(base, value, 'Local package path')
    if not path.is_dir():
        raise ValueError(f'Local package path is not a directory: {path}')
    pyproject = path / 'pyproject.toml'
    if not pyproject.is_file():
        raise ValueError(f'Local package must contain pyproject.toml: {path}')
    try:
        document = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(
            f'Invalid local package metadata: {pyproject}'
        ) from error
    project = document.get('project')
    if not isinstance(project, dict) or not project.get('name'):
        raise ValueError(
            f'Local package must declare [project].name: {pyproject}'
        )
    dependencies = project.get('dependencies', ())
    dynamic = project.get('dynamic', ())
    if dependencies or (
        isinstance(dynamic, list) and 'dependencies' in dynamic
    ):
        raise ValueError(
            'Worker package runtime dependencies must be declared by the '
            f'environment manifest, not {pyproject}'
        )
    return LocalPackageRecipe(path)


def _environment_declaration(
    manifest: PluginManifest,
    environment: Any,
) -> _EnvironmentDeclaration:
    environment_id = str(environment.id)
    display_name = getattr(environment, 'display_name', None)
    if not display_name:
        display_name = environment_id.rsplit('.', 1)[-1]
    provision = getattr(environment, 'provision', 'on_demand')
    provision = getattr(provision, 'value', provision)
    return _EnvironmentDeclaration(
        recipe=_environment_recipe(manifest, environment),
        display_name=str(display_name),
        provision=str(provision),
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


def _iter_manifests(
    *,
    enabled_only: bool = False,
) -> tuple[PluginManifest, ...]:
    from npe2 import plugin_manager

    return tuple(
        plugin_manager.iter_manifests(disabled=False if enabled_only else None)
    )


def _owner_for_contribution(contribution_id: str) -> str | None:
    owners = [
        manifest.name
        for manifest in _iter_manifests()
        if contribution_id.startswith(f'{manifest.name}.')
    ]
    return max(owners, key=len, default=None)


def _find_worker_command(command_id: str) -> WorkerCommand:
    for manifest in _iter_manifests(enabled_only=True):
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
                    return _environment_declaration(
                        manifest, environment
                    ).recipe
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
        self._plugin_lifecycle_locks: dict[str, threading.Lock] = {}
        self._tasks: set[PluginTask[Any]] = set()
        self._task_plugins: dict[PluginTask[Any], str] = {}
        self._task_environments: dict[PluginTask[Any], str] = {}
        self._control_tasks: set[PluginTask[Any]] = set()
        self._preparing: dict[str, int] = {}
        self._stopping_plugins: dict[str, int] = {}
        self._stopping: dict[str, int] = {}
        self._failures: dict[str, str] = {}
        self._closed = False
        self._lock = threading.RLock()

    @property
    def _ownership_path(self) -> Path:
        return self.root / 'state' / 'ownership.json'

    def _read_ownership(self) -> dict[str, dict[str, str]]:
        path = self._ownership_path
        try:
            document = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return {}
        except (OSError, UnicodeError, json.JSONDecodeError):
            logger.exception('Could not read plugin environment ownership')
            return {}
        if (
            not isinstance(document, dict)
            or document.get('schema_version') != 1
            or not isinstance(document.get('environments'), dict)
        ):
            logger.error('Invalid plugin environment ownership document')
            return {}
        records: dict[str, dict[str, str]] = {}
        for environment_id, record in document['environments'].items():
            if not isinstance(environment_id, str) or not isinstance(
                record, dict
            ):
                continue
            required = {
                'plugin',
                'physical_name',
                'fingerprint',
                'plugin_version',
                'display_name',
                'provision',
            }
            if required <= record.keys() and all(
                isinstance(record[key], str) for key in required
            ):
                records[environment_id] = {
                    key: record[key] for key in required
                }
        return records

    def _write_ownership(self, records: dict[str, dict[str, str]]) -> None:
        path = self._ownership_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {'schema_version': 1, 'environments': records},
            sort_keys=True,
            indent=2,
        )
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix='.ownership-',
            suffix='.json',
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, 'w', encoding='utf-8') as stream:
                stream.write(payload)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _record_ownership(
        self,
        declaration: _EnvironmentDeclaration,
        *,
        physical_name: str,
        fingerprint: str,
    ) -> None:
        recipe = declaration.recipe
        with self._lock:
            records = self._read_ownership()
            records[recipe.environment_id] = {
                'plugin': recipe.plugin,
                'physical_name': physical_name,
                'fingerprint': fingerprint,
                'plugin_version': recipe.plugin_version,
                'display_name': declaration.display_name,
                'provision': declaration.provision,
            }
            self._write_ownership(records)

    def list_environments(
        self, plugin: str | None = None
    ) -> tuple[PluginEnvironmentInfo, ...]:
        declarations: dict[str, _EnvironmentDeclaration] = {}
        invalid: dict[str, tuple[str, str, str, str]] = {}
        for manifest in _iter_manifests():
            if plugin is not None and manifest.name != plugin:
                continue
            for environment in (
                getattr(manifest.contributions, 'environments', None) or ()
            ):
                environment_id = str(environment.id)
                try:
                    declaration = _environment_declaration(
                        manifest, environment
                    )
                except (OSError, ValueError) as error:
                    display_name = (
                        getattr(environment, 'display_name', None)
                        or environment_id.rsplit('.', 1)[-1]
                    )
                    provision = getattr(environment, 'provision', 'on_demand')
                    provision = getattr(provision, 'value', provision)
                    invalid[environment_id] = (
                        manifest.name,
                        str(display_name),
                        str(provision),
                        str(error),
                    )
                else:
                    declarations[environment_id] = declaration

        with self._lock:
            ownership = self._read_ownership()
            preparing = set(self._preparing)
            stopping_plugins = set(self._stopping_plugins)
            stopping = set(self._stopping)
            failures = dict(self._failures)
            pools = tuple(self._pool_entries.values())

        relevant_ownership = {
            environment_id: record
            for environment_id, record in ownership.items()
            if plugin is None or record['plugin'] == plugin
        }
        environment_ids = (
            set(declarations) | set(invalid) | set(relevant_ownership)
        )
        known_names: set[str] | None = set()
        inspection_failure: str | None = None
        backend: EnvironmentBackend | None = None
        if relevant_ownership:
            try:
                backend = self._get_backend()
                known_names = set(backend.environment_names())
            except Exception as error:
                known_names = None
                inspection_failure = str(error)
                logger.exception(
                    'Could not inspect persistent plugin environments'
                )

        infos: list[PluginEnvironmentInfo] = []
        for environment_id in sorted(environment_ids):
            declaration = declarations.get(environment_id)
            record = relevant_ownership.get(environment_id)
            invalid_declaration = invalid.get(environment_id)
            fingerprint_failure: str | None = None
            if declaration is not None:
                recipe = declaration.recipe
                plugin_name = recipe.plugin
                display_name = declaration.display_name
                provision = declaration.provision
                if record is None:
                    fingerprint = None
                else:
                    try:
                        assert backend is not None
                        fingerprint = backend.fingerprint(recipe)
                    except Exception as error:
                        fingerprint_failure = str(error)
                        logger.exception(
                            'Could not fingerprint plugin environment %s',
                            environment_id,
                        )
                        fingerprint = None
            elif invalid_declaration is not None:
                recipe = None
                (
                    plugin_name,
                    display_name,
                    provision,
                    _declaration_failure,
                ) = invalid_declaration
                fingerprint = None
            else:
                recipe = None
                assert record is not None
                plugin_name = record['plugin']
                display_name = (
                    record['display_name'] if record else environment_id
                )
                provision = record['provision'] if record else 'on_demand'
                fingerprint = None

            failure = failures.get(environment_id)
            if invalid_declaration is not None:
                state = PluginEnvironmentState.FAILED
                failure = invalid_declaration[3]
            elif environment_id in preparing:
                state = PluginEnvironmentState.PREPARING
            elif failure is not None:
                state = PluginEnvironmentState.FAILED
            elif record is not None and inspection_failure is not None:
                state = PluginEnvironmentState.FAILED
                failure = inspection_failure
            elif fingerprint_failure is not None:
                state = PluginEnvironmentState.FAILED
                failure = fingerprint_failure
            elif (
                record is None
                or known_names is None
                or record['physical_name'] not in known_names
            ):
                state = PluginEnvironmentState.MISSING
            elif (
                fingerprint is not None
                and record['fingerprint'] != fingerprint
            ):
                state = PluginEnvironmentState.STALE
            else:
                state = PluginEnvironmentState.READY

            environment_pools = [
                entry
                for entry in pools
                if entry.environment_id == environment_id and not entry.retired
            ]
            if plugin_name in stopping_plugins or environment_id in stopping:
                worker_state = PluginWorkerState.STOPPING
            elif environment_pools:
                worker_state = PluginWorkerState.RUNNING
            else:
                worker_state = PluginWorkerState.STOPPED
            infos.append(
                PluginEnvironmentInfo(
                    plugin=plugin_name,
                    environment_id=environment_id,
                    display_name=display_name,
                    provision=provision,
                    recipe_fingerprint=fingerprint,
                    state=state,
                    worker_state=worker_state,
                    failure=failure,
                )
            )
        return tuple(infos)

    def prepare(self, environment_id: str) -> PluginTask[None]:
        owner = _owner_for_contribution(environment_id)
        task: PluginTask[None] = PluginTask(
            PluginTaskMetadata(
                operation=PluginEnvironmentOperation.PREPARE,
                plugin=owner,
                environment_ids=(environment_id,),
            )
        )
        if owner:
            self._associate_task(task, owner, environment_id)
            if self._is_stopping(owner, environment_id):
                task.cancel()

        def run() -> None:
            recipe = _find_environment(environment_id)
            self._associate_task(task, recipe.plugin, recipe.environment_id)
            environment, physical_name = self._prepare_recipe(task, recipe)
            del environment
            try:
                declaration = self._find_declaration(environment_id)
            except KeyError:
                declaration = _EnvironmentDeclaration(
                    recipe=recipe,
                    display_name=environment_id.rsplit('.', 1)[-1],
                    provision='on_demand',
                )
            backend = self._get_backend()
            task._report_progress(
                PluginTaskPhase.CLEANING_UP,
                'Finalizing managed plugin environment',
            )
            self._record_ownership(
                declaration,
                physical_name=physical_name,
                fingerprint=backend.fingerprint(recipe),
            )
            self._finalize_preparation(
                task,
                backend,
                recipe,
                keep=physical_name,
            )
            task._set_result(None)

        self._submit(task, run)
        return task

    @staticmethod
    def _find_declaration(
        environment_id: str,
    ) -> _EnvironmentDeclaration:
        for manifest in _iter_manifests():
            for environment in (
                getattr(manifest.contributions, 'environments', None) or ()
            ):
                if str(environment.id) == environment_id:
                    return _environment_declaration(manifest, environment)
        raise KeyError(f'No managed plugin environment {environment_id!r}')

    def execute(
        self,
        command_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> PluginTask[Any]:
        command: WorkerCommand | None = None
        command_error: Exception | None = None
        try:
            command = _find_worker_command(command_id)
        except Exception as error:  # noqa: BLE001
            command_error = error
        owner = (
            command.plugin
            if command is not None
            else _owner_for_contribution(command_id)
        )
        task: PluginTask[Any] = PluginTask(
            PluginTaskMetadata(
                operation=PluginEnvironmentOperation.EXECUTE,
                plugin=owner,
                environment_ids=(
                    (command.environment_id,) if command is not None else ()
                ),
                command_id=command_id,
            )
        )
        if command is not None:
            self._associate_task(task, command.plugin, command.environment_id)
            if self._is_stopping(command.plugin, command.environment_id):
                task.cancel()
        elif owner:
            self._associate_task(task, owner)

        def run() -> None:
            if command_error is not None:
                raise command_error
            assert command is not None
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

    def stop_workers(
        self,
        plugin: str,
        environment_id: str | None = None,
    ) -> PluginTask[None]:
        task: PluginTask[None] = PluginTask(
            PluginTaskMetadata(
                operation=PluginEnvironmentOperation.STOP,
                plugin=plugin,
                environment_ids=(
                    (environment_id,) if environment_id is not None else ()
                ),
            )
        )
        self._associate_task(task, plugin)
        selected = None if environment_id is None else {environment_id}
        with self._lock:
            self._control_tasks.add(task)
        self._mark_stopping(plugin, selected)
        self._cancel_plugin_tasks(plugin, selected, owner_task=task)

        def run() -> None:
            try:
                with self._plugin_lifecycle_lock(plugin):
                    self._stop_workers(
                        task,
                        plugin,
                        selected,
                        owner_task=task,
                    )
                    task._set_result(None)
            finally:
                self._clear_stopping(plugin, selected)

        submitted = self._submit(
            task,
            run,
            initial_phase=PluginTaskPhase.CLEANING_UP,
            initial_message=f'Stopping managed workers for {plugin}',
        )
        if not submitted:
            self._clear_stopping(plugin, selected)
        return task

    def _cancel_plugin_tasks(
        self,
        plugin: str,
        environment_ids: set[str] | None,
        *,
        owner_task: PluginTask[Any],
    ) -> tuple[PluginTask[Any], ...]:
        with self._lock:
            tasks = tuple(
                candidate
                for candidate, task_plugin in self._task_plugins.items()
                if candidate is not owner_task
                and candidate not in self._control_tasks
                and task_plugin == plugin
                and (
                    environment_ids is None
                    or self._task_environments.get(candidate)
                    in {None, *environment_ids}
                )
            )
        for candidate in tasks:
            candidate.cancel()
        return tasks

    def _stop_workers(
        self,
        progress_task: PluginTask[Any],
        plugin: str,
        environment_ids: set[str] | None,
        *,
        owner_task: PluginTask[Any],
    ) -> None:
        tasks = self._cancel_plugin_tasks(
            plugin,
            environment_ids,
            owner_task=owner_task,
        )
        progress_task._report_progress(
            PluginTaskPhase.CLEANING_UP,
            f'Stopping managed workers for {plugin}',
        )
        for candidate in tasks:
            if candidate.state is not PluginTaskState.PENDING:
                with suppress(PluginEnvironmentError):
                    candidate.result()
        with self._lock:
            entries = [
                (name, entry)
                for name, entry in self._pool_entries.items()
                if entry.plugin == plugin
                and (
                    environment_ids is None
                    or entry.environment_id in environment_ids
                )
            ]
            active = [entry for _name, entry in entries if entry.active]
            if active:
                raise PluginEnvironmentError(
                    'Managed worker is still active',
                    plugin=plugin,
                    environment=active[0].environment_id,
                    phase=PluginTaskPhase.CLEANING_UP,
                )
            for name, _entry in entries:
                self._pool_entries.pop(name, None)
        errors: list[str] = []
        for _name, entry in entries:
            try:
                entry.pool.close()
            except Exception as error:  # noqa: BLE001
                errors.append(f'{entry.environment_id}: {error}')
        if errors:
            raise PluginEnvironmentError(
                'Could not stop every managed plugin worker',
                plugin=plugin,
                environment=(
                    next(iter(environment_ids))
                    if environment_ids and len(environment_ids) == 1
                    else None
                ),
                phase=PluginTaskPhase.CLEANING_UP,
                details='\n'.join(errors),
            )

    def remove_environments(
        self,
        plugin: str,
        environment_ids: tuple[str, ...] | None = None,
    ) -> PluginTask[None]:
        task: PluginTask[None] = PluginTask(
            PluginTaskMetadata(
                operation=PluginEnvironmentOperation.REMOVE,
                plugin=plugin,
                environment_ids=environment_ids or (),
            )
        )
        self._associate_task(task, plugin)
        selected = None if environment_ids is None else set(environment_ids)
        with self._lock:
            self._control_tasks.add(task)
        self._mark_stopping(plugin, selected)
        self._cancel_plugin_tasks(plugin, selected, owner_task=task)

        def run() -> None:
            try:
                with self._plugin_lifecycle_lock(plugin):
                    self._stop_workers(
                        task,
                        plugin,
                        selected,
                        owner_task=task,
                    )
                    with self._lock:
                        records = self._read_ownership()
                    owned = [
                        (environment_id, record)
                        for environment_id, record in records.items()
                        if record['plugin'] == plugin
                        and (selected is None or environment_id in selected)
                    ]
                    if not owned:
                        task._set_result(None)
                        return
                    backend = self._get_backend()
                    try:
                        known_names = set(backend.environment_names())
                    except BackendFailure as error:
                        raise PluginEnvironmentProvisioningError(
                            str(error),
                            plugin=plugin,
                            phase=PluginTaskPhase.CLEANING_UP,
                            details=error.details,
                        ) from error
                    total = len(owned)
                    for index, (environment_id, record) in enumerate(owned, 1):
                        if task.cancellation_requested:
                            raise BackendCanceled
                        task._report_progress(
                            PluginTaskPhase.CLEANING_UP,
                            f'Removing {environment_id}',
                            index - 1,
                            total,
                        )
                        prefix = (
                            record['physical_name'].rsplit('-', 1)[0] + '-'
                        )
                        targets = sorted(
                            name
                            for name in known_names
                            if name.startswith(prefix)
                        )
                        for physical_name in targets:
                            try:
                                backend.remove_environment(
                                    physical_name,
                                    progress=lambda update: self._report(
                                        task, update
                                    ),
                                    set_cancel_callback=(
                                        task._set_cancel_callback
                                    ),
                                )
                            except BackendFailure as error:
                                raise PluginEnvironmentProvisioningError(
                                    str(error),
                                    plugin=plugin,
                                    environment=environment_id,
                                    phase=PluginTaskPhase.CLEANING_UP,
                                    details=error.details,
                                ) from error
                            known_names.discard(physical_name)
                        with self._lock:
                            current = self._read_ownership()
                            current.pop(environment_id, None)
                            self._write_ownership(current)
                            self._failures.pop(environment_id, None)
                    task._set_result(None)
            finally:
                self._clear_stopping(plugin, selected)

        submitted = self._submit(
            task,
            run,
            initial_phase=PluginTaskPhase.CLEANING_UP,
            initial_message=f'Removing managed environments for {plugin}',
        )
        if not submitted:
            self._clear_stopping(plugin, selected)
        return task

    def _submit(
        self,
        task: PluginTask[Any],
        runner: Callable[[], None],
        *,
        initial_phase: PluginTaskPhase = PluginTaskPhase.PREPARING,
        initial_message: str = 'Preparing managed plugin environment',
    ) -> bool:
        def execute() -> None:
            task._set_running(
                initial_phase,
                initial_message,
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
                    self._task_environments.pop(task, None)
                    self._control_tasks.discard(task)

        with self._lock:
            if self._closed:
                self._task_plugins.pop(task, None)
                self._task_environments.pop(task, None)
                self._control_tasks.discard(task)
                task._set_error(
                    PluginEnvironmentUnavailableError(
                        'Plugin environment manager is closed'
                    )
                )
                return False
            self._tasks.add(task)
            try:
                self._executor.submit(execute)
            except RuntimeError as error:
                self._tasks.discard(task)
                self._task_plugins.pop(task, None)
                self._task_environments.pop(task, None)
                self._control_tasks.discard(task)
                task._set_error(
                    PluginEnvironmentUnavailableError(
                        'Plugin environment manager could not submit the task',
                        details=str(error),
                    )
                )
                return False
        return True

    def _associate_task(
        self,
        task: PluginTask[Any],
        plugin: str,
        environment_id: str | None = None,
    ) -> None:
        with self._lock:
            self._task_plugins[task] = plugin
            if environment_id is not None:
                self._task_environments[task] = environment_id

    def _plugin_lifecycle_lock(self, plugin: str) -> threading.Lock:
        with self._lock:
            return self._plugin_lifecycle_locks.setdefault(
                plugin, threading.Lock()
            )

    def _mark_stopping(
        self,
        plugin: str,
        environment_ids: set[str] | None,
    ) -> None:
        with self._lock:
            if environment_ids is None:
                self._stopping_plugins[plugin] = (
                    self._stopping_plugins.get(plugin, 0) + 1
                )
            else:
                for environment_id in environment_ids:
                    self._stopping[environment_id] = (
                        self._stopping.get(environment_id, 0) + 1
                    )

    def _clear_stopping(
        self,
        plugin: str,
        environment_ids: set[str] | None,
    ) -> None:
        with self._lock:
            if environment_ids is None:
                remaining = self._stopping_plugins[plugin] - 1
                if remaining:
                    self._stopping_plugins[plugin] = remaining
                else:
                    self._stopping_plugins.pop(plugin, None)
            else:
                for environment_id in environment_ids:
                    remaining = self._stopping[environment_id] - 1
                    if remaining:
                        self._stopping[environment_id] = remaining
                    else:
                        self._stopping.pop(environment_id, None)

    def _is_stopping(self, plugin: str, environment_id: str) -> bool:
        with self._lock:
            return (
                plugin in self._stopping_plugins
                or environment_id in self._stopping
            )

    def _raise_if_stopping(
        self,
        task: PluginTask[Any],
        recipe: EnvironmentRecipe,
    ) -> None:
        if task.cancellation_requested or self._is_stopping(
            recipe.plugin, recipe.environment_id
        ):
            raise BackendCanceled

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
        with self._lock:
            self._preparing[recipe.environment_id] = (
                self._preparing.get(recipe.environment_id, 0) + 1
            )
            self._failures.pop(recipe.environment_id, None)
        try:
            if backend is None or physical_name is None:
                backend, physical_name = self._environment_identity(
                    task, recipe
                )
            with self._lock:
                environment_lock = self._environment_locks.setdefault(
                    recipe.environment_id, threading.Lock()
                )
            with environment_lock:
                self._raise_if_stopping(task, recipe)
                environment = backend.prepare_environment(
                    physical_name,
                    recipe,
                    progress=lambda update: self._report(task, update),
                    set_cancel_callback=task._set_cancel_callback,
                )
        except BackendCanceled:
            raise
        except BackendUnavailable as error:
            failure = PluginEnvironmentUnavailableError(
                str(error),
                plugin=recipe.plugin,
                environment=recipe.environment_id,
                phase=task.phase,
                details=error.details,
            )
            with self._lock:
                self._failures[recipe.environment_id] = str(failure)
            raise failure from error
        except BackendFailure as error:
            failure = PluginEnvironmentProvisioningError(
                str(error),
                plugin=recipe.plugin,
                environment=recipe.environment_id,
                phase=task.phase,
                details=error.details,
            )
            with self._lock:
                self._failures[recipe.environment_id] = str(failure)
            raise failure from error
        except PluginEnvironmentError as error:
            with self._lock:
                self._failures[recipe.environment_id] = str(error)
            raise
        except Exception as error:
            with self._lock:
                self._failures[recipe.environment_id] = str(error)
            raise
        else:
            with self._lock:
                self._failures.pop(recipe.environment_id, None)
            return environment, physical_name
        finally:
            with self._lock:
                remaining = self._preparing[recipe.environment_id] - 1
                if remaining:
                    self._preparing[recipe.environment_id] = remaining
                else:
                    self._preparing.pop(recipe.environment_id, None)

    def _finalize_preparation(
        self,
        task: PluginTask[Any],
        backend: EnvironmentBackend,
        recipe: EnvironmentRecipe,
        *,
        keep: str,
    ) -> None:
        self._raise_if_stopping(task, recipe)
        self._retire_pool_generations(task, backend, recipe, keep=keep)
        self._remove_stale_generations(task, backend, recipe, keep=keep)
        self._raise_if_stopping(task, recipe)

    def _pool_for(
        self,
        task: PluginTask[Any],
        recipe: EnvironmentRecipe,
    ) -> _PoolEntry:
        self._raise_if_stopping(task, recipe)
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
        try:
            declaration = self._find_declaration(recipe.environment_id)
        except KeyError:
            declaration = _EnvironmentDeclaration(
                recipe=recipe,
                display_name=recipe.environment_id.rsplit('.', 1)[-1],
                provision='on_demand',
            )
        self._record_ownership(
            declaration,
            physical_name=physical_name,
            fingerprint=backend.fingerprint(recipe),
        )
        task._report_progress(
            PluginTaskPhase.CLEANING_UP,
            'Finalizing managed plugin environment',
        )
        self._finalize_preparation(
            task,
            backend,
            recipe,
            keep=physical_name,
        )
        self._raise_if_stopping(task, recipe)
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
            elif task.cancellation_requested or self._is_stopping(
                recipe.plugin, recipe.environment_id
            ):
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
        task: PluginTask[Any],
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
            self._close_retired_pool(backend, entry, task=task)

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
        *,
        task: PluginTask[Any] | None = None,
    ) -> None:
        try:
            entry.pool.close()
        except Exception:
            logger.exception(
                'Failed to close retired plugin environment pool %s',
                entry.physical_name,
            )
            return
        progress: Callable[[BackendProgress], None] | None = None
        cancel_callback: Callable[[Callable[[], Any]], None] | None = None
        if task is not None:
            if task.cancellation_requested:
                raise BackendCanceled
            task._report_progress(
                PluginTaskPhase.CLEANING_UP,
                f'Removing previous environment {entry.environment_id}',
            )

            def report_progress(update: BackendProgress) -> None:
                PluginEnvironmentManager._report(task, update)

            progress = report_progress
            cancel_callback = task._set_cancel_callback
        try:
            backend.remove_environment(
                entry.physical_name,
                progress=progress,
                set_cancel_callback=cancel_callback,
            )
        except BackendCanceled:
            raise
        except Exception:
            logger.exception(
                'Failed to remove retired plugin environment %s',
                entry.physical_name,
            )

    def _remove_stale_generations(
        self,
        task: PluginTask[Any],
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
                if task.cancellation_requested:
                    raise BackendCanceled
                task._report_progress(
                    PluginTaskPhase.CLEANING_UP,
                    f'Removing previous environment {recipe.environment_id}',
                )
                try:
                    backend.remove_environment(
                        name,
                        progress=lambda update: self._report(task, update),
                        set_cancel_callback=task._set_cancel_callback,
                    )
                except BackendCanceled:
                    raise
                except Exception:
                    logger.exception(
                        'Failed to remove stale plugin environment %s',
                        name,
                    )

    def release_plugin(self, plugin: str) -> None:
        """Stop worker pools owned by a disabled or unregistered plugin."""

        task: PluginTask[None] = PluginTask(
            PluginTaskMetadata(
                operation=PluginEnvironmentOperation.STOP,
                plugin=plugin,
            )
        )
        task._set_running(
            PluginTaskPhase.CLEANING_UP,
            f'Stopping managed workers for {plugin}',
        )
        self._mark_stopping(plugin, None)
        try:
            with self._plugin_lifecycle_lock(plugin):
                self._stop_workers(
                    task,
                    plugin,
                    None,
                    owner_task=task,
                )
                task._set_result(None)
        except PluginEnvironmentError:
            logger.exception(
                'Failed to stop workers for disabled plugin %s', plugin
            )
        finally:
            self._clear_stopping(plugin, None)

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
            self._task_environments.clear()
            self._control_tasks.clear()
            self._stopping.clear()
            self._stopping_plugins.clear()
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
