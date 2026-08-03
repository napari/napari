"""Private Wetlands backend for managed plugin environments."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from packaging.version import InvalidVersion, Version

from napari.plugins._environment_types import (
    BackendCanceled,
    BackendFailure,
    BackendProgress,
    BackendUnavailable,
    EnvironmentRecipe,
)
from napari.plugins.environments import PluginTaskPhase

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from napari.plugins._environment_types import (
        CancelCallbackSetter,
        ProgressCallback,
    )

logger = logging.getLogger(__name__)


def _diagnostic_value(value: Any, name: str) -> Any:
    """Return a diagnostic data attribute without exposing API callables."""
    result = getattr(value, name, None)
    return None if callable(result) else result


def _failure_details(
    error: BaseException,
    *,
    worker_environment: str | None = None,
    worker_pid: int | None = None,
) -> str | None:
    failure = getattr(error, 'failure', None)
    if failure is None:
        return None
    lines: list[str] = []
    for name in (
        'category',
        'task_id',
        'call_target',
        'stage',
        'step_id',
        'command',
        'returncode',
        'environment',
        'worker',
        'exit_code',
        'signal',
        'timeout',
        'elapsed',
        'serialization_context',
        'cleanup_error',
    ):
        value = _diagnostic_value(failure, name)
        if value is not None:
            lines.append(f'{name}: {value}')
    worker = _diagnostic_value(failure, 'worker')
    failure_environment = (
        _diagnostic_value(worker, 'environment')
        if worker is not None
        else None
    )
    failure_pid = (
        _diagnostic_value(worker, 'pid') if worker is not None else None
    )
    if failure_environment is None:
        failure_environment = worker_environment
    if failure_pid is None:
        failure_pid = worker_pid
    if failure_environment is not None:
        lines.append(f'worker_environment: {failure_environment}')
    if failure_pid is not None:
        lines.append(f'worker_pid: {failure_pid}')
    stdout = getattr(failure, 'stdout_tail', ())
    stderr = getattr(failure, 'stderr_tail', ())
    traceback = getattr(failure, 'traceback', None)
    if stdout:
        lines.extend(('stdout:', *map(str, stdout)))
    if stderr:
        lines.extend(('stderr:', *map(str, stderr)))
    if traceback:
        lines.extend(('remote traceback:', str(traceback)))
    return '\n'.join(lines) or str(failure)


def _execution_diagnostics(
    error: BaseException,
    *,
    worker_environment: str | None = None,
    worker_pid: int | None = None,
) -> dict[str, Any] | None:
    failure = getattr(error, 'failure', None)
    if failure is None or not hasattr(failure, 'category'):
        return None
    category = getattr(failure.category, 'value', failure.category)
    remote_exception = getattr(failure, 'remote_exception', None)
    worker = _diagnostic_value(failure, 'worker')
    failure_environment = (
        _diagnostic_value(worker, 'environment')
        if worker is not None
        else None
    )
    failure_pid = (
        _diagnostic_value(worker, 'pid') if worker is not None else None
    )
    return {
        'category': None if category is None else str(category),
        'message': str(getattr(failure, 'message', error)),
        'target': getattr(failure, 'call_target', None),
        'traceback': getattr(failure, 'traceback', None),
        'remote_exception_type': getattr(
            remote_exception, 'qualified_name', None
        ),
        'remote_exception_message': getattr(remote_exception, 'message', None),
        'worker_environment': failure_environment or worker_environment,
        'worker_pid': (failure_pid if failure_pid is not None else worker_pid),
        'exit_code': getattr(failure, 'exit_code', None),
        'signal': getattr(failure, 'signal', None),
        'timeout': getattr(failure, 'timeout', None),
        'elapsed': getattr(failure, 'elapsed', None),
        'serialization_context': getattr(
            failure, 'serialization_context', None
        ),
    }


def _normalize_error(
    error: BaseException,
    *,
    worker_environment: str | None = None,
    worker_pid: int | None = None,
) -> BackendFailure:
    summary = getattr(getattr(error, 'failure', None), 'summary', None)
    message = str(summary()) if callable(summary) else str(error)
    return BackendFailure(
        message,
        details=_failure_details(
            error,
            worker_environment=worker_environment,
            worker_pid=worker_pid,
        ),
        diagnostics=_execution_diagnostics(
            error,
            worker_environment=worker_environment,
            worker_pid=worker_pid,
        ),
    )


class WetlandsPool:
    """Adapter around a Wetlands worker pool."""

    def __init__(
        self,
        pool: Any,
        operation_canceled: type[Exception],
        *,
        environment_name: str | None = None,
        running_workers: Callable[[str], tuple[Any, ...]] | None = None,
    ) -> None:
        self._pool = pool
        self._operation_canceled = operation_canceled
        self._environment_name = environment_name
        self._running_workers = running_workers

    def _worker_diagnostics(self) -> tuple[str | None, int | None]:
        environment = self._environment_name
        if environment is None or self._running_workers is None:
            return environment, None
        try:
            workers = self._running_workers(environment)
        except Exception:
            logger.debug(
                'Could not inspect Wetlands workers for %s',
                environment,
                exc_info=True,
            )
            return environment, None
        if len(workers) != 1:
            return environment, None
        return environment, _diagnostic_value(workers[0], 'process_id')

    def execute(
        self,
        target: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        accepts_context: bool,
        progress: ProgressCallback,
        set_cancel_callback: CancelCallbackSetter,
    ) -> Any:
        try:
            task = self._pool.submit_import(
                target,
                args=args,
                kwargs=kwargs,
                context_keyword=(
                    'napari_context' if accepts_context else None
                ),
            )
            set_cancel_callback(task.cancel)

            def receive(event: Any) -> None:
                if getattr(getattr(event, 'kind', None), 'value', None) in {
                    'started',
                    'update',
                }:
                    progress(
                        BackendProgress(
                            PluginTaskPhase.EXECUTING,
                            event.message,
                            event.current,
                            event.maximum,
                        )
                    )

            task.listen(receive)
            return task.wait_for()
        except Exception as error:
            if isinstance(error, self._operation_canceled):
                raise BackendCanceled from error
            environment, worker_pid = self._worker_diagnostics()
            raise _normalize_error(
                error,
                worker_environment=environment,
                worker_pid=worker_pid,
            ) from error

    def close(self) -> None:
        self._pool.close()


class WetlandsBackend:
    """Translate napari recipes and tasks to Wetlands 2."""

    def __init__(self, root: Path) -> None:
        try:
            import wetlands
        except ImportError as error:
            raise BackendUnavailable(
                'Wetlands 2.2 or newer is required for managed plugin '
                'environments',
                details='Install wetlands>=2.2,<3.',
            ) from error
        version = wetlands.__version__
        try:
            supported = Version('2.2') <= Version(version) < Version('3')
        except InvalidVersion:
            supported = False
        if not supported:
            raise BackendUnavailable(
                'Wetlands 2.2 or newer is required, but '
                f'Wetlands {version} is installed',
                details='Install wetlands>=2.2,<3.',
            )
        self._version = version
        self._environment_spec = wetlands.EnvironmentSpec
        self._local_package = wetlands.LocalPackage
        self._operation_canceled = wetlands.OperationCanceled
        self._manager = wetlands.EnvironmentManager(root=root)

    def _spec(self, recipe: EnvironmentRecipe) -> Any:
        return self._environment_spec(
            python=recipe.python,
            conda=recipe.conda,
            pypi=recipe.pypi,
            channels=recipe.channels,
            local=tuple(
                self._local_package(package.path)
                for package in recipe.local_packages
            ),
            pixi_lock=recipe.lockfile,
        )

    def fingerprint(self, recipe: EnvironmentRecipe) -> str:
        spec = self._spec(recipe)
        payload = {
            'plugin': recipe.plugin,
            'plugin_version': recipe.plugin_version,
            'environment': recipe.environment_id,
            'recipe_hash': spec.recipe_hash,
            'wetlands_version': self._version,
            'napari_recipe_abi': 1,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(',', ':')
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _listen_operation(
        operation: Any,
        phase: PluginTaskPhase,
        progress: ProgressCallback,
    ) -> None:
        def receive(event: Any) -> None:
            kind = getattr(getattr(event, 'kind', None), 'value', None)
            if kind == 'state' and event.message == 'Operation completed':
                return
            event_phase = phase
            if kind == 'cleanup':
                event_phase = PluginTaskPhase.CLEANING_UP
            stage = event.stage
            stage_value = getattr(stage, 'value', stage)
            stage_text = '' if stage_value is None else str(stage_value)
            message = event.message
            if stage_text and stage_text not in message:
                message = f'{stage_text}: {message}'
            progress(
                BackendProgress(
                    event_phase,
                    message,
                    event.current,
                    event.maximum,
                )
            )

        operation.listen(receive)

    def prepare_environment(
        self,
        physical_name: str,
        recipe: EnvironmentRecipe,
        *,
        progress: ProgressCallback,
        set_cancel_callback: CancelCallbackSetter,
    ) -> Any:
        try:
            preparation = self._manager.prepare()
            set_cancel_callback(preparation.cancel)
            self._listen_operation(
                preparation, PluginTaskPhase.PREPARING, progress
            )
            preparation.wait_for()

            provisioning = self._manager.provision(
                physical_name, self._spec(recipe)
            )
            set_cancel_callback(provisioning.cancel)
            self._listen_operation(
                provisioning, PluginTaskPhase.PROVISIONING, progress
            )
            return provisioning.wait_for()
        except Exception as error:
            if isinstance(error, self._operation_canceled):
                raise BackendCanceled from error
            raise _normalize_error(error) from error

    def start_pool(
        self,
        environment: Any,
        *,
        progress: ProgressCallback,
    ) -> WetlandsPool:
        progress(
            BackendProgress(
                PluginTaskPhase.STARTING,
                f'Starting worker for {environment.name}',
            )
        )
        try:
            return WetlandsPool(
                environment.start(workers=1),
                self._operation_canceled,
                environment_name=environment.name,
                running_workers=self._manager.running_workers,
            )
        except Exception as error:
            raise _normalize_error(error) from error

    def remove_environment(
        self,
        physical_name: str,
        *,
        progress: ProgressCallback | None = None,
        set_cancel_callback: CancelCallbackSetter | None = None,
    ) -> None:
        try:
            operation = self._manager.remove(physical_name)
            if set_cancel_callback is not None:
                set_cancel_callback(operation.cancel)
            if progress is not None:
                self._listen_operation(
                    operation,
                    PluginTaskPhase.CLEANING_UP,
                    progress,
                )
            operation.wait_for()
        except Exception as error:
            if isinstance(error, self._operation_canceled):
                raise BackendCanceled from error
            raise _normalize_error(error) from error

    def environment_names(self) -> tuple[str, ...]:
        return tuple(
            info.name for info in self._manager.managed_environments()
        )

    def close(self) -> None:
        try:
            self._manager.close()
        except Exception as error:
            raise _normalize_error(error) from error
