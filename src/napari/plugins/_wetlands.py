"""Private Wetlands backend for managed plugin environments."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from napari.plugins._environment_types import (
    BackendCanceled,
    BackendFailure,
    BackendProgress,
    BackendUnavailable,
    EnvironmentRecipe,
)
from napari.plugins.environments import PluginTaskPhase

if TYPE_CHECKING:
    from pathlib import Path

    from napari.plugins._environment_types import (
        CancelCallbackSetter,
        ProgressCallback,
    )

logger = logging.getLogger(__name__)


def _failure_details(error: BaseException) -> str | None:
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
        value = getattr(failure, name, None)
        if value is not None:
            lines.append(f'{name}: {value}')
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


def _execution_diagnostics(error: BaseException) -> dict[str, Any] | None:
    failure = getattr(error, 'failure', None)
    if failure is None or not hasattr(failure, 'category'):
        return None
    category = getattr(failure.category, 'value', failure.category)
    remote_exception = getattr(failure, 'remote_exception', None)
    worker = getattr(failure, 'worker', None)
    return {
        'category': None if category is None else str(category),
        'message': str(getattr(failure, 'message', error)),
        'target': getattr(failure, 'call_target', None),
        'traceback': getattr(failure, 'traceback', None),
        'remote_exception_type': getattr(
            remote_exception, 'qualified_name', None
        ),
        'remote_exception_message': getattr(remote_exception, 'message', None),
        'worker_environment': getattr(worker, 'environment', None),
        'worker_pid': getattr(worker, 'pid', None),
        'exit_code': getattr(failure, 'exit_code', None),
        'signal': getattr(failure, 'signal', None),
        'timeout': getattr(failure, 'timeout', None),
        'elapsed': getattr(failure, 'elapsed', None),
        'serialization_context': getattr(
            failure, 'serialization_context', None
        ),
    }


def _normalize_error(error: BaseException) -> BackendFailure:
    summary = getattr(getattr(error, 'failure', None), 'summary', None)
    message = str(summary()) if callable(summary) else str(error)
    return BackendFailure(
        message,
        details=_failure_details(error),
        diagnostics=_execution_diagnostics(error),
    )


class WetlandsPool:
    """Adapter around a Wetlands worker pool."""

    def __init__(self, pool: Any, operation_canceled: type[Exception]) -> None:
        self._pool = pool
        self._operation_canceled = operation_canceled

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
            raise _normalize_error(error) from error

    def close(self) -> None:
        self._pool.close()


class WetlandsBackend:
    """Translate napari recipes and tasks to Wetlands 2."""

    def __init__(self, root: Path) -> None:
        try:
            import wetlands
        except ImportError as error:
            raise BackendUnavailable(
                'Wetlands 2 is required for managed plugin environments',
                details=(
                    'Install the released Wetlands 2 package, or install the '
                    'local Wetlands 2 checkout for development.'
                ),
            ) from error
        version = wetlands.__version__
        if version.partition('.')[0] != '2':
            raise BackendUnavailable(
                f'Wetlands 2 is required, but Wetlands {version} is installed',
                details='Install a Wetlands release in the >=2,<3 range.',
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
                self._local_package(
                    package.path,
                    editable=package.editable,
                    extras=package.extras,
                )
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
            event_phase = phase
            if (
                getattr(getattr(event, 'kind', None), 'value', None)
                == 'cleanup'
            ):
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
            )
        except Exception as error:
            raise _normalize_error(error) from error

    def remove_environment(self, physical_name: str) -> None:
        try:
            self._manager.remove(physical_name).wait_for()
        except Exception:
            # Stale environment cleanup is best-effort and must not turn a
            # successful new generation into a failure.
            logger.warning(
                'Failed to remove stale managed environment %s',
                physical_name,
                exc_info=True,
            )

    def environment_names(self) -> tuple[str, ...]:
        return tuple(
            info.name for info in self._manager.managed_environments()
        )

    def close(self) -> None:
        try:
            self._manager.close()
        except Exception as error:
            raise _normalize_error(error) from error
