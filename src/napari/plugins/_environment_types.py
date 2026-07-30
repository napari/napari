"""Private backend-neutral types for managed plugin environments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from napari.plugins.environments import PluginTaskPhase

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class LocalPackageRecipe:
    path: Path


@dataclass(frozen=True)
class EnvironmentRecipe:
    plugin: str
    plugin_version: str
    environment_id: str
    python: str
    conda: tuple[str, ...]
    pypi: tuple[str, ...]
    channels: tuple[str, ...]
    local_packages: tuple[LocalPackageRecipe, ...]
    lockfile: bytes | None


@dataclass(frozen=True)
class WorkerCommand:
    plugin: str
    environment_id: str
    command_id: str
    target: str
    accepts_context: bool
    recipe: EnvironmentRecipe


@dataclass(frozen=True)
class BackendProgress:
    phase: PluginTaskPhase
    message: str
    current: int | None = None
    total: int | None = None


ProgressCallback = Callable[[BackendProgress], None]
CancelCallbackSetter = Callable[[Callable[[], Any]], None]


class BackendCanceled(RuntimeError):
    """A private backend operation was canceled."""


class BackendFailure(RuntimeError):
    """A normalized failure from a managed environment backend."""

    def __init__(
        self,
        message: str,
        *,
        details: str | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details
        self.diagnostics = diagnostics


class BackendUnavailable(BackendFailure):
    """The configured backend cannot be imported or initialized."""


class BackendPool(Protocol):
    def execute(
        self,
        target: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        accepts_context: bool,
        progress: ProgressCallback,
        set_cancel_callback: CancelCallbackSetter,
    ) -> Any: ...

    def close(self) -> None: ...


class EnvironmentBackend(Protocol):
    def fingerprint(self, recipe: EnvironmentRecipe) -> str: ...

    def prepare_environment(
        self,
        physical_name: str,
        recipe: EnvironmentRecipe,
        *,
        progress: ProgressCallback,
        set_cancel_callback: CancelCallbackSetter,
    ) -> Any: ...

    def start_pool(
        self,
        environment: Any,
        *,
        progress: ProgressCallback,
    ) -> BackendPool: ...

    def remove_environment(
        self,
        physical_name: str,
        *,
        progress: ProgressCallback | None = None,
        set_cancel_callback: CancelCallbackSetter | None = None,
    ) -> None: ...

    def environment_names(self) -> tuple[str, ...]: ...

    def close(self) -> None: ...
