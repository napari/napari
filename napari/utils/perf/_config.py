"""Perf configuration flags.
"""
import errno
import json
import os
from pathlib import Path
from typing import List, Optional

import wrapt

from ...utils.patcher import patch_callables
from ._utils import perf_timer

PERFMON_ENV_VAR = "NAPARI_PERFMON"


class PerfmonConfigError(Exception):
    """Error parsing or interpreting config file."""

    def __init__(self, message):
        self.message = message


def _patch_perf_timer(parent, callable: str, label: str) -> None:
    """Patches the callable to run it inside a perf_timer.

    Parameters
    ----------
    parent
        The module or class that contains the callable.
    callable : str
        The name of the callable (function or method).
    label : str
        The <function> or <class>.<method> we are patching.
    """

    @wrapt.patch_function_wrapper(parent, callable)
    def perf_time_callable(wrapped, instance, args, kwargs):
        with perf_timer(f"{label}"):
            return wrapped(*args, **kwargs)


class PerfmonConfig:
    """Reads the perfmon config file and sets up performance monitoring.

    Parameters
    ----------
    config_path : Path
        Path to the perfmon configuration file (JSON format).

    Config File Format
    ------------------
    {
        "trace_qt_events": true,
        "trace_callables": [
            "my_callables_1",
            "my_callables_2",
        ],
        "callable_lists": {
            "my_callables_1": [
                "module1.module2.Class1.method1",
                "module1.Class2.method2",
                "module2.module3.function1"
            ],
            "my_callables_2": [
                ...
            ]
        }
    }
    """

    def __init__(self, config_path: Optional[str]):
        self.config_path = config_path
        if config_path is None:
            return  # Legacy mode, trace Qt events only.

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(
                errno.ENOENT, f"Config file {PERFMON_ENV_VAR} not found", path,
            )

        with path.open() as infile:
            self.data = json.load(infile)

        self.patched = False

    def patch_callables(self):
        """Patch callables according to the config file.

        Call once at startup but after main() has started running. Do not
        call at module init or you will likely get circular dependencies.
        This function potentially imports a lot of your modules.
        """
        assert self.patched is False
        self._patch_callables()
        self.patched = True

    def _get_callables(self, callable_list) -> List[str]:
        """Get the list of callables from the config file.
        """
        try:
            return self.data["callable_lists"][callable_list]
        except KeyError:
            raise PerfmonConfigError(
                f"{self.config_path} has no callable list '{callable_list}'"
            )

    def _patch_callables(self):
        """Add a perf_timer to every callable.

        Notes
        -----
        data["trace_callables"] should contain the names of one or more
        lists of callables which are defined in data["callable_lists"].
        """
        for list_name in self.data["trace_callables"]:
            callable_list = self._get_callables(list_name)
            patch_callables(callable_list, _patch_perf_timer)

    @property
    def trace_qt_events(self) -> bool:
        """Return True if we should time Qt events.
        """
        if self.config_path is None:
            return True  # legacy mode
        try:
            return self.data["trace_qt_events"]
        except KeyError:
            return False


def _create_perf_config():
    value = os.getenv("NAPARI_PERFMON", "0")

    if value is None:
        # Totally disabled
        return None
    elif value == "1":
        # Legacy mode, no config file, trace Qt events only.
        return PerfmonConfig(None)
    else:
        # Normal mode, parse the config file.
        return PerfmonConfig(value)


# The global instance
perf_config = _create_perf_config()
