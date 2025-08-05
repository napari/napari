import time
import warnings
from dataclasses import dataclass
from pathlib import Path

from napari.utils.translations import trans


@dataclass
class StartupScriptStatusInfo:
    startup_time: float
    script_path: Path
    script_code: str


startup_script_status_info: StartupScriptStatusInfo | None = None


def _maybe_run_startup_script():
    from napari.settings import get_settings

    if not get_settings().application.startup_script:
        return
    script_path = (
        Path(get_settings().application.startup_script).expanduser().resolve()
    )

    if not (script_path.exists() and script_path.is_file()):
        warnings.warn(
            trans._(
                'Startup script path is set to {script_path}. This path does not have a valid startup script. Please check the setting. napari will be launched without a startup script.',
                deferred=True,
                script_path=script_path,
            )
        )

    from napari_builtins.io._read import (
        execute_python_code,
    )

    script_code = script_path.read_text()
    start_time = time.time()

    execute_python_code(script_code, script_path)

    total_time = time.time() - start_time

    globals()['startup_script_status_info'] = StartupScriptStatusInfo(
        startup_time=total_time,
        script_path=script_path,
        script_code=script_code,
    )
