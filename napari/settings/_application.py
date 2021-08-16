from __future__ import annotations

from typing import List, Optional, Tuple, Union

from pydantic import Field, validator

from ..utils._base import _DEFAULT_LOCALE
from ..utils.events.custom_types import conint
from ..utils.events.evented_model import EventedModel
from ..utils.notifications import NotificationSeverity
from ..utils.translations import trans
from ._constants import LoopMode
from ._fields import Language, SchemaVersion

GridStride = conint(ge=-50, le=50, ne=0)
GridWidth = conint(ge=-1, ne=0)
GridHeight = conint(ge=-1, ne=0)


class ApplicationSettings(EventedModel):
    # 1. If you want to *change* the default value of a current option, you need to
    #    do a MINOR update in config version, e.g. from 3.0.0 to 3.1.0
    # 2. If you want to *remove* options that are no longer needed in the codebase,
    #    or if you want to *rename* options, then you need to do a MAJOR update in
    #    version, e.g. from 3.0.0 to 4.0.0
    # 3. You don't need to touch this value if you're just adding a new option
    schema_version: Union[SchemaVersion, Tuple[int, int, int]] = (0, 2, 1)
    first_time: bool = Field(
        True,
        title=trans._('First time'),
        description=trans._(
            'Indicate if napari is running for the first time. This setting is managed by the application.'
        ),
    )
    ipy_interactive: bool = Field(
        True,
        title=trans._('IPython interactive'),
        description=trans._(
            r'Toggle the use of interactive `%gui qt` event loop when creating napari Viewers in IPython.'
        ),
    )
    language: Language = Field(
        _DEFAULT_LOCALE,
        title=trans._("Language"),
        description=trans._(
            "Select the display language for the user interface."
        ),
    )
    # Window state, geometry and position
    save_window_geometry: bool = Field(
        True,
        title=trans._("Save window geometry"),
        description=trans._(
            "Toggle saving the main window size and position."
        ),
    )
    save_window_state: bool = Field(
        False,  # changed from True to False in schema v0.2.1
        title=trans._("Save window state"),
        description=trans._("Toggle saving the main window state of widgets."),
    )
    window_position: Optional[Tuple[int, int]] = Field(
        None,
        title=trans._("Window position"),
        description=trans._(
            "Last saved x and y coordinates for the main window. This setting is managed by the application."
        ),
    )
    window_size: Optional[Tuple[int, int]] = Field(
        None,
        title=trans._("Window size"),
        description=trans._(
            "Last saved width and height for the main window. This setting is managed by the application."
        ),
    )
    window_maximized: bool = Field(
        False,
        title=trans._("Window maximized state"),
        description=trans._(
            "Last saved maximized state for the main window. This setting is managed by the application."
        ),
    )
    window_fullscreen: bool = Field(
        False,
        title=trans._("Window fullscreen"),
        description=trans._(
            "Last saved fullscreen state for the main window. This setting is managed by the application."
        ),
    )
    window_state: Optional[str] = Field(
        None,
        title=trans._("Window state"),
        description=trans._(
            "Last saved state of dockwidgets and toolbars for the main window. This setting is managed by the application."
        ),
    )
    window_statusbar: bool = Field(
        True,
        title=trans._("Show status bar"),
        description=trans._(
            "Toggle diplaying the status bar for the main window."
        ),
    )
    preferences_size: Optional[Tuple[int, int]] = Field(
        None,
        title=trans._("Preferences size"),
        description=trans._(
            "Last saved width and height for the preferences dialog. This setting is managed by the application."
        ),
    )
    gui_notification_level: NotificationSeverity = Field(
        NotificationSeverity.INFO,
        title=trans._("GUI notification level"),
        description=trans._(
            "Select the notification level for the user interface."
        ),
    )
    console_notification_level: NotificationSeverity = Field(
        NotificationSeverity.NONE,
        title=trans._("Console notification level"),
        description=trans._("Select the notification level for the console."),
    )
    open_history: List[str] = Field(
        [],
        title=trans._("Opened folders history"),
        description=trans._(
            "Last saved list of opened folders. This setting is managed by the application."
        ),
    )
    save_history: List[str] = Field(
        [],
        title=trans._("Saved folders history"),
        description=trans._(
            "Last saved list of saved folders. This setting is managed by the application."
        ),
    )
    playback_fps: int = Field(
        10,
        title=trans._("Playback frames per second"),
        description=trans._("Playback speed in frames per second."),
    )
    playback_mode: LoopMode = Field(
        LoopMode.LOOP,
        title=trans._("Playback loop mode"),
        description=trans._("Loop mode for playback."),
    )

    grid_stride: GridStride = Field(
        default=1,
        title=trans._("Grid Stride"),
        description=trans._("Number of layers to place in each grid square."),
    )

    grid_width: GridWidth = Field(
        default=-1,
        title=trans._("Grid Width"),
        description=trans._("Number of columns in the grid."),
    )

    grid_height: GridHeight = Field(
        default=-1,
        title=trans._("Grid Height"),
        description=trans._("Number of rows in the grid."),
    )

    @validator('window_state')
    def _validate_qbtye(cls, v):
        if v and (not isinstance(v, str) or not v.startswith('!QBYTE_')):
            raise ValueError(
                trans._("QByte strings must start with '!QBYTE_'")
            )
        return v

    class Config:
        use_enum_values = False  # https://github.com/napari/napari/issues/3062

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = [
            "schema_version",
            "preferences_size",
            "first_time",
            "window_position",
            "window_size",
            "window_maximized",
            "window_fullscreen",
            "window_state",
            "window_statusbar",
            "open_history",
            "save_history",
            "ipy_interactive",
        ]
