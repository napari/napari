from __future__ import annotations

import inspect
from pathlib import Path
from typing import Annotated, Any

from psutil import virtual_memory
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from napari.settings._constants import (
    BrushSizeOnMouseModifiers,
    LabelDTypes,
    LoopMode,
)
from napari.utils.camera_orientations import (
    DEFAULT_ORIENTATION_TYPED,
    DepthAxisOrientation,
    HorizontalAxisOrientation,
    VerticalAxisOrientation,
)
from napari.utils.events import Event
from napari.utils.events.custom_types import NotEqual
from napari.utils.events.evented_model import EventedModel
from napari.utils.notifications import NotificationSeverity


# we could use a smaller or greater 'le' for spacing,
# this is just meant to be a somewhat reasonable upper limit,
# as even on a 4k monitor a 2x2 grid will break calculation with >1300 spacing
MAX_GRID_SPACING = 1500

GridStride = Annotated[int, NotEqual(0)]
GridWidth = Annotated[int, Field(ge=-1), NotEqual(0)]
GridHeight = Annotated[int, Field(ge=-1), NotEqual(0)]
GridSpacing = Annotated[float, Field(ge=0, le=MAX_GRID_SPACING)]

_DEFAULT_MEM_FRACTION = 0.25
MAX_CACHE = virtual_memory().total * 0.5 / 1e9


class DaskSettings(EventedModel):
    enabled: bool = True
    cache: float = Field(
        virtual_memory().total * _DEFAULT_MEM_FRACTION / 1e9,
        ge=0,
        le=MAX_CACHE,
        title='Cache size (GB)',
    )


class ApplicationSettings(EventedModel):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # register callback to update brush size on mouse move modifiers
        self.events.brush_size_on_mouse_move_modifiers.connect(
            brush_size_on_mouse_move_modifiers_callback
        )
        self.events.brush_size_on_mouse_move_modifiers(
            value=self.brush_size_on_mouse_move_modifiers
        )

    first_time: bool = Field(
        True,
        title='First time',
        description='Indicate if napari is running for the first time. This setting is managed by the application.',
    )
    ipy_interactive: bool = Field(
        True,
        title='IPython interactive',
        description=r'Toggle the use of interactive `%gui qt` event loop when creating napari Viewers in IPython.',
    )
    # Window state, geometry and position
    save_window_geometry: bool = Field(
        True,
        title='Save window geometry',
        description='Toggle saving the main window size and position.',
    )
    save_window_state: bool = Field(
        False,  # changed from True to False in schema v0.2.1
        title='Save window state',
        description='Toggle saving the main window state of widgets.',
    )
    window_position: tuple[int, int] | None = Field(
        None,
        title='Window position',
        description='Last saved x and y coordinates for the main window. This setting is managed by the application.',
    )
    window_size: tuple[int, int] | None = Field(
        None,
        title='Window size',
        description='Last saved width and height for the main window. This setting is managed by the application.',
    )
    window_maximized: bool = Field(
        False,
        title='Window maximized state',
        description='Last saved maximized state for the main window. This setting is managed by the application.',
    )
    window_fullscreen: bool = Field(
        False,
        title='Window fullscreen',
        description='Last saved fullscreen state for the main window. This setting is managed by the application.',
    )
    window_state: str | None = Field(
        None,
        title='Window state',
        description='Last saved state of dockwidgets and toolbars for the main window. This setting is managed by the application.',
    )
    window_statusbar: bool = Field(
        True,
        title='Show status bar',
        description='Toggle diplaying the status bar for the main window.',
    )
    preferences_size: tuple[int, int] | None = Field(
        None,
        title='Preferences size',
        description='Last saved width and height for the preferences dialog. This setting is managed by the application.',
    )
    gui_notification_level: NotificationSeverity = Field(
        NotificationSeverity.INFO,
        title='GUI notification level',
        description='Select the notification level for the user interface.',
    )
    console_notification_level: NotificationSeverity = Field(
        NotificationSeverity.NONE,
        title='Console notification level',
        description='Select the notification level for the console.',
    )
    open_history: list[str] = Field(
        [],
        title='Opened folders history',
        description='Last saved list of opened folders. This setting is managed by the application.',
    )
    save_history: list[str] = Field(
        [],
        title='Saved folders history',
        description='Last saved list of saved folders. This setting is managed by the application.',
    )
    playback_fps: int = Field(
        10,
        title='Playback frames per second',
        description='Playback speed in frames per second.',
    )
    playback_mode: LoopMode = Field(
        LoopMode.LOOP,
        title='Playback loop mode',
        description='Loop mode for playback.',
    )

    depth_axis_orientation: DepthAxisOrientation = Field(
        default=DEFAULT_ORIENTATION_TYPED[0],
        title='Depth Axis Orientation',
        description=trans._(
            'Orientation of the depth axis in 3D view.\n'
            'Default is "Towards"; <0.6.0 was "Away".'
        ),
    )
    vertical_axis_orientation: VerticalAxisOrientation = Field(
        default=DEFAULT_ORIENTATION_TYPED[1],
        title='Vertical Axis Orientation',
        description=trans._(
            'Orientation of the vertical axis in 2D and 3D view.\n'
            'Default is "Down".'
        ),
    )
    horizontal_axis_orientation: HorizontalAxisOrientation = Field(
        default=DEFAULT_ORIENTATION_TYPED[2],
        title='Horizontal Axis Orientation',
        description=trans._(
            'Orientation of the horizontal axis in 2D and 3D view.\n'
            'Default is "Right".'
        ),
    )

    grid_stride: GridStride = Field(
        default=1,
        title='Grid Stride',
        description=trans._(
            'Number of layers to place in each grid viewbox before moving on to the next viewbox.\n'
            'A negative stride will cause the order in which the layers are placed in the grid to be reversed.\n'
            '0 is not a valid entry.'
        ),
    )

    grid_width: GridWidth = Field(
        default=-1,
        title='Grid Width',
        description='Number of columns in the grid.',
    )

    grid_height: GridHeight = Field(
        default=-1,
        title='Grid Height',
        description='Number of rows in the grid.',
    )

    grid_spacing: GridSpacing = Field(
        default=0,
        title='Grid Spacing',
        description=trans._(
            'The amount of spacing inbetween grid viewboxes.\n'
            'If between 0 and 1, it is interpreted as a proportion of the size of the viewboxes.\n'
            'If equal or greater than 1, it is interpreted as screen pixels.'
        ),
    )

    confirm_close_window: bool = Field(
        default=True,
        title='Confirm window or application closing',
        description='Ask for confirmation before closing a napari window or application (all napari windows).',
    )
    hold_button_delay: float = Field(
        default=0.5,
        title='Delay to treat button as hold in seconds',
        description='This affects certain actions where a short press and a long press have different behaviors, such as changing the mode of a layer permanently or only during the long press.',
    )

    brush_size_on_mouse_move_modifiers: BrushSizeOnMouseModifiers = Field(
        BrushSizeOnMouseModifiers.ALT,
        title='Brush size on mouse move modifiers',
        description='Modifiers to activate changing the brush size by moving the mouse.',
    )

    # convert cache (and max cache) from bytes to mb for widget
    dask: DaskSettings = Field(
        default=DaskSettings(),
        title='Dask cache',
        description='Settings for dask cache (does not work with distributed arrays)',
    )

    new_labels_dtype: LabelDTypes = Field(
        default=LabelDTypes.uint8,
        title='New labels data type',
        description='data type for labels layers created with the "new labels" button.',
    )

    plugin_widget_positions: dict[str, str] = Field(
        default={},
        title='Plugin widget positions',
        description='Per-widget last saved position of plugin dock widgets. This setting is managed by the application.',
    )

    startup_script: Path = Field(
        default=Path(),
        title='Full path to a startup script',
        description=trans._(
            'Path to a Python script that will be executed on napari startup.\n'
            'This can be used to customize the behavior of napari or load specific plugins automatically.',
        ),
        json_schema_extra={'file_extension': 'py'},
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name == 'startup_script' and value is not None:
            # Ensure the script is a valid Python file
            frame = inspect.currentframe()
            if frame is None:
                raise ValueError(
                    "The 'startup_script' setting can only be set by the napari application itself."
                )

            caller_frame = frame.f_back
            if caller_frame is None or not caller_frame.f_globals.get(
                '__name__', ''
            ).startswith('napari.'):
                raise ValueError(
                    "The 'startup_script' setting can only be set by the napari application itself."
                )
        super().__setattr__(name, value)

    @field_validator('window_state')
    @classmethod
    def _validate_qbtye(cls, v: str) -> str:
        if v and (not isinstance(v, str) or not v.startswith('!QBYTE_')):
            raise ValueError(
                "QByte strings must start with '!QBYTE_'"
            )
        return v

    model_config = SettingsConfigDict(use_enum_values=False)

    class NapariConfig:
        # Napari specific configuration
        preferences_exclude = (
            'schema_version',
            'preferences_size',
            'first_time',
            'window_position',
            'window_size',
            'window_maximized',
            'window_fullscreen',
            'window_state',
            'window_statusbar',
            'open_history',
            'save_history',
            'ipy_interactive',
            'plugin_widget_positions',
        )


def brush_size_on_mouse_move_modifiers_callback(event: Event) -> None:
    from napari.layers.labels._labels_mouse_bindings import (
        change_brush_size_on_mouse_move_modifiers,
    )

    change_brush_size_on_mouse_move_modifiers(event.value.split('+'))
