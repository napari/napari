(preferences)=

# Preferences

Starting with version 0.4.6, napari provides persistent settings.

Settings are managed by the global `SETTINGS` object and can be imported as:

```python
from napari.utils.settings import SETTINGS
```

## Sections

The settings are grouped by sections and napari core provides the following:

### APPEARANCE

User interface appearance settings.


#### Highlight thickness

*Select the highlight thickness when hovering over shapes/points.*

* <small>Access programmatically with `SETTINGS.appearance.highlight_thickness`.</small>

* <small>Type: `napari.utils.events.evented_model.ConstrainedIntValue`.</small>

* <small>Default: `1`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>
#### Show layer tooltips

*Toggle to display a tooltip on mouse hover.*

* <small>Access programmatically with `SETTINGS.appearance.layer_tooltip_visibility`.</small>

* <small>Type: `builtins.bool`.</small>

* <small>Default: `False`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>
#### Theme

*Select the user interface theme.*

* <small>Access programmatically with `SETTINGS.appearance.theme`.</small>

* <small>Type: `napari.utils.settings._defaults.Theme`.</small>

* <small>Default: `'dark'`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>

### APPLICATION

Main application settings.


#### Console notification level

*Select the notification level for the console.*

* <small>Access programmatically with `SETTINGS.application.console_notification_level`.</small>

* <small>Type: `napari.utils.notifications.NotificationSeverity`.</small>

* <small>Default: `<NotificationSeverity.NONE: 'none'>`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>
#### First time

*Indicate if napari is running for the first time. This setting is managed by the application.*

* <small>Access programmatically with `SETTINGS.application.first_time`.</small>

* <small>Type: `builtins.bool`.</small>

* <small>Default: `True`.</small>

#### GUI notification level

*Select the notification level for the user interface.*

* <small>Access programmatically with `SETTINGS.application.gui_notification_level`.</small>

* <small>Type: `napari.utils.notifications.NotificationSeverity`.</small>

* <small>Default: `<NotificationSeverity.INFO: 'info'>`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>
#### IPython interactive

*Toggle the use of interactive `%gui qt` event loop when creating napari Viewers in IPython.*

* <small>Access programmatically with `SETTINGS.application.ipy_interactive`.</small>

* <small>Type: `builtins.bool`.</small>

* <small>Default: `True`.</small>

#### Language

*Select the display language for the user interface.*

* <small>Access programmatically with `SETTINGS.application.language`.</small>

* <small>Type: `napari.utils.settings._defaults.Language`.</small>

* <small>Default: `'en'`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>
#### Opened folders history

*Last saved list of opened folders. This setting is managed by the application.*

* <small>Access programmatically with `SETTINGS.application.open_history`.</small>

* <small>Type: `builtins.str`.</small>

* <small>Default: `[]`.</small>

#### Preferences size

*Last saved width and height for the preferences dialog. This setting is managed by the application.*

* <small>Access programmatically with `SETTINGS.application.preferences_size`.</small>

* <small>Type: `typing.Tuple[int, int]`.</small>

* <small>Default: `None`.</small>

#### Saved folders history

*Last saved list of saved folders. This setting is managed by the application.*

* <small>Access programmatically with `SETTINGS.application.save_history`.</small>

* <small>Type: `builtins.str`.</small>

* <small>Default: `[]`.</small>

#### Save window geometry

*Toggle saving the main window size and position.*

* <small>Access programmatically with `SETTINGS.application.save_window_geometry`.</small>

* <small>Type: `builtins.bool`.</small>

* <small>Default: `True`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>
#### Save window state

*Toggle saving the main window state of widgets.*

* <small>Access programmatically with `SETTINGS.application.save_window_state`.</small>

* <small>Type: `builtins.bool`.</small>

* <small>Default: `True`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>
#### Window fullscreen

*Last saved fullscreen state for the main window. This setting is managed by the application.*

* <small>Access programmatically with `SETTINGS.application.window_fullscreen`.</small>

* <small>Type: `builtins.bool`.</small>

* <small>Default: `None`.</small>

#### Window maximized state

*Last saved maximized state for the main window. This setting is managed by the application.*

* <small>Access programmatically with `SETTINGS.application.window_maximized`.</small>

* <small>Type: `builtins.bool`.</small>

* <small>Default: `None`.</small>

#### Window position

*Last saved x and y coordinates for the main window. This setting is managed by the application.*

* <small>Access programmatically with `SETTINGS.application.window_position`.</small>

* <small>Type: `typing.Tuple[int, int]`.</small>

* <small>Default: `None`.</small>

#### Window size

*Last saved width and height for the main window. This setting is managed by the application.*

* <small>Access programmatically with `SETTINGS.application.window_size`.</small>

* <small>Type: `typing.Tuple[int, int]`.</small>

* <small>Default: `None`.</small>

#### Window state

*Last saved state of dockwidgets and toolbars for the main window. This setting is managed by the application.*

* <small>Access programmatically with `SETTINGS.application.window_state`.</small>

* <small>Type: `builtins.str`.</small>

* <small>Default: `None`.</small>

#### Show status bar

*Toggle diplaying the status bar for the main window.*

* <small>Access programmatically with `SETTINGS.application.window_statusbar`.</small>

* <small>Type: `builtins.bool`.</small>

* <small>Default: `True`.</small>


### PLUGINS

Plugins settings.


#### Plugin sort order

*Sort plugins for each action in the order to be called.*

* <small>Access programmatically with `SETTINGS.plugins.call_order`.</small>

* <small>Type: `typing.List[napari.utils.settings._defaults.PluginHookOption]`.</small>

* <small>Default: `None`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>
#### Disabled plugins

*Plugins to disable on application start.*

* <small>Access programmatically with `SETTINGS.plugins.disabled_plugins`.</small>

* <small>Type: `builtins.str`.</small>

* <small>Default: `set()`.</small>


### EXPERIMENTAL

Experimental settings.


#### Render Images Asynchronously

*Asynchronous loading of image data. 
This setting partially loads data while viewing. 
You must restart napari for changes of this setting to apply.*

* <small>Access programmatically with `SETTINGS.experimental.async_`.</small>

* <small>Type: `builtins.bool`.</small>

* <small>Default: `False`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>
#### Enable Asynchronous Tiling of Images

*Renders images asynchronously using tiles. 
You must restart napari for changes of this setting to apply.*

* <small>Access programmatically with `SETTINGS.experimental.octree`.</small>

* <small>Type: `typing.Union[bool, str]`.</small>

* <small>Default: `False`.</small>
* <small>UI: This setting can be configured via the preferences dialog.</small>

**Support for plugin specific settings will be provided in an upcoming release.**

## Changing settings programmatically

```python
from napari.utils.settings import SETTINGS

SETTINGS.appearance.theme = "light"
```

## Reset to defaults via CLI

To reset all napari settings to the default values:

```bash
napari --reset
```

## The preferences dialog

Starting with version 0.4.6, napari provides a preferences dialog to manage
some of the provided options.

### Appearance

![appearance](../images/_autogenerated/preferences-appearance.png)



### Application

![application](../images/_autogenerated/preferences-application.png)



### Plugins

![plugins](../images/_autogenerated/preferences-plugins.png)



### Experimental

![experimental](../images/_autogenerated/preferences-experimental.png)



### Reset to defaults via UI

To reset the preferences click on the `Restore defaults` button and continue
by clicking on `Restore`.

![](../images/_autogenerated/preferences-reset.png)
