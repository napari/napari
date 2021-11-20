(npe2-manifest-spec)=

# [npe2][] manifest specification

## TODO

Doc

- [ ] overall plugin/manifest versioning
- [ ] sample data hook
- [x] widget hook
- [ ] plugin life cycle. activation/deactivation
- [x] head matter
- [x] commands
- [x] reader
- [x] writer
- [x] theme
- [ ] menu/keybinding

Impl

- [ ] sample data hook
- [ ] configuration/settings
- [ ] Menu/submenu
- [ ] keybindings

Audit

- [ ] examples
  - [ ] exist
  - [ ] correct
- [ ] accuracy with spec.
  - [ ] are required fields all there
  - [ ] are optional things in the right place
  - [ ] fields removed from spec are not mentioned

## Introduction

The **manifest** is a specially formatted text file declaring the functionality of a [npe2][] plugin. A **plugin** is a python package that contains the manifest together with a suitable _[entry point group][epg]_ in the package metadata.

Manifest files may be [json][], [yaml][], or [toml][] files conforming to the manifest schema. The **schema** defines what to expect in a manifest by defining the fields and their data types. These fields and their meanings are described below.

A **plugin manager** is used to discover plugins, provide utilities for querying and manipulating plugins, and for exposing plugin-backed functionality to _napari_. **Discovery** is the process by that finds plugins, parses the manifests and indexes them for later use.

> describe lazy import and activation

## Configuring a python package to use a plugin manifest

### 1. Add package metadata for locating the manifest

The manifest file should be specified in the plugin's `setup.cfg` or `setup.py` file using the _[entry point group][epg]_: `napari.manifest`. For example, this would be the section for a plugin `npe2-tester` with `napari.yaml` as the manifest file:

```cfg
[options.entry_points]
napari.manifest =
    npe2-tester = npe2_tester:napari.yaml
```

The manifest file is specified relative to the submodule root path.
So for the example it will be loaded from: `<path/to/npe2-tester>/npe2_tester/napari.yaml`.

### 2. Include the manifest in the package distribution

The manifest file needs to be included as _[package data][pd]_ in distributable forms for the package. For example:

```toml
[metadata]
...
include_package_data=True

[options.package_data]
npe2_tester =
    napari.yaml
```

> What about MANIFEST.in

## Versioning

> XXX This section very rough

A set of conventions define how the plugin engine interacts with plugins. We want to continue to evolve these while providing a stable platform for plugin developers.

To that end, each set of conventions is assigned a **version identifier**.

> version id needs definition and a name. Using semver seems easiest. I kind of like `plugin=1.0` or something like that. We don't _need_ an identifier right now.

These conventions include:

- How the plugin is discovered
- How the plugin exposes functionality to the plugin engine
- Function signatures and calling conventions
- Manifest schema
- Behavior expected around plugin state

That covers a lot of ground!

### Identifying the plugin spec version

As of the introduction of `npe2`, there are two plugin systems in napari: `original` and `npe2`.

The original plugin system comprises `napari-plugin-engine` library and a `plugin_manager` contained within _napari_.

#### napari-plugin-engine

The original `napari-plugin-engine` describes one set of conventions for defining plugins. These plugins don't declare a version. These are implicitly identified. If a plugin is not first recognized as a newer version, but follows the `napari-plugin-engine` rules around discovery then it must be a `original` plugin.

#### npe2

There is only one version of npe2 at the moment. A plugin detected as an npe2 plugin will be assumed to have that version. When a future version of the plugin engine needs to be indicated, an identifer will be added to the manifest.

### Forwards compatibility

In the wild, there are a distribution of napari versions being run at any one time. The newest versions will have access to the latest plugin engine, but older versions will not. How will these old versions deal with plugins written for a future plugin engine?

A user running napari 0.4.11 uses the original `napari-plugin-engine` to interact with plugins. Newer `npe2`-style plugins won't be visible.

This convention is extended. A plugin engine supporting up to _plugin_spec_version=X_ will ignore any plugins declaring a version _Y_ when _X<Y_.

### Backwards compatibility

Wherever possible a plugin engine supporting _plugin_spec_version=X_ should support versions _Y≤X_.

It may be necessary to deprecate certain plugin types, over time.

### Migrating plugins

> TODO write and link to the [migration guide][mg].

> TODO cli tool for automating migration

## Manifest schema

When read, a manifest file is first parsed into an intermediate representation (a python `dict`) that is then validated and transformed into a [PluginManifest][]. These last steps are defined using [pydantic][]. For details refer to the `PluginManifest` api documentation.

The `PluginManifest` is structured hierarchically. These are broken into a set of top-level properties and several sections that are outlined below.

### Top-level properties

> **Chopping block**: categories, license, preview, private

> **non-doc**: categories

> **to add**: manifest/plugin-api version identifier

#### Required

- **publisher**: The name of the publisher. Example: `org.napari`. A _manifest key_ of the form `<publisher>.<name>` is used to index plugins with the `PluginManager`.
- **display_name**: User-facing text to display as the name of this plugin. Must be 3-40 characters long. Example: `napari SVG writer`.
- **entry_point**: The module containing the `activate` function. An `activate` function is not required but an `entry_point` module must be specified regardless. Example: `napari.plugins._builtins`.

##### Example

```yaml
name: napari_svg
display_name: napari SVG
license: BSD-3-Clause
entry_point: napari_svg
```

#### Optional

Python package metadata (`setup.py` or `setup.cfg`) may be used to populate missing optional fields. This only happens when loading the manifest from a python package.

- **name** The name of the plugin. Example: `napari_svg`. Should be a [PEP 508][]-compatible package name. If missing, this is populated from the python package [name][setup-name].
- **description**: User-facing text that describes what your extension is and does. If missing, this is populated from the python package [description][setup-desc].
- **version**: The current version of the plugin. If missing, this is populated from the python package [version][setup-version].
- **license**: The copyright license. Preferably a [SPDX][] compatible identifier. If missing, this is populated from the python package's [license][setup-lic] field. This may effect the visual appearance of a plugin within the application.
- **preview**: Indicates this plugin isn't quite ready for prime time. If missing, this is populated from the _development status_ [classifier][setup-classifier] in the python package metadata. This may effect the visual appearance of a plugin within the application.

  - more likely look at version number: If it ends with beta/alpha/rc, or is below 1.0.0 hide by default.

- **private**: Indicates this plugin should be exempted from plugin listings in the application. For example, perhaps napari builtins should not be enabled/disabled like other plugins and so they should be marked private.

### Contributions

The contributions section is a collection of entities declaring functionality.

The main entity here is `Commands`. A **command** is a python function associated with an _id_. A **command id** is used as a unique identifer for the command. This is how other contributions, like _readers_, _writers_ or _keyboard shortcuts_, reference a command.

Commands can statically specify their associated python function in the manifest, or dynamically during the plugin's `activate()` function.

Some commands are executed in specific contexts that require the callable function to conform to certain requirements. For example, a command that is reference by a _reader_ must conform to the `napari_get_reader` [hook-specification][get-reader-hook].

## Commands

Many plugin contributions rely on calling a python function. _Commands_ is a collection of these callable's and associated metadata.

In addition to being listed in this section, _commands_ may be dynamically registered by the plugin's _activate()_ function.

### Required fields

- **id** An identifer used to reference this command within this plugin.
- **title** Title by which the command is represented in the UI

### Optional fields

- **icon** Icon which is used to represent the command in the UI. Either a file path, an object with file paths for dark and light themes, or a theme icon references, like `$(zap)`
- **enablement** A predicate python expression evaluated during runtime to determine the presentation of related UI elements within different contexts.
- **python_name** Fully qualified name to callable python object implementing this command. This usually takes the form of `{obj.__module__}:{obj.__qualname__}` (e.g. `my_package.a_module:some_function`). If provided, using `register_command` in the plugin activate function is optional (but takes precedence).

## Readers

### Required fields

- **command** Identifier of the _command_ to execute.
- **filename_patterns** List of filename patterns (for fnmatch) that this reader can accept. Reader will be tried only if `fnmatch(filename, pattern) == True`.
- **accepts_directories** Whether this reader accepts directories

##### Example

```yaml
contributions:
  commands:
    - id: napari_builtins.get_reader
      python_name: napari.plugins._builtins:napari_get_reader
      title: Builtin Reader
  readers:
    - command: napari_builtins.get_reader
      accepts_directories: true
      filename_patterns: ["*.csv", "*.npy"]
```

### Calling convention

```python
def reader_function(path:str|List[str])->Optional[Callable[str,List[LayerData]]]:
    ...
```

###### Parameters

path(str or list of str)
: Path(s) to resources to read.

###### Returns

If the resource indicated by `path` is incompatible with the reader, `None` is returned. Otherwise, a function is returned that will return a collection of `LayerData`.

`LayerData` is a 1-, 2-, or 3-tuple of (data,), (data, meta), or (data, meta, layer_type).

###### Compatibility

The calling convention is compatible with the [`napari_get_reader`][get-reader-hook] hook.

## Writers

### Required fields

- **command** Identifier of the _command_ providing the writer.
- **layer_types** List of layer type constraints. These determine what combinations of layers this writer handles.
- **filename_extensions** List of filename extensions compatible with this writer. The first entry is used as the default if necessary.
- **save_dialog_title** Brief text used to describe this writer when presented in a save dialog. When not specifed the _command's_ title is used instead.

###### Example

Single-layer writer

```yaml
contributions:
  commands:
    - id: napari_builtins.write_points
      python_name: napari.plugins._builtins:napari_write_points
      title: napari built-in points writer
      short_title: napari points
  writers:
    - command: napari_builtins.write_points
      filename_extensions: [".csv"]
      layer_types: ["points"]
```

###### Example

Multi-layer writer

```yaml
contributions:
  commands:
    - id: napari_svg.svg_writer
      title: Write SVG
      python_name: napari_svg.hook_implementations:writer
  writers:
    - command: napari_svg.svg_writer
      layer_types: ["image*", "labels*", "points*", "shapes*", "vectors*"]
      filename_extensions: [".svg"]
```

### Layer type constraints

Given a set of layers, compatible writer plugins are selected based their _layer type constraints_.

A writer plugin can declare that it will write between _m_ and _n_ layers of a specific type where _0≤m≤n_.

For example:

```
    image      Write exactly 1 image layer.
    image?     Write 0 or 1 image layers.
    image+     Write 1 or more image layers.
    image*     Write 0 or more image layers.
    image{k}   Write exactly k image layers.
    image{m,n} Write between m and n layers (inclusive range). Must have m<=n.
```

When a type is not present in the list of constraints, that corresponds to a writer that is not compatible with that type. For example, a writer declaring:

```
    layer_types=["image+", "points*"]
```

would not be selected when trying to write an `image` and a `vector` layer because the above only works for cases with 0 `vector` layers.

Note that just because a writer declares compatibility with a layer type does not mean it actually writes that type. In the example above, the writer might accept a set of layers containing `image`s and `point`s, but the write command might just ignore the `point` layers. The writer must return `None` for unwritten layers.

### Calling convention

Currently, two calling conventions are supported for writers: single-layer and multi-layer writers. When at most one layer can be matched by a writer, it must use the single-layer convention. Otherwise, the multi-layer convention must be used.

###### Compatibility

The single-layer writer calling convention is compatible with the single-layer hooks like ['napari_write_image'][write-image-hook] and friends.

The multi-layer writer calling convention is _not_ compatible with [`napari_get_writer`][get-writer-hook] hooks, but it is compatible with the writers returned by those hooks.

#### multi-layer writer

```python
def writer_function(
    path: str, layer_data: List[LayerData]
) -> List[str]:
    ...
```

###### Parameters

path(str)
: Path to file, directory, or resource (like a url).

layer_data(list of LayerData)
: Each `LayerData` element is a tuple of `(data,meta,layer_type)`.

###### Return

Returns a list of paths that were successfully written.

#### single-layer writer

```python
def writer_function(
    path: str, data, meta
) -> Optional[str]:
    ...
```

###### Parameters

path(str)
: Path to file, directory, or resource (like a url).

data(array or list of array)
: Image data. Can be N dimensional. If meta[‘rgb’] is True then the data should be interpreted as RGB or RGBA. If meta[‘multiscale’] is True, then the data should be interpreted as a multiscale image.

meta(dict)
: Image metadata.

###### Return

If data is successfully written, return the path that was written. Otherwise, if nothing was done, return None.

## Themes

### Required fields

- **label** Label of the color theme as shown in the UI.
- **id** Id of the color theme as used in the user settings.
- **type** "dark" or "light"
- **colors**
  - **canvas**
  - **console**
  - **background**
  - **foreground**
  - **primary**
  - **secondary**
  - **highlight**
  - **text**
  - **icon**
  - **warning**
  - **current**

###### Example

```yaml
themes:
  - label: "Monokai"
    id: "monokai"
    type: "dark"
    colors:
      canvas: "#000000"
      console: "#000000"
      background: "#272822"
      foreground: "#75715e"
      primary: "#cfcfc2"
      secondary: "#f8f8f2"
      highlight: "#e6db74"
      text: "#a1ef34"
      warning: "#f92672"
      current: "#66d9ef"
```

## Widgets (Experimental)

Provides widgets to be docked in the viewer.

This plugin contribution is marked as experimental as the API or how the returned value is handled may change here more frequently then the rest of the codebase.

### Required fields

command
: Identifier of a command that returns the widget instance.

### Optional fields

name
: User facing name of the widget. If multiple widgets are provided by the same plugin, the name cannot be an empty string.

###### Example

Manifest

```yaml
contributions:
  command:
    - id: my_plugin.make_widget
      title: Willy's Wild Widget
  widgets:
    - command: my_plugin.make_widget
```

With a QtWidget:

```python
from qtpy.QtWidgets import QWidget

class MyWidget(QWidget):
     def __init__(self, napari_viewer):
         self.viewer = napari_viewer
         super().__init__()

         # initialize layout
         layout = QGridLayout()

         # add a button
         btn = QPushButton('Click me!', self)
         def trigger():
             print("napari has", len(napari_viewer.layers), "layers")
         btn.clicked.connect(trigger)
         layout.addWidget(btn)

         # activate layout
         self.setLayout(layout)

def make_widget():
     return MyWidget
```

With magicgui:

```python=
from magicgui import magic_factory

@magic_factory(auto_call=True, threshold={'max': 2 ** 16})
def threshold(
    data: 'napari.types.ImageData',
    threshold: int
) -> 'napari.types.LabelsData':
    return (data > threshold).astype(int)

def make_widget():
    return threshold
```

### Calling convention

```python
def widget_function()->FunctionGui | QWidget:
    ...
```

###### Return

A _callable_ that returns an instance of either a `QWidget` or a `FunctionGui`.

###### Compatibility

The calling convention for this command is compatible with [`napari_experimental_dock_widget`][dock-widget-hook].

# TODO

## Configuration

TODO

## Menus

- **command_pallete**
- **layers\_\_context**
- **plugins\_\_widgets**
- **test_menu**

### MenuItem

A list of items that one of either `MenuCommand` or `Submenu`:

- **MenuCommand**
  - **command** Identifier of the _command_ to execute.
  - **alt** Identifier of an alternative _command_ to execute. It will be shown and invoked when pressing Alt while opening a menu.
- **Submenu**
  - **submenu** Identifier of the submenu to display in this item. The submenu must be declared in `contributions.submenus`.

### Submenus

A list of items with the following properties:

- **id** identifier of the submenu
- **label** User-facing text shown as the menu label.
- **icon** (optional) Either a file-path, a theme icon reference (e.g. `$(zap)`), or an object with paths for dark and light themes.

## Keybindings

- **command** Identifier of the command to run when keybinding is triggered.
- **key** Key or key sequence (separate keys with plus-sign and sequences with space, e.g. Ctrl+O and Ctrl+L L for a chord.
- **mac** Mac specific key or key sequence.
- **linux** Linux specific key or key sequence.
- **win** Windows specific key or key sequence.
- **when** Condition when the key is active.

[epg]: https://packaging.python.org/specifications/entry-points/
[pd]: https://setuptools.pypa.io/en/latest/userguide/datafiles.html
[npe2]: https://github.com/tlambert03/npe2
[json]: https://www.json.org/
[yaml]: https://yaml.org/
[toml]: https://toml.io/
[pydantic]: https://pydantic-docs.helpmanual.io/
[pluginmanifest]: https://github.com/tlambert03/npe2/blob/main/npe2/manifest/schema.py
[pep 508]: https://www.python.org/dev/peps/pep-0423/
[setup-name]: https://packaging.python.org/guides/distributing-packages-using-setuptools/#name
[setup-version]: https://packaging.python.org/guides/distributing-packages-using-setuptools/#version
[setup-desc]: https://packaging.python.org/guides/distributing-packages-using-setuptools/#description
[setup-lic]: https://packaging.python.org/guides/distributing-packages-using-setuptools/#license
[setup-classifier]: https://packaging.python.org/guides/distributing-packages-using-setuptools/#classifiers
[spdx]: https://spdx.org/licenses/
[get-reader-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_get_reader
[get-writer-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_get_writer
[write-image-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_write_image
[dock-widget-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_experimental_provide_dock_widget
[mg]: https://hackmd.io/XltMlKUUT_KmOnZPx4RvAQ
