(npe2-manifest-spec)=

# npe2 manifest specification

The **plugin manifest** is a specially formatted text file declaring the
functionality of a [npe2][] plugin. A **plugin** is a python package that
contains the manifest together with a suitable _[entry point group][epg]_ in
the package metadata.

Manifest files may be [json][], [yaml][], or [toml][] files conforming to the
manifest schema. The **schema** defines what to expect in a manifest by
defining the fields and their data types. These fields and their meanings are
described below.

A **plugin engine** is used to discover plugins, provide utilities for
querying and manipulating plugins, and for exposing plugin-backed
functionality to _napari_. **Discovery** is the process by that finds plugins,
parses the manifests and indexes them for later use. The [npe2][] library
manages these responsibilities.

```{admonition} Backward compatibility
Plugins targeting `napari-plugin-engine` will continue to work, but we
recommend migrating to `npe2` as soon as possible. `npe2` includes tooling to
help automate the process of migrating plugins. See the [migration
guide](npe2-migration-guide) for details.
```

## Configuring a python package to use a plugin manifest

### 1. Add package metadata for locating the manifest

The manifest file should be specified in the plugin's `setup.cfg` or
`setup.py` file using the _[entry point group][epg]_: `napari.manifest`. For
example, this would be the section for a plugin `npe2-tester` with
`napari.yaml` as the manifest file:

```cfg
[options.entry_points]
napari.manifest =
    npe2-tester = npe2_tester:napari.yaml
```

The manifest file is specified relative to the submodule root path.
So for the example it will be loaded from: `<path/to/npe2-tester>/napari.yaml`.

### 2. Include the manifest in the package distribution

The manifest file needs to be included as _[package data][pd]_ in
distributable forms for the package. For example:

```toml
[metadata]
...
include_package_data=True

[options.package_data]
npe2_tester =
    napari.yaml
```

## Manifest schema

The plugin manifest file is a [json][], [yaml][], or [toml][] conforming to
a schema. The `npe2` package includes a command-line tool that can be used
to validate a schema:

```
pip install npe2
npe2 validate my_plugin_manifest.yml
```

```{note}
Internally, the manifest is represented by a [pydantic][] model, the
[PluginManifest][].
```

The manifest file's fields are described in detail below. The manifest file's
overall structure can be viewed as a hierarchy:

```
<top-level properties>
contributions:
  commands:     <list of commands>
  readers:      <list of readers>
  writers:      <list of writers>
  sample_data:  <list of sample data providers>
  widgets:      <list of widget providers>
  themes:       <list of themes>
```

Readers, writers, sample data providers and widget providers refer to callable
python functions that a plugin defines. Each callable is identified with an
entry in the list of `commands` via a unique id. For more see [Commands] below.

```{note}
Python package metadata (`setup.py` or `setup.cfg`) may be used to populate
selected properties of the [PluginManifest][].
```

## Top-level properties

### Required

- **name** The name of the plugin. Example: `napari_svg`. Should be a
  [PEP-8][]-compatible package name. If missing, this is populated from the
  python package [name][setup-name].

### Optional

- **display_name**: User-facing text to display as the name of this plugin.
  Example: `napari SVG`. Must be 3-40 characters long, containing
  printable word characters, and must not begin or end with an underscore,
  white space, or non-word character.
- **entry_point**: The module containing the `activate()` function. Example:
  `foo.bar.baz`. This should be a fully qualified module string.

## Commands

**command** is a python function associated with an id. The **id** is
a unique identifier to which other contributions, like readers, can refer.

### Required fields

- **id** An identifer used to reference this command within this plugin.
- **title** A description of the command. This might be used, for example,
  when searching in a command pallette. Example: "Generate lily sample",
  or "Read tiff image".

### Optional fields

- **icon** Icon which is used to represent the command in the UI. Either a file path, an object with file paths for dark and light themes, or a theme icon references, like `$(zap)`
- **enablement** A predicate python expression evaluated during runtime to determine the presentation of related UI elements within different contexts.
- **python_name** Fully qualified name to callable python object implementing
  this command. This usually takes the form of
  `{obj.__module__}:{obj.__qualname__}` (e.g.
  `my_package.a_module:some_function`). If provided, using `register_command`
  in the plugin activate function is optional (but takes precedence).

### Example

```yaml
name: my-plugin
contribution:
  commands:
    - id: my-plugin.publish_paper
      title: Do experiments, analysis, write paper, and submit
      python_name: my_plugin.publish_func
```

## Readers

### Required fields

- **command** Identifier of the _command_ to execute.

### Optional fields

- **filename_patterns** List of filename patterns (for fnmatch) that this
  reader can accept. Reader will be tried only if `fnmatch(filename, pattern) == True`.
  Empty list by default.
- **accepts_directories** If true, the reader will accept paths to directories
  for reading data.

### Example

```yaml
name: my-reader
contribution:
  commands:
    - id: my-reader.read_npy_image
      title: Open an npy file as an image
      python_name: my_reader.load_as_image
  readers:
    - command: my-reader.read_npy_image
      accepts_directories: false
      filename_patterns: ["*.npy"]
```

### Calling convention

```python
def get_reader(path:str)->Optional[Callable[[str],List[LayerData]]]:
  ...
  return reader

def reader(path: str) -> List[LayerData]:
    ...
```

```{note}
The reader command is compatible with functions used for the `napari_get_reader`
[hook specification][get-reader-hook].
```

**Parameters**

path(str)
: Path to file, directory, or resource (like a url).

**Return**

layer_data(list of LayerData)
: Each `LayerData` element is a tuple of `(data,meta,layer_type)`.

## Writers

### Required fields

- **command** Identifier of the _command_ providing the writer.
- **layer_types** List of layer type constraints. These determine what combinations of layers this writer handles.

### Optional fields

- **name** Brief text used to describe this writer when presented. Empty by
  default.
- **filename_extensions** List of filename extensions compatible with this
  writer. The first entry is used as the default if necessary. Empty by default.

### Examples

**Example**

Single-layer writer

```yaml
name: napari
contributions:
  commands:
    - id: napari.write_points
      python_name: napari.plugins._builtins:napari_write_points
      title: Save points layer to csv
  writers:
    - command: napari.write_points
      filename_extensions: [".csv"]
      layer_types: ["points"]
```

**Example**

Multi-layer writer

```yaml
name: napari-svg
contributions:
  commands:
    - id: napari-svg.svg_writer
      python_name: napari-svg.hook_implementations:writer
      title: Save layers as SVG
  writers:
    - command: napari_svg.svg_writer
      layer_types: ["image*", "labels*", "points*", "shapes*", "vectors*"]
      filename_extensions: [".svg"]
```

### Layer type constraints

Given a set of layers, compatible writer plugins are selected based their
_layer type constraints_.

A writer plugin can declare that it will write between _m_ and _n_ layers of a
specific type where _0≤m≤n_.

For example:

```
    image      Write exactly 1 image layer.
    image?     Write 0 or 1 image layers.
    image+     Write 1 or more image layers.
    image*     Write 0 or more image layers.
    image{k}   Write exactly k image layers.
    image{m,n} Write between m and n layers (inclusive range). Must have m<=n.
```

When a type is not present in the list of constraints, that corresponds to a
writer that is not compatible with that type. For example, a writer declaring:

```
    layer_types=["image+", "points*"]
```

would not be selected when trying to write an `image` and a `vector` layer
because the above only works for cases with 0 `vector` layers.

Note that just because a writer declares compatibility with a layer type does
not mean it actually writes that type. In the example above, the writer might
accept a set of layers containing `image`s and `point`s, but the write command
might just ignore the `point` layers. The writer must return `None` for
unwritten layers.

### Calling convention

Currently, two calling conventions are supported for writers: single-layer and
multi-layer writers. When at most one layer can be matched by a writer, it
must use the single-layer convention. Otherwise, the multi-layer convention
must be used.

#### multi-layer writer

```python
def writer_function(
    path: str, layer_data: List[LayerData]
) -> List[str]:
    ...
```

**Parameters**

path(str)
: Path to file, directory, or resource (like a url).

layer_data(list of LayerData)
: Each `LayerData` element is a tuple of `(data,meta,layer_type)`.

**Return**

Returns a list of paths that were successfully written.

#### single-layer writer

```python
def writer_function(
    path: str, data, meta
) -> Optional[str]:
    ...
```

**Parameters**

path(str)
: Path to file, directory, or resource (like a url).

data(array or list of array) : Layer data. Can be N dimensional. If an image
and meta[‘rgb’] is True then the data should be interpreted as RGB or RGBA. If
meta[‘multiscale’] is True, then the data should be interpreted as a
multiscale image.

meta(dict)
: Layer metadata.

**Return**

If data is successfully written, return the path that was written. Otherwise,
if nothing was done, return None.

## Sample Data

There are two ways of specifying sample data.

1. As a **sample data generator**, a function that returns layer data.
2. As a **sample data uri**, a `uri` that should be read using a `reader`
   plugin.

### Required fields

_sample data generator_

- **command** Identifier of a command that returns a layer data tuple.
- **display_name** String to show in the GUI when referring to this sample.
- **key** A key to identify this sample. Must be unique across the samples
  provided by a single plugin.

_sample data uri_

- **uri** Path or URL to a data resource.
- **display_name** String to show in the GUI when referring to this sample.
- **key** A key to identify this sample. Must be unique across the samples
  provided by a single plugin.

### Optional fields

_sample data uri_

- **reader_plugin** Name of plugin to use to open the `uri`.

### Examples

**Example**

Sample data generator

```yaml
name: napari
contributions:
  commands:
    - id: napari.data.astronaut
      title: Generate astronaut sample
      python_name: napari.plugins._skimage_data:astronaut
  sample_data:
    - display_name: Astronaut (RGB)
      key: astronaut
      command: napari.data.astronaut
```

**Example**

Sample data uri

```yaml
name: my-sample
contributions:
  sample_data:
    - display_name: Tabueran Kribati
      key: napari
      uri: https://en.wikipedia.org/wiki/Napari#/media/File:Tabuaeran_Kiribati.jpg
```

### Calling convention

_sample data generator_

```python
def sample_data_generator()->List[LayerData]:
  ...
```

## Widgets

### Required fields

- **command** Identifier of a command that returns a Widget instance.
- **name** User-facing name to use for the widget in, for example, menu items.

### Examples

```yaml
name: napari-animation
contributions:
  commands:
    - id: napari-animation.AnimationWidget
      python_name: napari_animation._hookimpls:AnimationWidget
      title: Open animation wizard
  widgets:
    - command: napari-animation.AnimationWidget
      name: Wizard
```

### Calling Convention

```python

# Bind the constructor as the callable:
# e.g. python_name: my-plugin.MyWidget
class MyWidget(QWidget):
  ...

# Bind the function as the callable:
# e.g. python_name: my_typed_function
@magic_factory
def my_typed_function(...):
  ...
```

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

### Example

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

[epg]: https://packaging.python.org/specifications/entry-points/
[pd]: https://setuptools.pypa.io/en/latest/userguide/datafiles.html
[npe2]: https://github.com/tlambert03/npe2
[json]: https://www.json.org/
[yaml]: https://yaml.org/
[toml]: https://toml.io/
[pydantic]: https://pydantic-docs.helpmanual.io/
[pluginmanifest]: https://github.com/tlambert03/npe2/blob/main/npe2/manifest/schema.py
[pep 8]: https://www.python.org/dev/peps/pep-0008/
[setup-name]: https://packaging.python.org/guides/distributing-packages-using-setuptools/#name
[setup-version]: https://packaging.python.org/guides/distributing-packages-using-setuptools/#version
[setup-desc]: https://packaging.python.org/guides/distributing-packages-using-setuptools/#description
[setup-lic]: https://packaging.python.org/guides/distributing-packages-using-setuptools/#license
[setup-classifier]: https://packaging.python.org/guides/distributing-packages-using-setuptools/#classifiers
[spdx]: https://spdx.org/licenses/
[get-reader-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_get_reader
[get-writer-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_get_writer
[write-image-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_write_image
