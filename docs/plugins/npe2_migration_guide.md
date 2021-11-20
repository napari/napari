(npe2-migration-guide)=

    tutorial
    - hookspec/contribution point mapping
    - tooling - cli for generating a manifest
    - where to use original plugin system
        - what's unsupported at the moment

# npe2: Migrating your plugin the hard way.

`napari` enables users to extend functionality of the program by writing plugins. The [`napari-plugin-engine`][npe1] implements a system used to interact with compatible plugins. However, that system has some significant limitations.

[`npe2`][npe2] is a reimagining of how napari interacts with plugins. Rather than importing a package to discover plugin functionality, a static manifest file is used to declaritively describe a plugin's capabilities. This makes plugin discovery faster and more reliable.

Existing `napari-plugin-engine` plugins expose functionality via _hook implementations_. These are functions decorated to indicate they fullfill a _hook specification_ described by napari. Though there are some exceptions, most _hook implentations_ can be straightforwardly mapped as a _contribution_ in the `npe2` manifest. More information can be found in the manifest [specification][ms].

Below, we will walk through migrating different kinds of plugins to npe2 using examples. At the end we'll describe the "easy way" to migrate.

## Overview

Migration involves these basic steps:

1. Configure a python package to use an npe2 manifest (e.g. by editing `setup.cfg`):
   - Create the _[entry point group][epg]_
   - Make sure the manifest file is added to `package_data`.
2. Create the plugin manifest file.

## Configuring a python package to use a plugin manifest

### 1. Add package metadata for locating the manifest

The manifest file should be specified in the plugin's `setup.cfg` or `setup.py` file using the _[entry point group][epg]_: `napari.manifest`. For example, this would be the section for a plugin `npe2-tester` with `napari.yaml` as the manifest file:

```cfg
[options.entry_points]
napari.manifest =
    npe2-tester = npe2_tester:napari.yaml
```

The manifest file is specified relative to the submodule root path. From the example, that path is: `<path/to/npe2-tester>/npe2_tester/napari.yaml`.

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

## Create the manifest

For each type of _hook specification_ there is a corresponding section below. Each lists the _hook specifications_ that a relevant to that section and an example manifest. For details, refer to the manifest [specification][ms].

### Readers

#### napari_hook_spec

```python=
def napari_get_reader(path: Union[str, List[str]]) -> Optional[ReaderFunction]
```

#### npe2 contributions

```yaml=
display_name: napari
name: napari_builtins
contributions:
  commands:
  - id: napari_builtins.get_reader
    python_name: napari.plugins._builtins:napari_get_reader
    title: Builtin Reader
  readers:
  - command: napari_builtins.get_reader
    accepts_directories: true
    filename_patterns: ['*.csv','*.npy']
```

#### Compatibility

Functions that acted as `napari_get_reader` hooks can be bound as the command for an `npe2` reader.

### Writers: Single-layer writers

#### napari_hook_spec

```python
def napari_write_image(path: str, data: Any, meta: dict) -> Optional[str]
def napari_write_labels(path: str, data: Any, meta: dict) -> Optional[str]
def napari_write_points(path: str, data: Any, meta: dict) -> Optional[str]
def napari_write_shapes(path: str, data: Any, meta: dict) -> Optional[str]
def napari_write_surfaces(path: str, data: Any, meta: dict) -> Optional[str]
def napari_write_vectors(path: str, data: Any, meta: dict) -> Optional[str]
```

#### Example npe2 contribution

```yaml=
name: napari_svg
display_name: napari SVG
entry_point: napari_svg
contributions:
  commands:
  - id: napari_svg.write_image
    python_name: napari_svg.hook_implementations:napari_write_image
    title: Write Image as SVG
  writers:
    - command: napari_svg.write_image
      layer_types: ["image"]
      filename_extensions: [".svg"]
```

#### Compatibility

Functions that act as single-layer writers like `napari_write_image` hooks can be bound as the command for an `npe2` writer. The layer constraint(`layer_types`) and `filename_extensions` fields need to be populated.

### Writers: Multi-layer writers

#### napari_hook_spec

```python
def napari_get_writer(
    path: str, layer_types: List[str]
) -> Optional[WriterFunction]
```

Where the `WriterFunction` is something like:

```python
def writer_function(
    path: str, layer_data: List[Tuple[Any, Dict, str]]
    ) -> List[str]
```

#### Example npe2 contribution

```yaml=
name: napari_svg
display_name: napari SVG
entry_point: napari_svg
contributions:
  commands:
    - id: napari_svg.svg_writer
      title: Write SVG
      python_name: napari_svg.hook_implementations:writer
  writers:
    - command: napari_svg.svg_writer
      layer_types: ["image*","labels*","points*","shapes*","vectors*"]
      filename_extensions: [".svg"]
```

#### Compatibility

A `napari_get_writer` hook may not be bound as an npe2 writer command.

In npe2, the writer specification declares what file-extensions and layer types are compatible. In the original plugin engine, this was the responsibility of the `napari_get_writer` hook.

Usually, the npe2 writer command should be bound to one of the functions returned by `napari_get_writer`. From the example above, this is the `writer` function.

When migrating, you'll need to fill out the `layer_types` and `filename_extensions` used by your writer. `layer_types` is a set of constraints describing the combinations of layer types acceptable by this writer. See the manifest [specification][ms] for details.

In the example above, the svg writer accepts a set of layers with 0 or more images, and 0 or more label layers, and so on. It will not accept surface layers, so if any surface layer is present this writer won't be invoked.

Because layer type constraints are specified in the manifest, no plugin code has to be imported or run until a compatible writer is found.

### Widgets

#### napari_hook_spec

```python=
def napari_experimental_provide_dock_widget() -> Union[
    AugmentedWidget, List[AugmentedWidget]
]
```

or

```python=
def napari_experimental_provide_function() -> Union[
    FunctionType, List[FunctionType]
]
```

#### Example npe2 contribution

```yaml=
name: napari_animation
display_name: animation
entry_point: napari_animation
contributions:
  commands:
    - id: napari_animation.widget
      python_name: napari_animation._qt:AnimationWidget
      title: Make animation widget
  widgets:
    - command: napari_animation.widget
      name: wizard
```

## The easy way: using the `npe2` command line tool

`npe2` provides a command line interface and can be used to generate a template manifest for an installed `napari-plugin-engine`-style plugin.

```
Usage: npe2 [OPTIONS] COMMAND [ARGS]...

Options:
  --help                          Show this message and exit.

Commands:
  convert   Convert existing plugin to new manifest.
  parse     Show parsed manifest as yaml
  validate  Validate manifest for a distribution name or manifest filepath.
```

Let use the command line tool to migrate the `napari-animation` plugin as an example.

First make sure npe2 is installed. Checkout the plugin and locally install it. For example, from the terminal:

```
> pip install npe2
> git clone https://github.com/napari/napari-animation.git
> cd napari-animation
> pip install -e .
```

In this case, the `napari-animation` package contains an _entry point group_ in it's python package metadata:

```ini
[options.entry_points]
napari.plugin =
    animation = napari_animation
```

That metadata gives the name of the plugin: "animation". The name is used in the next step.

To create the manifest, use the npe2 command in the terminal:

```
> npe2 convert animation --out napari_animation/napari.yaml
```

This generates `napari_animation/napari.yaml` with contents:

```yaml=
description: A plugin for making animations in napari
name: napari_animation
publisher: Nicholas Sofroniew, Alister Burt, Guillaume Witz, Faris Abouakil, Talley
  Lambert
version: 0.0.3.dev79+gb8d41cd.d20211116
contributions:
  commands:
  - id: napari_animation.experimental_provide_dock_widget
    python_name: napari_animation._hookimpls:napari_experimental_provide_dock_widget
    title: Experimental Provide Dock Widget
```

> - [ ] TODO: Fix this with the proper output when cli tool is fixed

In this case, the manifest could be created without any intervention, but sometimes the generated manifest needs to be edited. This is especially true for writers.

After editing the generated manifest and making sure it looks right, the package metadata needs to be updated. In `setup.cfg`, edit the `entry_points`:

```ini
[options.entry_points]
# napari.plugin =
	# animation = napari_animation
napari.manifest =
    napari-animation = napari_animation:napari.yaml
```

and make sure the manifest gets included as package data:

```ini
[options]
include_package_data = True
   ...

[options.package_data]
napari_animation =
    napari.yaml
```

All done! Update the local package by repeating

```
> pip install -e .
```

and the next time napari is run `napari-animation` will be discovered as an `npe2` plugin.

[epg]: https://packaging.python.org/specifications/entry-points/
[pd]: https://setuptools.pypa.io/en/latest/userguide/datafiles.html
[npe1]: https://github.com/napari/napari-plugin-engine
[npe2]: https://github.com/tlambert03/npe2
[json]: https://www.json.org/
[yaml]: https://yaml.org/
[toml]: https://toml.io/
[get-reader-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_get_reader
[get-writer-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_get_writer
[write-image-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_write_image
[dock-widget-hook]: https://napari.org/plugins/stable/hook_specifications.html#napari.plugins.hook_specifications.napari_experimental_provide_dock_widget
[ms]: https://hackmd.io/UK4NhwUaSpGkaqUGkx_1OA
