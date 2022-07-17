(npe2-migration-guide)=

# `npe2` migration guide

This document details how to convert a plugin using the first generation
`napari-plugin-engine`, to the new `npe2` format.  

The primary difference between the first generation and second generation plugin
system relates to how napari *discovers* plugin functionality. In the first
generation plugin engine, napari had to *import* plugin modules to search for
hook implementations decorated with `@napari_hook_implementation`. In `npe2`,
plugins declare their contributions *statically* with a [manifest
file](./manifest).

## Migrating using the `npe2` command line tool

`npe2` provides a command line interface to help convert a
`napari-plugin-engine`-style plugin.

### 1. Install `npe2`

```sh
pip install npe2
```

The `npe2` command line tool provides a few commands to help you develop your
plugin. In this case we're going to use `convert` to *modify* a repository
to fit the new pattern:

```sh
$ npe2 convert --help

Usage: npe2 convert [OPTIONS] PATH

  Convert first generation napari plugin to new (manifest) format.

Arguments:
  PATH  Path of a local repository to convert (package must also be installed
        in current environment). Or, the name of an installed package/plugin.
        If a package is provided instead of a directory, the new manifest will
        simply be printed to stdout.  [required]


Options:
  -n, --dry-runs  Just print manifest to stdout. Do not modify anything
                  [default: False]

  --help          Show this message and exit.
```

### 2. Ensure your plugin is intalled in your environment

This step is critical: your first-generation plugin *must* be installed
in your currently active environment for `npe2 convert` to find it.

Typically this will look something like:

```
conda activate your-env
cd path/to/your/plugin/repository
pip install -e .
```

If `npe2 convert` cannot import your plugin, you will likely get an error
like:

```pytb
PackageNotFoundError: We tried hard! but could not detect a plugin named '...'
```

### 3. Convert your plugin with `npe2 convert`

```{warning}
Executing `npe2 convert .` will **modify** the current directory!
```

The `npe2 convert` command will:

1. Inspect your plugin for hook implementations, and generate an npe2-compatible
   [manifest file](./manifest), called `napari.yaml`.
2. **Modify** your `setup.cfg` to use the new `napari.manifest` entry point, and 
   include the manifest file in your package data.

Use the `npe2 convert` command, passing a path to a plugin
repository (here, the current directory `.`)

```bash
# convert the current directory
❯ npe2 convert .
✔  Conversion complete!
New manifest at /Users/talley/Desktop/napari-animation/napari_animation/napari.yaml.
If you have any napari_plugin_engine imports or hook_implementation decorators, you may remove them now.
```

You are encouraged to inspect the newly-generated `napari.yaml` file.  Refer to
the [manifest](./manifest) and [contributions](./contributions) references pages
for details on each field in the manifest.

```{note}
In some cases the conversion tool may not be able to completely convert your
plugin.  Notable cases include:

- multi-layer writers using the `napari_get_writer` hook specification
- *locally* scoped functions returned from `napari_experimental_provide_function`.
  All [command contributions](contributions-commands)
  must have global `python_paths`.
  
Feel free to contact us on zulip or github if you need help converting!.
```

Now, update the local package metadata by repeating:

```
> pip install -e .
```

The next time napari is run, your plugin should be discovered as an
`npe2` plugin.

----------------

## Migration Reference

> *This section goes into detail on the differences between first-generation and
second-generation implementations. In many cases, this will be more detail than
you need.  If you are still struggling with a specific conversion after using
`npe2 convert` and reading the [contributions](./contributions) reference and
[guides](./guides), this section may be of help.*


Existing `napari-plugin-engine` plugins expose functionality via *hook
implementations*. These are functions decorated to indicate they fullfil a
*hook specification* described by napari. Though there are some exceptions,
most *hook implementations* can be straightforwardly mapped to npe2 [contributions](./contributions)

`npe2` provides a command-line tool that will generate plugin manifests by
inspecting exposed *hook implementations*. Below, we will walk through the
kinds of migrations `npe2 convert` helps with.

For each type of *hook specification* there is a corresponding section below
with migration tips. Each lists the *hook specifications* that are relevant to
that section and an example manifest. For details, refer to the
[Contributions references](./contributions).

### Readers

Functions that acted as `napari_get_reader` hooks can be bound directly as the
command for an `npe2` reader.

#### napari_hook_spec

```python
def napari_get_reader(path: Union[str, List[str]]) -> Optional[ReaderFunction]
```

#### npe2 contributions

```yaml
name: napari
contributions:
  commands:
    - id: napari.get_reader
      python_name: napari.plugins._builtins:napari_get_reader
      title: Read data using napari's builtin reader
  readers:
    - command: napari.get_reader
      accepts_directories: true
      filename_patterns: ["*.csv", "*.npy"]
```

### Writers: Single-layer writers

Functions that act as single-layer writers like `napari_write_image` hooks can
be bound directly as the command for an `npe2` writer. The layer
constraint(`layer_types`) and `filename_extensions` fields need to be
populated.

Since these writers handle only one layer at a time, the `layer_type` is
straightforward: `['image']` for an image writer, `['points']` for a point-set
writer, etc.

The list of `filename_extensions` is used to determine how the writer is
presented in napari's "Save As" dialog.

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

```yaml
name: napari_svg
display_name: napari svg
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

### Writers: Multi-layer writers

In npe2, the writer specification declares what file extensions and layer
types are compatible with a writer. This is a departure from the behavior of
the `napari_get_writer` which was responsible for rejecting data that was
incompatible.

Usually, the npe2 writer command should be bound to one of the functions
returned by `napari_get_writer`. From the example below, this is the `writer`
function.

When migrating, you'll need to fill out the `layer_types` and
`filename_extensions` used by your writer. `layer_types` is a set of
constraints describing the combinations of layer types acceptable by this
writer. More about layer types can be found in the
[Writer contribution guide](layer-type-constraints).

In the example below, the svg writer accepts a set of layers with 0 or more
images, and 0 or more label layers, and so on. It will not accept surface
layers, so if any surface layer is present this writer won't be invoked.

Because layer type constraints are specified in the manifest, no plugin code
has to be imported or run until a compatible writer is found.

#### napari_hook_spec

```python
def napari_get_writer(
    path: str, layer_types: List[str]
) -> Optional[WriterFunction]
```

Where the `WriterFunction` is something like:

```python
def writer(
    path: str, layer_data: List[Tuple[Any, Dict, str]]
    ) -> List[str]
```

#### Example npe2 contribution

```yaml
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
      layer_types: ["image*", "labels*", "points*", "shapes*", "vectors*"]
      filename_extensions: [".svg"]
```

### Widgets

`napari_experimental_provide_dock_widget` hooks return another function that
can be used to instantiate a widget and, optionally, arguments to be passed to
that function.

In contrast the callable for an `npe2` widget contribution is bound to the
function actually instantiating the widget. It accepts only one argument: a
napari `Viewer` proxy instance. The proxy restricts access to some `Viewer`
functionality like private methods.

Similarly `napari_experimental_provide_function` hooks return ane or more
functions to be wrapped with [magicgui]. In `npe2`, each of these functions
should be added as a `Command` contribution with an associated `Widget`
contribution. For each of these `Widget` contributions, the manifest
`autogenerate: true` flag should be set so that `npe2` knows to use `magicgui`.

#### napari_hook_spec

```python
def napari_experimental_provide_dock_widget() -> Union[
    AugmentedWidget, List[AugmentedWidget]
]
```

or

```python
def napari_experimental_provide_function() -> Union[
    FunctionType, List[FunctionType]
]
```

#### Example npe2 contribution

_Dock Widget_

```yaml
name: napari-animation
display_name: animation
contributions:
  commands:
    - id: napari-animation.widget
      python_name: napari_animation._qt:AnimationWidget
      title: Make animation widget
  widgets:
    - command: napari_animation.widget
      display_name: Wizard
```

_Function widget_

```yaml
name: my-function-plugin
display_name: My function plugin!
contributions:
  commands:
    - id: my-function-plugin.func
      python_name: my_function_plugin:my_typed_function
      title: Open widget for my function
  widgets:
    - command: my-function-plugin.func
      display_name: My function
      autogenerate: true # <-- will wrap my_typed_function with magicgui
```

### Sample data providers

Each sample returned from `napari_provide_sample_data()` should be bound as an
individual sample data contribution.

#### napari_hook_spec

```python
def napari_provide_sample_data() -> Dict[str, Union[SampleData, SampleDict]]
```

#### Example npe2 contribution

This example sample data provider:

```python
def _generate_random_data(shape=(512, 512)):
    data = np.random.rand(*shape)
    return [(data, {'name': 'random data'})]

@napari_hook_implementation
def napari_provide_sample_data():
    return {
        'random data': _generate_random_data,
        'random image': 'https://picsum.photos/1024',
        'sample_key': {
            'display_name': 'Some Random Data (512 x 512)'
            'data': _generate_random_data,
        }
    }
```

Should be migrated to:

```yaml
name: my-plugin
contributions:
  commands:
    - id: my-plugin.random
      title: Generate random data
      python_name: my_plugin:_generate_random_data
  sample_data:
    - display_name: Some Random Data (512 x 512)
      key: random data
      command: my-plugin.random
    - uri: https://picsum.photos/1024
      key: random image
```

### Themes

`napari_experimental_provide_theme()` hooks return a dictionary of theme
properties. These properties can be directly embedded in `npe2` theme
contributions. This allows napari to read the theming data without running any
code in the plugin package!

#### Example

The theme provided by this hook:

```python
def get_new_theme() -> Dict[str, Dict[str, Union[str, Tuple, List]]:
    # specify theme(s) that should be added to napari
    themes = {
        "super_dark": {
            "name": "super_dark",
            "background": "rgb(12, 12, 12)",
            "foreground": "rgb(65, 72, 81)",
            "primary": "rgb(90, 98, 108)",
            "secondary": "rgb(134, 142, 147)",
            "highlight": "rgb(106, 115, 128)",
            "text": "rgb(240, 241, 242)",
            "icon": "rgb(209, 210, 212)",
            "warning": "rgb(153, 18, 31)",
            "current": "rgb(0, 122, 204)",
            "syntax_style": "native",
            "console": "rgb(0, 0, 0)",
            "canvas": "black",
        }
    }
    return themes
```

becomes this theme contribution in the plugin manifest:

```yaml
name: my-plugin
contributions:
  themes:
    - label: Super dark
      id: super_dark
      type: dark
      colors:
        background: "rgb(12, 12, 12)"
        foreground: "rgb(65, 72, 81)"
        primary: "rgb(90, 98, 108)"
        secondary: "rgb(134, 142, 147)"
        highlight: "rgb(106, 115, 128)"
        text: "rgb(240, 241, 242)"
        icon: "rgb(209, 210, 212)"
        warning: "rgb(153, 18, 31)"
        current: "rgb(0, 122, 204)"
        syntax_style: "native"
        console: "rgb(0, 0, 0)"
        canvas: "black"
```

## The npe2 adapter

Starting with napari v0.4.17+ napari will begin allowing users to use their
npe1 plugins as if they were npe2 plugins using a "npe1 -> npe2 adaptor"
(this option is at `Preferences > Plugins > Use npe2 adaptor`)

When this option is enabled and a plugin using the legacy plugin manager API
is loaded for the first time, the plugin will be imported as usual and
contributions will be discovered. A "shim" npe2 manifest representing the
plugin's contributions will be created and cached locally.  On all future
launches of napari, that cached manifest will be used and the plugin will
*not* be imported immediately when napari boots.

### Benefits

The benefits for an end-user opting in to the npe2 adaptor are:

- dramatically reduced time to load napari.  By avoiding importing all plugins
  at launch, napari can boot *significantly* faster.
- Plugins are imported lazily, only after one of their commands or menu items
  has been requested.
- It will become clearer to the user which plugin has errored (if any), since it
  will only occur when the plugin's functionality has been requested.
- For napari, the internal codebase becomes *much* simpler since only the npe2 API
  needs to be used.

### Caveats

There are a couple npe1 features that are no longer supported in npe2, and the
following things will be "ignored" for a user using the npe2 adaptor:

- The "plugin sort order" preference is deprecated, and will not be used when
  loading npe2 plugins or npe1 plugins loaded with the npe2 adaptor.
- arguments for `add_dock_widget` returned from
  `napari_experimental_provide_dock_widget` (such as `area=` or
  `add_vertical_stretch=`) will no longer do anything:  `area` will always be
  `'right'` and `add_vertical_stretch` will always be `True`.

There is nothing a plugin can do to prevent a user from using the npe2 adaptor,
it is a user decision.  Furthermore, as we deprecate the legacy
napari-plugin-engine API, the npe2 adapter will likely become the only way that
npe1 plugins are supported in the future, and the option to *not* use the npe2
adaptor will be removed.


[epg]: https://packaging.python.org/specifications/entry-points/
[pd]: https://setuptools.pypa.io/en/latest/userguide/datafiles.html
[npe1]: https://github.com/napari/napari-plugin-engine
[npe2]: https://github.com/tlambert03/npe2
[json]: https://www.json.org/
[yaml]: https://yaml.org/
[toml]: https://toml.io/
[magicgui]: https://napari.org/magicgui
