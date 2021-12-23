(npe2-migration-guide)=

# npe2 migration guide

```{warning}
This guide is still a work in progress.

There are inaccuracies and mistakes. If you notice any, feel free to submit
issues to the [napari github repository](https://github.com/napari/napari).
```

We've introduced a new plugin engine. The new library [npe2] is a
re-imagining of how napari interacts with plugins. Rather than importing a
package to discover plugin functionality, a static manifest file is used to
declaratively describe a plugin's capabilities. Details can be found in the
[](npe2-manifest-spec).

Plugins targeting `napari-plugin-engine` will continue to work, but we
recommend migrating to `npe2`. This guide will help you learn how!

## Migrating using the `npe2` command line tool

`npe2` provides a command line interface to help convert a
`napari-plugin-engine`-style plugin.

### 1. Install the `npe2` command

```bash
pip install npe2
npe2 convert --help
```

The `npe2` tool provides a few commands to help you develop your plugin. In
this case we're asking for help with the `convert` command:

```
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

### 2. Install your `napari-plugin-engine` based plugin.

As an example, we'll walk through the process using the `napari-animation`
plugin.

We checkout the plugin and install it as a local editable package.

```bash
git clone https://github.com/napari/napari-animation.git
cd napari-animation
pip install -e .
```

### 3. Inspect package metadata

The `napari-animation` package defines it's core metadata in `setup.cfg`.
Inside, it defines an _entry point group_. That section should initially
contain:

```ini
[options.entry_points]
napari.plugin =
    animation = napari_animation
```

That metadata gives the name of the plugin: "animation". The name is used in
the next step. Later we're going to come back and update this section.

### 4. Convert your plugin.

Use the `npe2` command in the terminal:

```bash
# we're in the napari-animation directory
> npe2 convert .
✔  Conversion complete!
If you have any napari_plugin_engine imports or hook_implementation decorators, you may remove them now.
```

```{note}
This step uses `napari-plugin-engine` to discover the plugins installed on the
system. If you have other plugins installed there's a chance they may interfere.
```

This generates `napari_animation/napari.yaml` and modifies the package metadta
(in `setup.cfg` or `setup.py`).

The plugin manifest contains:

```yaml
# Manifest is written to napari-animation/napari_animation/napari.yaml
contributions:
  commands:
    - id: napari-animation.AnimationWidget
      python_name: napari_animation._hookimpls:AnimationWidget
      title: Create Wizard
  widgets:
    - command: napari-animation.AnimationWidget
      display_name: Wizard
engine: 0.1.0
name: napari-animation
```

```{note}
In this case, the manifest could be created without any intervention, but
sometimes the generated manifest needs to be edited. The conversion tool will
let you know when this happens.
```

The package metadata will have a new entry point. The old one is removed:

```diff
[options.entry_points]
- napari.plugin =
-    animation = napari_animation
+ napari.manifest =
+    napari-animation = napari_animation:napari.yaml
```

the manifest was also added to `options.package_data` so that it will be
included with any distribution.

```diff
+ [options.package_data]
+ napari_animation = napari.yaml
```

All done! Update the local package metadata by repeating:

```
> pip install -e .
```

and the next time napari is run `napari-animation` will be discovered as an
`npe2` plugin!

## Migration details

Existing `napari-plugin-engine` plugins expose functionality via _hook
implementations_. These are functions decorated to indicate they fullfil a
_hook specification_ described by napari. Though there are some exceptions,
most _hook implementations_ can be straightforwardly mapped as a
_contribution_ in the `npe2` manifest. More information can be found in the
{ref}`npe2-manifest-spec` - Look for "calling conventions" for each field.

`npe2` provides a command-line tool that will generate plugin manifests by
inspecting exposed _hook implementations_. Below, we will walk through the
kinds of migrations `npe2 convert` helps with.

For each type of _hook specification_ there is a corresponding section below
with migration tips. Each lists the _hook specifications_ that are relevant to
that section and an example manifest. For details, refer to the
{ref}`npe2-manifest-spec`.

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
writer. More about layer types can be found in the {ref}`npe2-manifest-spec`.

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
[magicgui]: https://napari.org/magicgui
