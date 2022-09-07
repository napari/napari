(napari-plugin-engine)=
# 1st Gen Plugin Guide (*Deprecated*)

```{Admonition} DEPRECATED 
:class: warning
We introduced a new plugin engine ([`npe2`][npe2]) in December 2021.

Plugins targeting the first generation `napari-plugin-engine`
(described on this page) will continue to work for at least the
first half of 2022, but we recommend that new plugins use `npe2` and
existing plugins consider migrating soon. See the 
[npe2 migration guide](npe2-migration-guide) for details.

The content below describes the original
[`napari-plugin-engine`](https://github.com/napari/napari-plugin-engine)
and exists for archival reference purposes during the deprecation period.
```


## Overview

`napari` supports plugin development through **hooks**:
specific places in the napari codebase where functionality can be extended.

1. **Hook specifications**: The available hooks are declared as
   "_hook specifications_": function signatures that define the API (or
   "contract") that a plugin developer must adhere to when writing their function
   that we promise to call somewhere in the napari codebase.
   See {ref}`plugins-hook-spec`.

2. **Hook implementations**: To make a plugin, plugin developers then write functions ("_hook
   implementations_") and mark that function as meeting the requirements of a
   specific _hook specification_ offered by napari.
   See {ref}`plugins-hook-implement`.

3. **Plugin discovery**: Plugins that are installed in the same python
   environment as napari can make themselves known to napari. `napari` will then
   scan plugin modules for _hook implementations_ that will be called at the
   appropriate time and place during the execution of `napari`.
   See {ref}`plugin-discovery`.

4. **Plugin sharing**: When you are ready to share your plugin, tag your repo
   with `napari-plugin`, push a release to pypi, and announce it on Image.sc.
   Your plugin will then be available for users on the
   [napari hub](https://napari-hub.org/). See {ref}`plugin-sharing`.

(plugins-hook-spec)=

### Step 1: Choose a hook specification to implement

The functionality of plugins, as currently designed and implemented in
`napari`, is rather specific in scope: They are _not_ just independent code
blocks with their own GUIs that show up next to the main napari window. Rather,
plugin developers must decide which of the current _hook specifications_
defined by napari they would like to implement.

For a complete list of _hook specifications_ that developers can implement, see
the {ref}`hook-specifications-reference`.

A single plugin package may implement more than one _hook specification_, and
each _hook specification_ could have multiple _hook implementations_ within
a single package.


Let's take the {func}`~napari.plugins.hook_specifications.napari_get_reader`
hook (our primary "reader plugin" hook) as an example. It is defined as:

```python
   LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
   ReaderFunction = Callable[[str], List[LayerData]]

   @napari_hook_specification(firstresult=True)
   def napari_get_reader(
       path: Union[str, List[str]]
   ) -> Optional[ReaderFunction]:
       ...
```

Note that it takes a `str` or a `list` of `str` and either returns
`None` or a function. From the {func}`docstring <napari.plugins.hook_specifications.napari_get_reader>` of the hook
specification, we see that the implementation should return `None` if the
path is of an unrecognized format, otherwise it should return a
`ReaderFunction`, which is a function that takes a `str` (the filepath to
read) and returns a `list` of `LayerData`, where `LayerData` is any one
of `(data,)`, `(data, meta)`, or `(data, meta, layer_type)`.

That seems like a bit of a mouthful! But it's a precise (though flexible)
contract that you can follow, and know that napari will handle the rest.

(plugins-hook-implement)=

### Step 2: Write your hook implementation

Once you have identified the {ref}`hook specification <hook-specifications-reference>` that you want to implement, you have to create
a _hook implementation_: a function that accepts the arguments specified by the
hook specification signature and returns a value with the expected return type.

Here's an example hook implementation for
{func}`~napari.plugins.hook_specifications.napari_get_reader` that enables
napari to open a numpy binary file with a `.npy` extension (previously saved
with {func}`numpy.save`)

```python
   import numpy as np
   from napari_plugin_engine import napari_hook_implementation

   def npy_file_reader(path):
      array = np.load(path)
      # return it as a list of LayerData tuples,
      # here with no optional metadata
      return [(array,)]

   # this line is explained below in "Decorating your function..."
   @napari_hook_implementation
   def napari_get_reader(path):
      # remember, path can be a list, so we check its type first...
      # (this example plugin doesn't handle lists)
      if isinstance(path, str) and path.endswith(".npy"):
         # If we recognize the format, we return the actual reader function
         return npy_file_reader
      # otherwise we return None.
      return None
```

(hookimplementation-decorator)=

#### Decorating your function with `HookImplementationMarker`

In order to let `napari` know that one of your functions satisfies the API of
one of the napari _hook specifications_, you must decorate your function with
an instance of {class}`~napari_plugin_engine.HookImplementationMarker`,
initialized with the name `"napari"`. As a convenience, napari provides this
decorator at `napari_plugin_engine.napari_hook_implementation` as shown in
the example above.

However, it's not required to import from or depend on napari _at all_ when
writing a plugin. You can import a `napari_hook_implementation` decorator
directly from `napari_plugin_engine` (a very lightweight dependency that uses
only standard lib python).

```python
   from napari_plugin_engine import napari_hook_implementation
```

##### Matching hook implementations to specifications

By default, `napari` matches your implementation to one of our hook
specifications by looking at the _name_ of your decorated function. So in the
example above, because the hook implementation was literally
named `napari_get_reader`, it gets interpreted as an implementation for the
hook specification of the same name.

```python
   @napari_hook_implementation
   def napari_get_reader(path: str):
      ...
```

However, you may also mark _any_ function as satisfying a particular napari
hook specification (regardless of the function's name) by providing the name of
the target hook specification to the `specname` argument in your
implementation decorator:

```python
   @napari_hook_implementation(specname="napari_get_reader")
   def whatever_name_you_want(path: str):
      ...
```

This allows you to specify multiple hook implementations of the same hook
specification in the same module or class, without needing a separate entry point.

(plugin-discovery)=

### Step 3: Make your plugin discoverable

Packages and modules installed in the same environment as `napari` may make
themselves "discoverable" to napari using package metadata, as outlined in the
[Python Packaging Authority guide](https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata).

By providing an `entry_points` argument with the key `napari.plugin` to
`setup()` in `setup.py`, plugins can register themselves for discovery.

For example if you have a package named `mypackage` with a submodule
`napari_plugin` where you have decorated one or more napari hook
implementations, then if you include in `setup.py`:

```python
   # setup.py

   setup(
      ...
      entry_points={'napari.plugin': 'plugin-name = mypackage.napari_plugin'},
      ...
   )
```

... then napari will search the `mypackage.napari_plugin` module for
functions decorated with the `HookImplementationMarker("napari")` decorator
and register them under the plugin name `"plugin-name"`.

A user would then be able to use `napari`, extended with your package's
functionality by simply installing your package along with napari:

```sh
   pip install napari mypackage
```

(plugin-sharing)=
### Step 4: Deploy your plugin

See [testing and deploying](./test_deploy) your plugin.  (This hasn't changed
significantly with the secod generation (`npe2`) plugin engine).


(plugin-cookiecutter-template)=

## Cookiecutter template

To quickly generate a new napari plugin project, you may wish to use the
[cookiecutter-napari-plugin](https://github.com/napari/cookiecutter-napari-plugin) template. This uses
the [cookiecutter](https://github.com/cookiecutter/cookiecutter) command line
utility, which will ask you a few questions about your project and get you
started with a ready-to-go package layout where you can begin implementing your
plugin.

Install cookiecutter and use the template as follows:

```sh
pip install cookiecutter
cookiecutter https://github.com/napari/cookiecutter-napari-plugin
```

See the [readme](https://github.com/napari/cookiecutter-napari-plugin) for details


----------------------------

(hook-specifications-reference)=
## Hook Specification Reference


```{eval-rst}
.. automodule:: napari.plugins.hook_specifications
  :noindex:

.. currentmodule:: napari.plugins.hook_specifications
```

### IO hooks

```{eval-rst}
.. autofunction:: napari_provide_sample_data
.. autofunction:: napari_get_reader
.. autofunction:: napari_get_writer
```

(write-single-layer-hookspecs)=

#### Single Layers IO

The following hook specifications will be called when a user saves a single
layer in napari, and should save the layer to the requested format and return
the save path if the data are successfully written. Otherwise, if nothing was saved, return `None`.
They each accept a `path`.
It is up to plugins to inspect and obey the extension of the path (and return
`False` if it is an unsupported extension). The `data` argument will come
from `Layer.data`, and a `meta` dict that will correspond to the layer's
{meth}`~napari.layers.base.base.Layer._get_state` method.

```{eval-rst}
.. autofunction:: napari_write_image
.. autofunction:: napari_write_labels
.. autofunction:: napari_write_points
.. autofunction:: napari_write_shapes
.. autofunction:: napari_write_surface
.. autofunction:: napari_write_vectors
```

### Analysis hooks

```{eval-rst}
.. autofunction:: napari_experimental_provide_function
```

### GUI hooks

```{eval-rst}
.. autofunction:: napari_experimental_provide_theme
.. autofunction:: napari_experimental_provide_dock_widget
```

## Help

If you run into trouble creating your plugin, please don't hesitate to reach
out for help in the [Image.sc Forum](https://forum.image.sc/tag/napari).
Alternatively, if you find a bug or have a specific feature request for plugin
support, please open an issue at our [GitHub issue tracker](https://github.com/napari/napari/issues/new/choose).

[npe2]: https://github.com/napari/npe2
