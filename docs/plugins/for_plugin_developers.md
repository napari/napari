(plugins-for-plugin-developers)=

# Creating a napari plugin

This document explains how to extend napari's functionality by writing a plugin
that can be installed with `pip` and autodetected by napari.  For more
information on how plugins are implemented internally in napari, see
{ref}`plugins-for-napari-developers`.

## Overview

`napari` supports plugin development through **hooks**:
specific places in the napari
codebase where functionality can be extended.
For example, when a user tries to open a filepath in napari, we
might want to enable plugins to extend the file formats that can be handled.  A
*hook*, then, is the place within napari where we
"promise" to call functions created by external developers & installed by the user.

1. **Hook specifications**:  For each supported hook, we have created
"*hook specifications*", which are
well-documented function signatures that define the API (or
"contract") that a plugin developer must adhere to when writing their function
that we promise to call somewhere in the napari codebase.
See {ref}`plugins-hook-spec`.

2. **Hook implementations**: To make a plugin, plugin developers then write functions ("*hook
implementations*") and mark that function as meeting the requirements of a
specific *hook specification* offered by napari.
See {ref}`plugins-hook-implement`.

3. **Plugin discovery**: Plugins that are installed in the same python
environment as napari can make themselves known to napari. `napari` will then
scan plugin modules for *hook implementations* that will be called at the
appropriate time and place during the execution of `napari`.
See {ref}`plugin-discovery`.

4. **Plugin sharing**: When you are ready to share your plugin, tag your repo
with `napari-plugin`, push a release to pypi, and announce it on Image.sc.
Your plugin will then be available for users on the [napari hub](https://napari-hub.org/). See {ref}`plugin-sharing`.

(plugin-cookiecutter-template)=

## Cookiecutter template

Hook implementations defined by plugin developers typically live inside a
module of a Python package. This allows them to be "pip installable" and
shared via [PyPI](https://pypi.org/) and the [napari hub](https://napari-hub.org/).

To quickly generate a new napari plugin project, you may wish to use the
[cookiecutter-napari-plugin](https://github.com/napari/cookiecutter-napari-plugin) template.  This uses
the [cookiecutter](https://github.com/cookiecutter/cookiecutter) command line
utility, which will ask you a few questions about your project and get you
started with a ready-to-go package layout where you can begin implementing your
plugin.

Install cookiecutter and use the template as follows:

```sh
   pip install cookiecutter
   cookiecutter https://github.com/napari/cookiecutter-napari-plugin
```

The cookiecutter template is a great place to start if you want to focus on
implementing your plugin functionality without having to manually create a
package and all its associated configuration.

(plugins-hook-spec)=

## Step 1: Choose a hook specification to implement

The functionality of plugins, as currently designed and implemented in
`napari`, is rather specific in scope: They are *not* just independent code
blocks with their own GUIs that show up next to the main napari window. Rather,
plugin developers must decide which of the current *hook specifications*
defined by napari they would like to implement.

For a complete list of *hook specifications* that developers can implement, see
the {ref}`hook-specifications-reference`.

A single plugin package may implement more than one *hook specification*, and
each *hook specification* could have multiple *hook implementations* within
a single package.

```{note}
   One of the primary ways that we will extend the functionality of napari over
   time is by identifying new ideas for *hook specifications* that developers
   can implement.  If you have a plugin idea that requires napari to create a
   new hook specification, we'd love to hear about it!  Please think about what
   the signature of your proposed hook specification would look like, and where
   within the napari codebase you'd like your hook implementation to be called,
   and `open a feature request
   <https://github.com/napari/napari/issues/new?template=feature_request.md>`_
   in the napari issue tracker with your proposal.
```

Let's take the {func}`~napari.plugins.hook_specifications.napari_get_reader`
hook (our primary "reader plugin" hook) as an example.  It is defined as:

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
`None` or a function.  From the {func}`docstring <napari.plugins.hook_specifications.napari_get_reader>` of the hook
specification, we see that the implementation should return `None` if the
path is of an unrecognized format, otherwise it should return a
`ReaderFunction`, which is a function that takes a `str` (the filepath to
read) and returns a `list` of `LayerData`, where `LayerData` is any one
of `(data,)`, `(data, meta)`, or `(data, meta, layer_type)`.

That seems like a bit of a mouthful!  But it's a precise (though flexible)
contract that you can follow, and know that napari will handle the rest.

(plugins-hook-implement)=

## Step 2: Write your hook implementation

Once you have identified the {ref}`hook specification <hook-specifications-reference>` that you want to implement, you have to create
a *hook implementation*: a function that accepts the arguments specified by the
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

### Decorating your function with `HookImplementationMarker`

In order to let `napari` know that one of your functions satisfies the API of
one of the napari *hook specifications*, you must decorate your function with
an instance of {class}`~napari_plugin_engine.HookImplementationMarker`,
initialized with the name `"napari"`.  As a convenience, napari provides this
decorator at `napari_plugin_engine.napari_hook_implementation` as shown in
the example above.

However, it's not required to import from or depend on napari *at all* when
writing a plugin. You can import a `napari_hook_implementation` decorator
directly from `napari_plugin_engine` (a very lightweight dependency that uses
only standard lib python).

```python
   from napari_plugin_engine import napari_hook_implementation
```

#### Matching hook implementations to specifications

By default, `napari` matches your implementation to one of our hook
specifications by looking at the *name* of your decorated function.  So in the
example above, because the hook implementation was literally
named `napari_get_reader`, it gets interpreted as an implementation for the
hook specification of the same name.

```python
   @napari_hook_implementation
   def napari_get_reader(path: str):
      ...
```

However, you may also mark *any* function as satisfying a particular napari
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

## Step 3: Make your plugin discoverable

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

## Step 4: Preparing for release

To make your plugin easily discoverable by napari users, you can use the 
`'Framework :: napari'` [classifier](https://pypi.org/classifiers/) in your 
`setup.py` file, which will allow your package to be 
[displayed on the napari-hub](https://napari-hub.org/) and easily searched
for on PyPI.

Once your package, with its `'Framework :: napari'` classifier,  is listed on [PyPI](https://pypi.org/), it will also be visible
on the [napari hub](https://napari-hub.org/), alongside all other napari plugins.

The napari hub reads the metadata of your package and displays it in a number of places
so that users can easily find your plugin and decide if it provides functionality they 
need. Most of this metadata lives inside your package configuration files. You can customize 
your plugin's listing for the hub by following [this guide](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/customizing-plugin-listing.md).

You can also include a napari hub specific description file at `/.napari/DESCRIPTION.md`.
The hub preferentially displays this file over your repository's `README.md` when it's
available. This file allows you to maintain a more developer/repository oriented `README.md`
while still making sure potential users get all the information they need to get started
with your plugin. For more information on this file, see [the guide](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/customizing-plugin-listing.md).

Finally, once you have curated your package metadata and description, you can preview your
metadata, and check any missing fields using the `napari-hub-cli` tool. Install this tool
using 

```sh
   pip install napari-hub-cli
```

and preview your metadata with 

```sh
   napari-hub-cli preview-metadata /tmp/example-plugin
```

For more information on the tool see the repository README](https://github.com/chanzuckerberg/napari-hub-cli).

If you want your plugin to be available on PyPI, but not visible on the napari hub, 
submit an issue on the [napari hub repository]
(https://github.com/chanzuckerberg/napari-hub/issues/new) or send an email to 
`team@napari-hub.org` and it will be removed.

(plugin-sharing)=

## Step 5: Share your plugin with the world

Once you are ready to share your plugin, [upload the Python package to PyPI](https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives)
and it can then be installed with a simple `pip install mypackage`.
If you used the {ref}`plugin-cookiecutter-template`, you can also [setup automated deployments](https://github.com/napari/cookiecutter-napari-plugin#set-up-automatic-deployments).

If you are using Github, add the ["napari-plugin" topic](https://github.com/topics/napari-plugin) to your repo so other developers can
see your work.

The [napari hub](https://www.napari-hub.org/)
automatically displays information about PyPI packages annotated with the
`Framework :: napari` [Trove classifier](https://pypi.org/classifiers/),
to help end users discover plugins that fit their needs.
To ensure you are providing the relevant metadata and description
for your plugin, see the following documentation in the
[napari hub GitHub](https://github.com/chanzuckerberg/napari-hub/tree/main/docs)’s docs folder:

- [Customizing your plugin’s listing](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/customizing-plugin-listing.md)
- [Writing the perfect description for your plugin](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/writing-the-perfect-description.md)

For more about the napari hub, see the [napari hub About page](https://www.napari-hub.org/about).
To learn more about the hub’s development process, see the [napari hub GitHub’s Wiki](https://github.com/chanzuckerberg/napari-hub/wiki).

When you are ready for users, announce your plugin on the [Image.sc Forum](https://forum.image.sc/tag/napari).

## Example plugins

For a minimal working plugin example, see the [napari-dv](https://github.com/tlambert03/napari-dv) plugin, which allows `napari` to
read the [Priism/MRC/Deltavision image file format](https://github.com/tlambert03/mrc).

For a more thorough plugin see [napari-aicsimageio](https://github.com/AllenCellModeling/napari-aicsimageio), one of the first
community plugins developed for napari.  This plugin takes advantage of
{ref}`entry_point discovery <plugin-discovery>` to offer multiple
readers for both in-memory and lazy-loading of image files.

More examples of plugins can be found on the [napari hub](https://napari-hub.org/).

## Help

If you run into trouble creating your plugin, please don't hesitate to reach
out for help in the [Image.sc Forum](https://forum.image.sc/tag/napari).
Alternatively, if you find a bug or have a specific feature request for plugin
support, please open an issue at our [GitHub issue tracker](https://github.com/napari/napari/issues/new/choose).
