(npe2-getting-started)=

# npe2 getting started guide

This guide will walk you through the steps to write an `npe2`-style napari
plugin from scratch.

At the end of this guide, you will have a working plugin that can be installed
with `pip` and autodetected by napari.

## Overview

Plugins are just python packages. They include contributions that napari
may use when performing tasks (like reading data), and a "manifest" file that
tells napari where in the package to find these contributions.

This guide covers `npe2`-style plugins. `npe2` plugins declare the
functionality that they contribute in a file called the plugin manifest.

Creating a new plugin involves the following steps:

1. Configure a python package to use an npe2 manifest (e.g. by editing
   `setup.cfg`):
   - Create the _entry point group_.
   - Make sure the manifest file is added to `package_data`.
2. Create the plugin manifest file.

This guide will walk you through the steps using a [cookiecutter][] template.
cookiecutter is a command line utility that generates a python package for you
after asking a few questions. In this case, the package contains all the
boilerplate needed for documenting, testing and deploying your plugin.

```{note}
Minimally, a plugin could be just three files: `setup.cfg`, a python file, and
the plugin manifest file.
```

## 1. Create the project using cookiecutter

Install cookiecutter and use the template as follows:

```bash
pip install cookiecutter
cookiecutter https://github.com/napari/cookiecutter-napari-plugin --checkout npe2
```

`cookiecutter` will ask you a series of questions
about the functionality you want your plugin to provide. In this guide we'll
focus on creating a reader that can read numpy ('\*.npy') files.

```sh
# questions asked when running cookiecutter:
...
plugin_name: my-npy-reader
...
include_reader_plugin [y]: y
...
```

Some of the cookiecutter prompts are elided above. We named our plugin
`my_npy_reader`. This is the name that will be used for the python package. It
should conform to the [PEP8][] naming convention.

When the cookiecutter asked to include a reader plugin, we selected `y`, and
in the next question we told cookiecutter that our reader should be invoked
for files matching the `\*.npy` glob pattern.

After answering all the prompts, `cookiecutter` will create a directory called
`my_npy_reader` in the current directory that holds the generated files. It
will look something like this:

```
my-npy-reader
├── LICENSE
├── MANIFEST.in
├── README.md
├── docs
│   └── index.md
├── mkdocs.yml
├── requirements.txt
├── setup.cfg                  <-- Defines an entry point group
├── setup.py                       pointing to the plugin manifest
├── src
│   └── my_npy_reader
│       ├── __init__.py
│       ├── _reader.py         <-- example reader code
│       ├── _tests
│       │   ├── __init__.py
│       │   └── test_reader.py
│       └── napari.yml         <-- The plugin manifest
└── tox.ini
```

Inside `setup.cfg` we added an [entry point group][epg] that is used to identify
`npe2`-style plugins and to locate the plugin manifest file, `napari.yml`.

```cfg
[options.entry_points]
napari.manifest =
    my-npy-reader = my_npy_reader:napari.yaml
```

The plugin manifest file is specified relative to the top level module path. For
the example it will be loaded from:
`<path/to/my-npy-reader>/my_npy_reader/src/my_npy_reader/napari.yaml`.

The generated `napari.yml` file looks like this:

```yaml
name: my-npy-reader
display_name: My Plugin
contributions:
  commands:
    - id: my-npy-reader.get_reader
      python_name: my_plugin._reader:napari_get_reader
      title: Open data with napari FooBar
  readers:
    - command: my-npy-reader.get_reader
      accepts_directories: false
      filename_patterns: ["*.npy"]
```

With `npe2` installed, we can check that this is a valid plugin manifest:

```bash
> npe2 validate src/my_npy_reader
✔ Manifest for 'napari FooBar' valid!
```

See the {ref}`npe2-manifest-spec` for more details.

## 2. Write code to implement your plugin

For our example `*.npy` reader, edit `src/my_npy_reader/_reader.py`:

```python
import numpy as np

def npy_file_reader(path):
    array = np.load(path)
    # return it as a list of LayerData tuples,
    # here with no optional metadata
    return [(array,)]

# This is the function referenced by the reader command in `napari.yml`.
def napari_get_reader(path):
    # remember, path can be a list, so we check its type first...
    # (this example plugin doesn't handle lists)
    if isinstance(path, str) and path.endswith(".npy"):
        # If we recognize the format, we return the actual reader function
        return npy_file_reader
    # otherwise we return None.
    return None
```

You can start testing your plugin using a local editable install, and opening
napari:

```bash
pip install -e .
napari
```

## 3. Make your plugin discoverable

Packages and modules installed in the same environment as napari may make
themselves discoverable to napari using package metadata.

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

The manifest file also needs to be included as _[package data][pd]_ in
distributable forms for the package. For example:

```toml
[metadata]
...
include_package_data=True

[options.package_data]
npe2_tester =
    napari.yaml
```

A user can install napari and your plugin with:

```bash
pip install napari myplugin
```

## 4. Preparing for release

Use the `Framework :: napari` [classifier](https://pypi.org/classifiers/) in
your package's core metadata to make your plugin more discoverable.

Once your package, with its `Framework :: napari` classifier, is listed on
PyPI, it will also be visible on the [napari hub][hub], alongside all other
napari plugins.

You can customize your plugin’s listing for the hub by following this
[guide][hubguide].

The napari hub reads the metadata of your package and displays it in a number
of places so that users can easily find your plugin and decide if it provides
functionality they need. Most of this metadata lives inside your package
configuration files.

You can also include a napari hub specific description file at
/.napari/DESCRIPTION.md. The hub preferentially displays this file over your
repository’s README.md when it’s available. This file allows you to maintain a
more developer/repository oriented README.md while still making sure potential
users get all the information they need to get started with your plugin.

Finally, once you have curated your package metadata and description, you can
preview your metadata, and check any missing fields using the `napari-hub-cli`
tool. Install this tool using

```bash
pip install napari-hub-cli
```

and preview your metadata with

```bash
napari-hub-cli preview-metadata /tmp/example-plugin
```

For more information on the tool see the repository README](https://github.com/chanzuckerberg/napari-hub-cli).

If you want your plugin to be available on PyPI, but not visible on the napari hub,
submit an issue on the [napari hub repository]
(https://github.com/chanzuckerberg/napari-hub/issues/new) or send an email to
`team@napari-hub.org` and it will be removed.

## 5. Share your plugin with the world

Once you are ready to share your plugin, [upload the Python package to
PyPI][pypi-upload] and it can then be installed with a simple `pip install mypackage`. If you used the {ref}`plugin-cookiecutter-template`, you can also
[setup automated deployments][autodeploy].

If you are using Github, add the ["napari-plugin"
topic](https://github.com/topics/napari-plugin) to your repo so other
developers can see your work.

The [napari hub](https://www.napari-hub.org/) automatically displays
information about PyPI packages annotated with the `Framework :: napari`
[Trove classifier](https://pypi.org/classifiers/), to help end users discover
plugins that fit their needs. To ensure you are providing the relevant
metadata and description for your plugin, see the following documentation in
the [napari hub
GitHub](https://github.com/chanzuckerberg/napari-hub/tree/main/docs)’s docs
folder:

- [Customizing your plugin’s listing](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/customizing-plugin-listing.md)
- [Writing the perfect description for your plugin](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/writing-the-perfect-description.md)

For more about the napari hub, see the [napari hub About page](https://www.napari-hub.org/about).
To learn more about the hub’s development process, see the [napari hub GitHub’s Wiki](https://github.com/chanzuckerberg/napari-hub/wiki).

When you are ready for users, announce your plugin on the [Image.sc Forum](https://forum.image.sc/tag/napari).

[npe2]: https://github.com/napari/npe2
[json]: https://www.json.org/
[yaml]: https://yaml.org/
[toml]: https://toml.io/
[cookiecutter]: https://github.com/cookiecutter/cookiecutter
[pep8]: https://www.python.org/dev/peps/pep-0008/#package-and-module-names
[epg]: https://packaging.python.org/specifications/entry-points/
[pd]: https://setuptools.pypa.io/en/latest/userguide/datafiles.html
[hub]: https://www.napari-hub.org/
[hubguide]: https://github.com/chanzuckerberg/napari-hub/blob/main/docs/customizing-plugin-listing.md
[pypi-upload]: https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives
[autodeploy]: https://github.com/napari/cookiecutter-napari-plugin#set-up-automatic-deployments
