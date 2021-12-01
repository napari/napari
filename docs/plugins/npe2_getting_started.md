(npe2-getting-started)=

# [npe2][] Getting started guide

This guide will walk you through the steps to write an `npe2`-style napari plugin.

At the end of this guide, you will have a working plugin that can be installed
with `pip` and autodetected by napari.

## Overview

Plugins are special python packages. They define certain functions that napari
calls when it needs to do something like read data. The plugin also needs some
way of declaring what it does so that napari knows it can use it.

This guide covers `npe2`-style plugins. `npe2` plugins declare the
functionality that they contribute in a file called the plugin manifest.

Creating a new plugin from scratch involves the following steps:

1. Configure a python package to use an npe2 manifest (e.g. by editing `setup.cfg`):
   - Create the _[entry point group][epg]_
   - Make sure the manifest file is added to `package_data`.
2. Create the plugin manifest file.

This guide will walk you through the steps using a [cookiecutter][] template
project. cookiecutter is a command line utility that generates a project for
you after asking a few questions.

## 1. Create the project using cookiecutter

Install cookiecutter and use the template as follows:

```bash
pip install cookiecutter
cookiecutter https://github.com/napari/cookiecutter-napari-npe2-plugin
```

`cookiecutter` will start asking you questions. These will include questions
about the functionality you want your plugin to provide. In this guide we'll
focus on creating a reader that can read numpy ('\*.npy') files.

```
...
plugin_name: my_npy_reader
...
include_reader_plugin [y]: y
reader_filename_patterns [*.tif, *.tiff]: *.npy
...
```

Some of the cookiecutter prompts are elided above. We named our plugin
`my_npy_reader`. This is the name that will be used for the python package.
It should conform to the [PEP8][] naming convention.

When the cookiecutter asked to include a reader plugin, we selected "y", and
in the next question we told cookiecutter that our reader should be invoked
for files matching the '\*.npy' glob pattern.

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

Inside `setup.cfg` we added an entry point group that is used to identify
`npe2`-style plugins and to locate the plugin manifest file, `napari.yml`.

```cfg
[options.entry_points]
napari.manifest =
    my-npy-reader = my_npy_reader:napari.yaml
```

The plugin manifest file is specified relative to the submodule root path.
For the example it will be loaded from:
`<path/to/my-npy-reader>/my_npy_reader/src/my_npy_reader/napari.yaml`.

The generated `napari.yml` file looks like this:

```yaml
name: my-npy-reader
author: Hero Protagonist
display_name: My Plugin
entry_point: my_npy_reader

contributions:
  commands:
    - id: my_npy_reader.get_reader
      python_name: my_plugin._reader:napari_get_reader
      title: Open data with napari FooBar
  readers:
    - command: my_npy_reader.get_reader
      accepts_directories: false
      filename_patterns: ['*.npy']
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



[npe2]: https://github.com/napari/npe2
[json]: https://www.json.org/
[yaml]: https://yaml.org/
[toml]: https://toml.io/
[cookiecutter]: https://github.com/cookiecutter/cookiecutter
[pep8]: https://www.python.org/dev/peps/pep-0008/#package-and-module-names
```
