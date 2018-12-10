<h1> !!!This repo is under development and NOT stable!!! </h1>

# napari-gui
[![License](https://img.shields.io/pypi/l/napari-gui.svg)](https://github.com/Napari/napari-gui/raw/master/LICENSE)
[![Build Status](https://api.cirrus-ci.com/github/Napari/napari-gui.svg)](https://cirrus-ci.com/Napari/napari-gui)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-gui.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/napari-gui.svg)](https://pypi.org/project/napari-gui)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/napari-gui.svg)](https://pypistats.org/packages/napari-gui)
[![Development Status](https://img.shields.io/pypi/status/napari-gui.svg)](https://github.com/Napari/napari-gui)

---

Napari GUI is a high-performance, multi-dimensional image viewing tool and the official front-end of the [Napari](https://github.com/Napari) ecosytem.
It is designed to integrate with other [Napari plugins](https://github.com/topics/napari-plugin) while also serving as a stand-alone package.

This project is in the **pre-alpha** stage; **expect breaking changes** to be made from patch to patch.

## Installation

Napari GUI can be installed on most operating systems with `pip install napari-gui`.
It will fall under the `napari_gui` namespace.

### Downloading OpenGL

Windows and Linux users may need to install the [Mesa OpenGL](https://www.mesa3d.org/) library.

`vispy` provides an easily-downloadable (but out-of-date) Windows version for [64-](https://github.com/vispy/demo-data/raw/master/mesa/opengl32_mingw_64.dll) and [32-bit](https://github.com/vispy/demo-data/raw/master/mesa/opengl32_mingw_64.dll) operating systems.

Linux users can complete the installation process from the command line:

```sh
$ sudo apt-get update
$ sudo apt-get install libgl1-mesa-glx
```

## Contributing

Contributions are encouraged! Please read [our guide](https://github.com/Napari/napari-gui/blob/master/CONTRIBUTING.md) to get started.
