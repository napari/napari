# napari

### multi-dimensional image viewer for python

[![napari on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/napari/napari/main?urlpath=%2Fdesktop)
[![image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fnapari.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tag/napari)
[![License](https://img.shields.io/pypi/l/napari.svg)](https://github.com/napari/napari/raw/main/LICENSE)
[![Build Status](https://api.cirrus-ci.com/github/Napari/napari.svg)](https://cirrus-ci.com/napari/napari)
[![Code coverage](https://codecov.io/gh/napari/napari/branch/main/graph/badge.svg)](https://codecov.io/gh/napari/napari)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/napari.svg)](https://python.org)
[![Python package index](https://img.shields.io/pypi/v/napari.svg)](https://pypi.org/project/napari)
[![Python package index download statistics](https://img.shields.io/pypi/dm/napari.svg)](https://pypistats.org/packages/napari)
[![Development Status](https://img.shields.io/pypi/status/napari.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![DOI](https://zenodo.org/badge/144513571.svg)](https://zenodo.org/badge/latestdoi/144513571)

**napari** is a fast, interactive, multi-dimensional image viewer for Python. It's designed for browsing, annotating, and analyzing large multi-dimensional images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the scientific Python stack (numpy, scipy).

We're developing **napari** in the open! But the project is in an **alpha** stage, and there will still likely be **breaking changes** with each release. You can follow progress on [this repository](https://github.com/napari/napari), test out new versions as we release them, and contribute ideas and code.

We're working on [tutorials](https://napari.org/tutorials/), but you can also quickly get started by looking below.

## installation

It is recommended to install napari into a virtual environment, like this:

```sh
conda create -y -n napari-env -c conda-forge python=3.9
conda activate napari-env
python -m pip install "napari[all]"
```

If you prefer conda over pip, you can replace the last line with: `conda install -c conda-forge napari`

See here for the full [installation guide](https://napari.org/tutorials/fundamentals/installation.html), including how to [install napari as a bundled app](https://napari.org/tutorials/fundamentals/installation.html#install-as-a-bundled-app).

## simple example

(The examples below require the `scikit-image` package to run. We just use data samples from this package for demonstration purposes. If you change the examples to use your own dataset, you may not need to install this package.)

From inside an IPython shell, you can open up an interactive viewer by calling

```python
from skimage import data
import napari

viewer = napari.view_image(data.cells3d(), channel_axis=1, ndisplay=3)
```

![napari viewer showing an image of an astronaut.](resources/screenshot-add-image.png)

To use napari from inside a script, use `napari.run()`:

```python
from skimage import data
import napari

viewer = napari.view_image(data.cells3d(), channel_axis=1, ndisplay=3)
napari.run()  # start the "event loop" and show the viewer
```

## features

Check out the scripts in our [`examples` folder](examples) to see some of the functionality we're developing!

**napari** supports six main different layer types, `Image`, `Labels`, `Points`, `Vectors`, `Shapes`, and `Surface`, each corresponding to a different data type, visualization, and interactivity. You can add multiple layers of different types into the viewer and then start working with them, adjusting their properties.

All our layer types support n-dimensional data and the viewer provides the ability to quickly browse and visualize either 2D or 3D slices of the data.

**napari** also supports bidirectional communication between the viewer and the Python kernel, which is especially useful when launching from jupyter notebooks or when using our built-in console. Using the console allows you to interactively load and save data from the viewer and control all the features of the viewer programmatically.

You can extend **napari** using custom shortcuts, key bindings, and mouse functions.

## tutorials

For more details on how to use `napari` checkout our [tutorials](https://napari.org/tutorials/). These are still a work in progress, but we'll be updating them regularly.

## mission, values, and roadmap

For more information about our plans for `napari` you can read our [mission and values statement](https://napari.org/community/mission_and_values.html), which includes more details on our vision for supporting a plugin ecosystem around napari.
You can see details of [the project roadmap here](https://napari.org/roadmaps/index.html).

## contributing

Contributions are encouraged! Please read our [contributing guide](https://napari.org/developers/contributing.html) to get started. Given that we're in an early stage, you may want to reach out on our [Github Issues](https://github.com/napari/napari/issues) before jumping in.

## code of conduct

`napari` has a [Code of Conduct](https://napari.org/community/code_of_conduct.html) that should be honored by everyone who participates in the `napari` community.

## governance

You can learn more about how the `napari` project is organized and managed from our [governance model](https://napari.org/community/governance.html), which includes information about, and ways to contact the [@napari/steering-council and @napari/core-devs](https://napari.org/community/team.html#current-core-developers).

## citing napari

If you find `napari` useful please cite [this repository](https://github.com/napari/napari) using its DOI as follows:

> napari contributors (2019). napari: a multi-dimensional image viewer for python. [doi:10.5281/zenodo.3555620](https://zenodo.org/record/3555620)

Note this DOI will resolve to all versions of napari. To cite a specific version please find the
DOI of that version on our [zenodo page](https://zenodo.org/record/3555620). The DOI of the latest version is in the badge at the top of this page.

## help

We're a community partner on the [image.sc forum](https://forum.image.sc/tags/napari) and all help and support requests should be posted on the forum with the tag `napari`. We look forward to interacting with you there.

Bug reports should be made on our [github issues](https://github.com/napari/napari/issues/new?template=bug_report.md) using
the bug report template. If you think something isn't working, don't hesitate to reach out - it is probably us and not you!

## institutional and funding partners

![Chan Zuckerberg Initiative logo](https://chanzuckerberg.com/wp-content/themes/czi/img/logo.svg)
