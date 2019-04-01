<h4 align='center'><i>this repo is under development and not stable!</i></h4>

<h1 align='center'>napari</h1>

<h4 align='center'>multi-dimensional image viewer for python</h4>

<p align="center">
  <a href="https://github.com/napari/napari/raw/master/LICENSE"><img src="https://img.shields.io/pypi/l/napari.svg" alt="License"></a>
  <a href="https://cirrus-ci.com/napari/napari"><img src="https://api.cirrus-ci.com/github/Napari/napari.svg" alt="Build Status"></a>
  <a href="https://python.org"><img src="https://img.shields.io/pypi/pyversions/napari.svg" alt="Python Version"></a>
  <a href="https://pypi.org/project/napari"><img src="https://img.shields.io/pypi/v/napari.svg" alt="PyPI"></a>
  <a href="https://pypistats.org/packages/napari"><img src="https://img.shields.io/pypi/dm/napari.svg" alt="PyPI - Downloads"></a>
  <a href="https://github.com/napari/napari"><img src="https://img.shields.io/pypi/status/napari.svg" alt="Development Status"></a>

</p>
<br>

**napari** is a fast, interactive, multi-dimensional image viewer for Python. It's designed for browsing, annotating, and analyzing large multi-dimensional images. It's built on top of `PyQt` (for the GUI), `vispy` (for performant GPU-based rendering), and the scientific Python stack (`numpy`, `scipy`).

We're developing **napari** in the open! But the project is in a **pre-alpha** stage. You can follow progress on this repository, test out new versions as we release them, and contribute ideas and code. Expect **breaking changes** from patch to patch.

## installation

napari can be installed on most Mac OS X and Linux systems with Python 3.6 or 3.7 by calling 

```sh
$ pip install napari
```

We're working on adding Windows support.

## simple example

From inside the IPython shell or a Jupyter notebook you can run the following to open up an interactive viewer

```python
%gui qt5
from skimage import data
from napari import ViewerApp
viewer = ViewerApp(data.astronaut())
```

![image](resources/screenshot-add-image.png)

To do the same thing inside a script call

```python
from skimage import data
from napari import ViewerApp
from napari.util import app_context

with app_context():
	viewer = ViewerApp(data.astronaut())
```

## more features

Check out the scripts in the `examples` folder to see some of the functionality we're developing!

For example, you can add multiple images in different layers and adjust them

```python
from skimage import data
from skimage.color import rgb2gray
from napari import ViewerApp
from napari.util import app_context

with app_context():
    viewer = ViewerApp(astronaut=rgb2gray(data.astronaut()),
                       photographer=data.camera(),
                       coins=data.coins(),
                       moon=data.moon())

    viewer.layers.remove('coins')
    viewer.layers['astronaut', 'moon'] = viewer.layers['moon', 'astronaut']
```

![image](resources/screenshot-layers.png)

You can add markers on top of an image

```python
from skimage import data
from skimage.color import rgb2gray
from napari import ViewerApp
from napari.util import app_context

with app_context():
    viewer = ViewerApp()
    viewer.add_image(rgb2gray(data.astronaut()))
    points = np.array([[100, 100], [200, 200], [333, 111]])
    size = np.array([10, 20, 20])
    markers = viewer.add_markers(points, size=size)
```

![image](resources/screenshot-add-markers.png)

napari support bidirectional communication between the viewer and the Python kernel, which is especially useful in Jupyter notebooks -- in the example above you can retrieve the locations of the markers, including any additional ones you have drawn, by calling

```python
markers.coords
>> [[100 100]
    [200 200]
    [333 111]]
```

Finally, you can render and quickly browse slices of multi-dimensional arrays

```python

import numpy as np
from skimage import data
from napari import ViewerApp
from napari.util import app_context

with app_context():
    blobs = np.stack([data.binary_blobs(length=128, blob_size_fraction=0.05,
                                        n_dim=3, volume_fraction=f)
                     for f in np.linspace(0.05, 0.5, 10)], axis=-1)
    viewer = ViewerApp(blobs.astype(float))
```

![image](resources/screenshot-nD-image.png)

## plans

We're working on several features, including 

- shape-based annotation (for drawing polygons and bounding boxes)
- region labeling (for defining segmentation)
- 3D volumetric rendering
- support for a plugin ecosystem (for integrating image processing and machine learning tools)

See [this issue](https://github.com/napari/napari/issues/141) for some of the key use cases we're trying to enable, and feel free to add comments or ideas!

## contributing

Contributions are encouraged! Please read [our guide](https://github.com/napari/napari/blob/master/CONTRIBUTING.md) to get started. Given that we're in an early stage, you may want to reach out on [Github Issues](https://github.com/napari/napari/issues) before jumping in.
