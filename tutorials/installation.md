# napari installation tutorial

Welcome to the **napari** installation tutorial.

## instalation

**napari** can be installed on most Mac OS X and Linux systems with Python 3.6 or 3.7 by calling

```sh
$ pip install napari
```

We're working on improving Windows support.

To install from the master branch on Github use

```sh
$ pip install git+https://github.com/napari/napari
```

To clone the repository locally and install in editable mode use

```sh
$ git clone https://github.com/napari/napari.git
$ cd napari
$ pip install -e .
```

## upgrading

If you installed **napari** with `pip` you can upgrade by calling
```sh
$ pip install napari --upgrade
```

## troubleshooting

We're currently working on improving our windows support. For Mac 0S X we also require at least version 10.12.


## next steps

Now that you've got **napari** installed, checkout our [getting started tutorial](getting_started.md) to start learning how to use it!

## all tutorials

- [installing napari](installation.md)
- [getting started tutorial](getting_started.md)
- [napari viewer tutorial](viewer.md)
- [multidimensional tutorial](multidimensional_dimensional.md)
- [image layer tutorial](image.md)
- [labels layer tutorial](labels.md)
- [points layer tutorial](points.md)
- [shapes layer tutorial](shapes.md)
- [pyramid layer tutorial](pyramid.md)
- [vectors layer tutorial](vectors.md)
