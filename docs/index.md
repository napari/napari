# napari

**napari** is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of [Qt5](https://doc.qt.io/qt-5/) (for the GUI),
[VisPy](http://vispy.org/) (for performant GPU-based rendering), and the
scientific Python stack ([numpy](https://numpy.org/),
[scipy](https://www.scipy.org/)).

![add-image](images/screenshot-add-image.png)

## installation

napari can be installed on most macOS, Linux, and Windows systems with
Python 3.7 and 3.8 using pip:

```sh
pip install 'napari[all]'
```

napari needs a library called [Qt](https://www.qt.io/) to run its user
interface (UI). In Python, there are two alternative libraries to run this,
called [PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5)
and [PySide2](https://doc.qt.io/qtforpython/). By default, we don't choose
for you, and simply running `pip install napari` will not install either. You
*might* already have one of them installed in your environment, thanks to other
scientific packages such as Spyder or matplotlib. If neither is available,
running napari will result in an error message asking you to install one of
them.

To install napari with a specific UI framework, you can use

```sh
pip install 'napari[pyqt5]'
# or
pip install 'napari[pyside2]'
# or
pip install 'napari[all]'
```

This last option (`pip install napari[all]`) will choose a framework for
you — currently PyQt5, but this could change in the future.

## tutorials & getting started

If you are just getting started with napari, we suggest you check out the
tutorials at [napari.org](http://napari.org).  The documentation presented
here is more technical and focuses on understanding how napari works.

# developer introduction

Information on specific functions, classes, and methods is available in the
[API Reference](api/index.rst).
