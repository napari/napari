# How to install napari as a Python package

```{note}
This installation method allows you to use napari from Python to programmatically interact with the app. It is the best way to install napari and make full use of all its features.
```

```{note}
If you want to install napari as an app without worrying about Python environment, please follow the [bundled app installation guide](installation_bundle.md).
```

```{note} 
If you want to contribute code back into napari, please follow the [development installation instructions in the contributing guide](https://napari.org/developers/contributing.html).
```

## Prerequisites

It requires:
- Python 3.7 or higher from Python website(https://www.python.org/downloads/) or the [Anaconda distribution](https://www.anaconda.com/distribution/) (recommended)
- the ability to install python packages via [pip](https://pypi.org/project/pip/) or [conda-forge](https://conda-forge.org/docs/user/introduction.html)

You may also want:
- an environment manager like [conda](https://docs.conda.io/en/latest/miniconda.html) or
[venv](https://docs.python.org/3/library/venv.html) **(Highly recommended)**
- to learn more about [managing conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## Installation steps

Python package distributions of napari can be installed via `pip`, `conda-forge`, or from source.

````{important}
While not strictly required, it is highly recommended to install
napari into a clean virtual environment using an environment manager like
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or
[venv](https://docs.python.org/3/library/venv.html).

This should be set up *before* you install napari. For example, with `conda`:

```sh
conda create -y -n napari-env -c conda-forge python=3.9
conda activate napari-env
```
````

Choose one of the options below to install napari as a Python package.

````{admonition} **1. From pip**
:class: dropdown

napari can be installed on most macOS, Linux, and Windows systems with Python 3.7, 3.8, 3.9, and 3.10 using pip:

```sh
python -m pip install "napari[all]"
```
To upgrade napari to a new version:

```sh
python -m pip install "napari[all]" --upgrade
```

*(See [Choosing a different Qt backend](#choosing-a-different-qt-backend) below for an explanation of the `[all]`
notation.)*

````


````{admonition} **2. From conda-forge**
:class: dropdown

If you prefer to manage packages with conda, napari is available on the
conda-forge channel. You can install it with:

```sh
conda install -c conda-forge napari
```

To upgrade napari to a new version: 

```sh
conda update napari
```
````


````{admonition} **3. From the main branch on Github**
:class: dropdown

To install the latest version with yet to be released features from github via pip:

```sh
python -m pip install "git+https://github.com/napari/napari.git#egg=napari[all]"
```
````

## Choosing a different Qt backend

napari needs a library called [Qt](https://www.qt.io/) to run its user interface
(UI). In Python, there are two alternative libraries to run this, called
[PyQt5](https://www.riverbankcomputing.com/software/pyqt/download5) and
[PySide2](https://doc.qt.io/qtforpython/). By default, we don't choose for you,
and simply running `pip install napari` will not install either. You *might*
already have one of them installed in your environment, thanks to other
scientific packages such as Spyder or matplotlib. If neither is available,
running napari will result in an error message asking you to install one of
them.

Running `pip install "napari[all]"` will install the default framework – currently
PyQt5, but this could change in the future.

To install napari with a specific framework, you can use:

```sh
pip install "napari[pyqt5]"    # for PyQt5

# OR
pip install "napari[pyside2]"  # for PySide2
```

```{note}
If you switch backends, it's a good idea to `pip uninstall` the one
you're not using.
```

## Launching napari

There are four ways to launch the **napari** viewer:

- Command line
- Python script
- IPython console
- Jupyter notebook

All four of these methods will launch the same napari viewer, but depending on your use-case different ones may be preferable.

### Command line usage

To launch napari from the command line simply run

```sh
napari
```

This command will launch an empty viewer:

![image: an empty napari viewer](../assets/tutorials/launch_cli_empty.png)

Once you have the viewer open, you can add images through the `File/Open` dropdown menu
or by dragging and dropping images directly on the viewer.
napari natively only supports files that can be read with [`skimage.io.imread`](https://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imread), such as `tif`, `png`, and `jpg`. For reading other file formats, visit [napari hub](https://www.napari-hub.org/).

You can also directly load an image into the viewer from the command line by passing the path to the image as an argument as follows

```sh
napari my_image.png
```

If the image is `RGB` or `RGBA` use the `-r` or `--rgb` flag.

![image: napari viewer displaying an image layer](../assets/tutorials/launch_cli_image.png)

Launching napari directly from the command line is the simplest and fastest way to open the viewer, but it doesn't allow you to preprocess your images before opening them.

### Python script usage

To launch napari from a python script, inside your script (my_example_script.py), import `napari` and then create the `Viewer` by adding some data.

For example, to add an image inside your script:

```python
import napari

# create a Viewer and add an image here
viewer = napari.view_image(my_image_data)

# start the event loop and show the viewer
napari.run()
```

then run your script from the command line to launch the viewer with your data:

```sh
python my_example_script.py
```

See the scripts inside the [`examples`](https://github.com/napari/napari/tree/master/examples) in the main repository for examples of using napari this way.

![image: napari launched from a python script](../assets/tutorials/launch_script.png)

An advantage of launching napari from a python script is that you can preprocess your images and add multiple layers before displaying the viewer.

### IPython console usage

To launch napari from an IPython console import `napari` and create a `Viewer` object.

```python
import napari
from skimage.data import astronaut

# create the viewer and display the image
viewer = napari.view_image(astronaut(), rgb=True)
```

napari will automatically use the interactive [`%gui qt` event
loop](https://ipython.readthedocs.io/en/stable/config/eventloops.html#integrating-with-gui-event-loops)
from IPython

![image: napari launched from ipython](../assets/tutorials/launch_ipython.png)

An advantage of launching napari from an IPython console is that you can continue to programmatically interact with the viewer from the IPython console, including bidirectional communication, where code run in the console will update the current viewer and where data changed in the GUI will be accessible in the console.

IPython console is also available within napari viewer.

### Jupyter notebook usage

You can also launch napari from a jupyter notebook, such as [`examples/notebook.ipynb`](https://github.com/napari/napari/tree/master/examples/notebook.ipynb)

![image: napari launched from a jupyter notebook](../assets/tutorials/launch_jupyter.png)

Similar to launching from the IPython console,an advantage of launching napari from a jupyter notebook is that you can continue to programmatically interact with the viewer from jupyter notebook, including bidirectional communication, where code run in the notebook will update the current viewer and where data changed in the GUI will be accessible in the notebook.