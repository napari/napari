---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

(installation)=

# How to install napari on your machine

Welcome to the **napari** installation guide!

This guide will teach you how to do a clean install of **napari** and launch the viewer.

```{note} 
If you want to contribute code back into napari, you should follow the [development installation instructions in the contributing guide](https://napari.org/developers/contributing.html) instead.
```

## Prerequisites

Prerequisites differ depending on how you want to install napari.

### Prerequisites for installing napari as a Python package 
This installation method allows you to use napari from Python to programmatically 
interact with the app. It is the best way to install napari and make full use of
all its features.

It requires:
- [Python 3.8 or higher](https://www.python.org/downloads/)
- the ability to install python packages via [pip](https://pypi.org/project/pip/) OR [conda-forge](https://conda-forge.org/docs/user/introduction.html)

You may also want:
- an environment manager like [conda](https://docs.conda.io/en/latest/miniconda.html) or
[venv](https://docs.python.org/3/library/venv.html) **(Highly recommended)**

### Prerequisites for installing napari as a bundled app
This is the easiest way to install napari if you only wish to use it as a standalone GUI app.
This installation method does not have any prerequisites. 

```{important}
Note that the bundled app is still
in active development, and may not be very stable. We strongly recommend 
[installing as a Python package instead](#install-as-python-package-recommended).
```

[Click here](#install-as-a-bundled-app) to see instructions
for installing the bundled app.

## Install as Python package (recommended)

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

napari can be installed on most macOS, Linux, and Windows systems with Python
3.7, 3.8, and 3.9 using pip:

```sh
python -m pip install "napari[all]"
```
You can then upgrade napari to a new version using

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

You can then upgrade to a new version of napari using 

```sh
conda update napari
```

If you want to install napari with PySide2 as the backend you need to install it using

```sh
conda install -c conda-forge "napari=*=*pyside2"
```  

````


````{admonition} **3. From the main branch on Github**
:class: dropdown

To install the latest version with yet to be released features from github via pip, call

```sh
python -m pip install "git+https://github.com/napari/napari.git#egg=napari[all]"
```
````

<!-- #region -->
## Checking it worked

After installation you should be able to launch napari from the command line by
simply running

```sh
napari
```

An empty napari viewer should appear as follows.

![image: An empty napari viewer ](../assets/tutorials/launch_cli_empty.png)

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

## Install as a bundled app

napari can also be installed as a bundled app on each of the major platforms,
MacOS, Windows, and Linux with a simple one click download and installation
process. You might want to install napari as a bundled app if you are unfamiliar
with installing Python packages or if you were unable to get the installation
process described above working. The bundled app version of napari is the same
version that you can get through the above described processes, and can still be
extended with napari plugins installed directly via the app.

To access the cross platform bundles you can visit our [release
page](https://github.com/napari/napari/releases) and scroll to the release you
are interested in and expand the `assets` tab to get a view that looks like
this:

![image: release bundle
assets ](../assets/tutorials/installation/bundle_assets.png)

You can then download the appropriate zip file for your platform.

### Installing the MacOS bundle

Once you have downloaded the MacOS bundle zip you will have a zip file with a
name like `napari-0.3.7-macOS.zip`. After unzipping you will have a file with a
name like `napari-0.3.7.dmg`. Double clicking the `dmg` will open a new finder
window giving you the option to install napari into your `Applications` folder
by dragging the `napari` icon onto the `Applications` icon.

![image: MacOS install to
Applications ](../assets/tutorials/installation/bundle_install.png)

After you drag the `napari` icon onto the `Applications` icon napari will be
installed in your Applications folder and available for opening by double
clicking on it.

![image: MacOS Applications folder with
napari ](../assets/tutorials/installation/bundle_applications_folder.png)

The first time you try and open napari you will get the "unverified developer
warning" that appears below:

![image: MacOS developer
warning ](../assets/tutorials/installation/bundle_dev_warning.png)

Don't be alarmed, this is because the napari team has not yet obtained a
developer certificate from Apple, but we will be doing this promptly. It is safe
to click "Cancel". To get napari to open you must enter "Systems Preferences"
and click the "Security & Privacy" icon, circled in red below:

![image: MacOS Security & Privacy
settings ](../assets/tutorials/installation/bundle_security_privacy.png)

Once inside the "Security & Privacy" tab, you should see a message about napari
being blocked and an "Open Anyway" button, circled in red below:

![image: MacOS "Open Anyway"
settings ](../assets/tutorials/installation/bundle_open_anyway.png)

After clicking this button you'll get one final warning about napari not being
verified, but now you'll have an "Open" button, as seen below:

![image: MacOS open bundle dialog
box ](../assets/tutorials/installation/bundle_open.png)

After clicking "Open", the viewer should appear. Don't worry, you only have to
go through this process once when you install a new bundle.

### Installing the Windows bundle

Once you have downloaded the Windows bundle zip you will have a zip file with a
name like `napari-0.3.7-Windows.zip`. Unzip the bundle (you may like to use a
tool like [7-zip](https://www.7-zip.org/) for this) and double click on msi
file, eg: `napari-0.3.7.msi`

The napari setup wizard will then open. Click "Next" to begin the installation.

![image: Windows napari setup
wizard ](../assets/tutorials/installation/windows_bundle_installer_start.png)

![image: Windows napari setup wizard in
progress ](../assets/tutorials/installation/windows_bundle_installer_progress.png)

After the setup wizard has installed napari, click "Finish" to exit.

![image: Windows napari setup wizard
finished ](../assets/tutorials/installation/windows_bundle_installer_finish.png)

When you launch the bundled napari app on Windows, first you'll see an empty
command terminal appear (do not close this window - you can ignore it). The
command terminal will be followed by a napari splash screen, and then the main
napari user interface window will appear a few seconds later. Note that errors
which might occur during your napari session will appear on the command terminal - 
so if something is not working, it might pay to take a look at the terminal!

![image: Windows command terminal on napari
launch ](../assets/tutorials/installation/windows_bundle_command_terminal.png)

You can launch napari from the Windows start menu.

![image: Windows launch napari from start
menu ](../assets/tutorials/installation/windows_launch_napari.png)

The very first time you launch napari the startup time will be fairly slow, but
after that you will find that napari launches more quickly.

### Installing the Linux bundle

(Guide coming soon... In the meantime, if you try it and encounter issues, see
below for how to contact us.)
<!-- #endregion -->

## Next steps

- to start learning how to use napari, checkout our [getting
started](./getting_started) tutorial
- if you are interested in
contributing to napari please check our [contributing
guidelines](../../developers/contributing.md)
- if you are running into issues or bugs, please open a new issue on our [issue
tracker](https://github.com/napari/napari/issues)
    - include the output of `napari -info` 
    (or go to `Help>Info` in the viewer and copy paste the information)
- if you want help using napari, we are a community partner on the [imagesc
forum](https://forum.image.sc/tags/napari) and all usage support requests should
be posted on the forum with the tag `napari`. We look forward to interacting
with you there!
