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

### Prerequisites for installing napari as a bundled app
This is the easiest way to install napari if you only wish to use it as a standalone GUI app.
This installation method does not have any prerequisites. 

[Click here](#install-as-a-bundled-app) to see instructions
for installing the bundled app.

### Prerequisites for installing napari as a Python package 
This installation method allows you to use napari from Python to programmatically 
interact with the app. It is the best way to install napari and make full use of
all its features.

It requires:
- [Python 3.7 or higher](https://www.python.org/downloads/)
- the ability to install python packages via [pip](https://pypi.org/project/pip/) OR [conda-forge](https://conda-forge.org/docs/user/introduction.html)

You may also want:
- an environment manager like [conda](https://docs.conda.io/en/latest/miniconda.html) or
[venv](https://docs.python.org/3/library/venv.html) **(Highly recommended)**

[Click here](#install-as-a-python-package) to see instructions
for installing the napari as a python package.

## Install as a Python package (recommended)

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

![image: An empty napari viewer](../assets/tutorials/launch_cli_empty.png)

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

napari can be installed as a bundled app on MacOS, Windows, and Linux with a simple one click download and guided installation process. This installation method is best if you mainly want to use napari as a standalone GUI app. However, certain plugins may not be supported.

```{note}
If you want to use napari from Python to programmatically interact with the app, please follow the [Python package installation guide](installation_python.md). This installation method is recommended to take full advantage of napari’s features and to access additional plugins. 
```

```{note} 
If you want to contribute code back into napari, please follow the [development installation instructions in the contributing guide](https://napari.org/developers/contributing.html).
```

To start, visit the [napari release page](https://github.com/napari/napari/releases) and find the release tagged with “latest.” Within the release details, expand the ‘assets’ tab and download the file that corresponds to your operating system, and for MacOS users, download the file that corresponds to your processor (This can be checked by going to Apple menu > About This Mac. For Intel processors, download the x86 file, and for ARM processors, download the arm64 file.). Below are the installation guides for each operating system.

![image: expanded assets tab on the napari release page](.../docs/images/bundle_02.png)

```{note} 
If you are interested in an earlier version of napari, you may access those files by scrolling below the latest release on the [napari release page](https://github.com/napari/napari/releases). The instructions below will work for napari versions 0.4.15 and above.
```

## Prerequisites

This installation method does not have any prerequisites. 

### Installing the MacOS bundle

Once you have downloaded the appropriate MacOS package file, you will have a file with a name like ‘napari-0.4.15-macOS-x86_64.pkg’. Double click this file to open the installer.

![image: expanded assets tab on the napari release page](.../docs/images/bundle_04.png)

Click ‘Continue’ to open the Software License Agreement.

![image: napari Software License Agreement verbage](.../docs/images/bundle_06.png)

After reading this agreement, click ‘Continue’ to be prompted to agree to the Software License Agreement in order to proceed with installation.

![image: Prompt to agree to napari Software License Agreement](.../docs/images/bundle_07.png)

On the following page, you will be shown how much space the installation will use and can begin the standard installation by clicking ‘Install.’

![image: napari installer space requirement](.../docs/images/bundle_09.png)

However, if you would like to change the install location, you may specify a different location by clicking ‘Change Install Location…’ and following the subsequent prompts before starting the installation.

The installation progress can be monitored on the following window.

![image: napari installer progress monitoring page](.../docs/images/bundle_10.png)

If installation is successful, you will see the window shown below and you may now close the installation wizard and move it to trash.

![image: napari installer success page](.../docs/images/bundle_11.png)

You can now get started using napari! Use Launchpad to open the application. 

![image: napari icon in MacOS laundpad](.../docs/images/bundle_13.png)

```{note} 
The first time you open napari you must use the Launchpad, but subsequently, the napari application will show up in Spotlight search.
```

napari comes installed with sample images from scikit-image. Use the dropdown menu File > Open Sample > napari to open a sample image, or open one of your own images using File > Open or dragging and dropping your image onto the canvas. 

Next check out our [tutorial on the viewer](https://napari.org/tutorials/fundamentals/viewer.html) or explore any of the pages under the [Usage tab](https://napari.org/usage.html).

### Installing the Windows bundle

Once you have downloaded the Windows executable file, you will have a file with a name like `napari-0.4.15-Windows-x86_64.exe`. Double click this file to open the napari Setup Wizard. Click "Next" to continue.

![image: napari Setup Wizard start page](.../docs/images/bundle_17.png)

To continue, read and agree to the License Agreement by clicking ‘I Agree’.
 
![image: napari License Agreement](.../docs/images/bundle_18.png)

The recommended installation method is to install napari just for the current user. However, you may install for all users using administrator privileges.

![image: napari Setup Wizard user installation options](.../docs/images/bundle_19.png)

Next you will be shown how much space will be used by the installation and the default destination folder, which can be updated using the ‘Browse’ button. Click ‘Next’ to continue.

![image: napari Setup Wizard installation location](.../docs/images/bundle_20.png)

On the next page, we recommend you check ‘Clear the package cache upon completion’ since this frees up memory in your machine’s cache without compromising napari functionality following installation. Click ‘Install’ to start the installation process.

![image: napari Setup Wizard clear package cache prompt](.../docs/images/bundle_21.png)

Installation progress can be monitored on the following page.

![image: napari Setup Wizard installation progress bar](.../docs/images/bundle_22.png)

Once installation is complete, you will see the page below. Click ‘Finish’ to close the installation wizard.

![image: napari Setup Wizard installation completed](.../docs/images/bundle_24.png)

You can now get started using napari! A shortcut to launch napari can be found in the Windows Start menu. 

napari comes installed with sample images from scikit-image. Use the dropdown menu File>Open Sample>napari to open a sample image, or open one of your own images using File > Open or dragging and dropping your image onto the canvas. 

Next check out our [tutorial on the viewer](https://napari.org/tutorials/fundamentals/viewer.html) or explore any of the pages under the [Usage tab](https://napari.org/usage.html).

### Installing the Linux bundle

Once you have downloaded the Linux SH file, you will have a file with a name like `napari-0.4.15-Linux-x86_64.sh`. Double click this file to open the command in terminal or open terminal and run the command ‘bash [file name]’.

![image: linux file command in terminal](.../docs/images/bundle_28.png)

Press Enter to open the License Agreement.

![image: napari License Agreement](.../docs/images/bundle_29.png)

Read through the agreement shown below. You must agree to the terms by entering ‘yes’ to continue.

![image: napari License Agreement verbage](.../docs/images/bundle_30.png)

![image: napari License Agreement verbage continued](.../docs/images/bundle_31.png)

Next you will be shown the default location for the installation. You may confirm this location by hitting ENTER or specify a different location by writing out the filetree, which will begin the installation process. 

![image: napari License Agreement agreement prompt](.../docs/images/bundle_32.png)

If installation is successful, you will see ‘installation finished.’ in terminal.

![image: napari installation success notification](.../docs/images/bundle_33.png)

You can now get started using napari! A shortcut to launch napari should appear on your desktop or you can search for napari with the desktop searchbar.

![image: napari icon on desktop](.../docs/images/bundle_34.png)

![image: napari shortcut in searchbar](.../docs/images/bundle_35.png)

napari comes installed with sample images from scikit-image. Use the dropdown menu File>Open Sample>napari to open a sample image, or open one of your own images using File > Open or dragging and dropping your image onto the canvas. 

Next check out our [tutorial on the viewer](https://napari.org/tutorials/fundamentals/viewer.html) or explore any of the pages under the [Usage tab](https://napari.org/usage.html).

<!-- #endregion -->

## Next steps

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
