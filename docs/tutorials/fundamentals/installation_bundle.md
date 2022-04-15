# How to install napari as a bundled app

napari can be installed as a bundled app on [MacOS](#how-to-install-the-macos-bundle), [Windows](#how-to-install-the-windows-bundle), and [Linux](#how-to-install-the-linux-bundle) with a simple one click download and guided installation process. This installation method is best if you mainly want to use napari as a standalone GUI app. However, certain plugins may not be supported.

```{note}
If you want to use napari from Python to programmatically interact with the app, please follow the [Python package installation guide](installation_python.md). This installation method is recommended to take full advantage of napari’s features and to access additional plugins. 
```

```{note} 
If you want to contribute code back into napari, please follow the [development installation instructions in the contributing guide](https://napari.org/developers/contributing.html).
```

To start, visit the [latest napari release page](https://github.com/napari/napari/releases/latest) and go to the ‘Assets’ tab and download the file that corresponds to your operating system. For MacOS users, download the file that corresponds to your processor (This can be checked by going to Apple menu > About This Mac. For Intel processors, download the x86 file, and for ARM processors, download the arm64 file.). Below are the installation guides for each operating system.

![image: expanded assets tab on the napari release page](.../docs/images/bundle_02.png)

```{note} 
If you are interested in an earlier version of napari, you may access those files by scrolling below the latest release on the [napari release page](https://github.com/napari/napari/releases). The instructions below will work for napari versions 0.4.15 and above.
```

## Prerequisites

This installation method does not have any prerequisites. 

### How to Install the MacOS bundle

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
The first time you open napari you must use the Launchpad, but subsequently, the napari application should show up in Spotlight search.
```

napari comes installed with sample images from scikit-image. Use the dropdown menu File > Open Sample > napari to open a sample image, or open one of your own images using File > Open or dragging and dropping your image onto the canvas. 

Next check out our [tutorial on the viewer](https://napari.org/tutorials/fundamentals/viewer.html) or explore any of the pages under the [Usage tab](https://napari.org/usage.html).

### How to Install the Windows bundle

Once you have downloaded the Windows executable file, you will have a file with a name like `napari-0.4.15-Windows-x86_64.exe`. Double click this file to open the napari Setup Wizard. Click "Next" to continue.

![image: napari Setup Wizard start page](.../docs/images/bundle_17.png)

To continue, read and agree to the License Agreement by clicking ‘I Agree’.
 
![image: napari License Agreement](.../docs/images/bundle_18.png)

The recommended installation method is to install napari just for the current user. 

![image: napari Setup Wizard user installation options](.../docs/images/bundle_19.png)

Next you will be shown how much space will be used by the installation and the default destination folder, which can be updated using the ‘Browse’ button. Click ‘Next’ to continue.

![image: napari Setup Wizard installation location](.../docs/images/bundle_20.png)

On the next page, click ‘Install’ to start the installation process. Installation progress can be monitored on the following page.

![image: napari Setup Wizard installation progress bar](.../docs/images/bundle_22.png)

Once installation is complete, you will see the page below. Click ‘Finish’ to close the installation wizard.

![image: napari Setup Wizard installation completed](.../docs/images/bundle_24.png)

You can now get started using napari! A shortcut to launch napari can be found in the Windows Start menu. 

napari comes installed with sample images from scikit-image. Use the dropdown menu File>Open Sample>napari to open a sample image, or open one of your own images using File > Open or dragging and dropping your image onto the canvas. 

Next check out our [tutorial on the viewer](https://napari.org/tutorials/fundamentals/viewer.html) or explore any of the pages under the [Usage tab](https://napari.org/usage.html).

### How to Install the Linux bundle

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
