# Virtual environments  

This guide explains the value of using virtual environments and how to create and remove them. 

## This guide covers: 
* [The importance of virtual environments](#overview)
* [Creating environments](#creating-environments)  
* [Removing environments](#removing-environments)

## Overview
A virtual environment is an isolated collection of packages, settings, and an associated Python interpreter, that allows multiple different collections to exist on the same system. They are created on top of an existing Python installation, known as the virtual environment's “base” python, and may optionally be isolated from the packages in the base environment, so only those explicitly installed in the virtual environment are available.

 More information on why virtual environments are created and how they can help you can be found on the [python website](https://docs.python.org/3/library/venv.html#creating-virtual-environments) and in [this introductory workshop](https://hackmd.io/@talley/SJB_lObBi#What-is-a-virtual-environment). 

Virtual environments are super important! They allow you to isolate your project from other Python projects. They allow you to experiment with various packages and versions without fear of breaking your entire system (and needing to reinstall everything). As you install packages over time, you will inevitably install something that doesn’t “play well” with something else that is already installed. In some cases this can be hard to recover from. With virtual environments, you can just create a fresh environment and start again – without needing to do major surgery on your system.

There are several tools available for creating and managing virtual environments.  One of the most popular, comprehensive tools is Conda. Wikipedia explains that [Conda is an open-source, cross-platform, language-agnostic package manager and environment management system.](https://en.wikipedia.org/wiki/Conda_(package_manager)) It was originally developed to solve difficult package management challenges faced by Python data scientists. The Conda package and environment manager is included in all versions of **Anaconda**, **Miniconda**, and **Anaconda Repository**. 

## Install and Config
Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [mini forge](https://github.com/conda-forge/miniforge) (comes pre-configured with `conda-forge`) in the home directory.

Adding the `conda-forge` channel to the conda config makes packages in `conda-forge` visible to the conda installer. Setting `channel_priority` to strict ensures packages in high priority channels are always installed over packages of the same name in lower priority channels. See [this guide](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html) for more details. 

Make sure the `conda-forge` channel is in your config by using the following commands:
```console
    $conda config --add channels conda-forge  
    $conda config --set channel_priority strict  
```

**Note:** The default anaconda channel has some very outdated packages, e.g. [old version of Qt](https://forum.image.sc/t/napari-issues-on-bigsur/52630/10).   

## Creating environments  
  
Create environments liberally!  

To create an environment, use the following commands at the command prompt (terminal):

```console
    $ conda create -n name-of-env python
    $ conda activate name-of-env
    $ pip/conda install <whatever> 
```

Virtual environments are made to be ephemeral.  

## Removing environments
Consider your environment to be disposable.
If you are ever having weird problems, nuke your environment and start over using the following commands:  

```console
    $ conda activate base  
    $ conda remove -n name-of-env --all -y
    $ conda create -n name-of-env python
    $ conda activate name-of-env
    $ pip/conda install <whatever>
```  
  
Encourage your users to do the same. You can waste a lot of time trying to debug something that someone unknowingly did when installing a variety of things into their environment. If they can provide a repeatable example (starting from environment creation), then it's worth debugging.

## Other topics in this series:  

* [Deploying your plugin](./2-deploying-your-plugin.md)  
* [Version management](./3-version-management.md)     
* [Developer tools](./4-developer-tools.md)   
* [Survey/Q&A](./5-survey.md)   

The next topic in this series is [Deploying your plugin](./2-deploying-your-plugin.md). 
