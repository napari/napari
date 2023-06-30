# Developer tools

This guide explains the different types of tools that can help you develop and test your code.  

## This guide covers:   
* [General tools](#general-tools)
    - [Linting tools](#linting-tools)  
    - [Formatting tools](#formatting-tools)  
    - [Pre-commit tools](#pre-commit-tools)  
* [napari plugin-specific tools](#napari-plugin-specific-tools)
  
## General tools  
All of these are *optional*.  
Many are very helpful, but they do take a little time to learn. The more time you spend coding, the greater the return-on-investment for using them. It's a personal decision on whether the time saved by using these outweighs the time required to understand the tools.

### Linting tools   
These _check_ your code.  
* [flake8](https://flake8.pycqa.org/) - checks various code style conventions, unused variables, line spacings, etc…  
* [mypy](https://github.com/python/mypy)  
    - Static type checker: enforces proper usage of types.  
    - Super useful once you get the hang of it, but definitely an intermediate-advanced tool.  
    - Along with high test coverage, probably the best time-saver and project robustness tool.    

### Formatting tools 
These _auto-modify_ your code.  
* [black](https://github.com/psf/black)  
  Forces code to follow specific style, indentations, etc...  
* [autoflake](https://github.com/myint/autoflake)  
  Auto-fixes some flake8 failures.  
* [isort](https://github.com/PyCQA/isort)  
  Auto-sorts and formats your imports.
* [setup-cfg-fmt](https://github.com/asottile/setup-cfg-fmt)  
  Sorts and enforces conventions in setup.cfg.  

### Pre-commit tools
* [pre-commit](https://pre-commit.com/), runs all your checks each time you run git commit, preventing bad code from ever getting checked in.  
```console  
     $ pip install pre-commit
     # install the pre-commit "hook"  
     $ pre-commit install  
     # then configure in .pre-commit-config.yaml  
     # (optionally) Run hooks on demand  
     $ pre-commit run --all-files  
```  

* [pre-commit-ci](https://pre-commit.ci/)
    - Runs all your pre-commit hooks on CI (Continuous Integration).
    - Useful even if contributors don't install and run your pre-commit hooks locally before they open a PR.  
  
## Napari plugin-specific tools  

* [Static plugin checks](https://github.com/tlambert03/napari-plugin-checks)
    - This is a *pre-commit hook*. It is intended to be added to your 
    `.pre-commit-config.yaml` file.
    - It *statically* (without importing) checks various best practices about your plugin:  
```yaml  
    repo: https://github.com/tlambert03/napari-plugin-action  
    rev: v0.2.0  
    hooks: id: napari-plugin-checks  
```     

* [Plugin check GitHub action](https://github.com/tlambert03/napari-plugin-action)  (work in progress)  
    - It is intended to be added to your GitHub workflow.
    - It (currently) checks that your plugin is installable, and performs a few sanity checks about Qt backends and dock widgets.  
```yaml     
     uses: tlambert03/napari-plugin-action@main  
     with: package_name:  <your-package-name>  
```

The next topic in this series is the [Survey/Q&A](./5-survey.md). 

## Other topics in this series:  
* [Virtual environments](./1-virtual-environments)  
* [Deploying your plugin](./2-deploying-your-plugin.md)    
* [Version management](./3-version-management.md)   
* [Survey/Q&A](./5-survey.md) 
