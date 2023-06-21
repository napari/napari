# Version management  

This guide covers the methods of updating your version string everywhere.

## This guide covers:  
* [Using git tags](#using-git-tags)
* [Using a local script to edit files](#using-a-local-script-to-edit-files)  
* [Manually](#manually)

Your goal is to make sure that you bump your version string everywhere it may appear, in unison, prior to publishing your package.  A version number can be in `init.py`, `setup.cfg`, etc.

In increasing order of work, but decreasing order of magic, the methods of bumping your version string are listed below. 

## Using git tags:  
You can use [setuptools_scm](https://github.com/pypa/setuptools_scm) to automatically generate version numbers for your package based on tagged commits.

   ```console  
   # configure in pyproject.toml, then…  
   $ git tag -a v0.1.0 -m v0.1.0
  ```

  The next time you run `python -m build`, either locally or in GitHub actions, your package version will be based on the latest git tag.

## Using a local script to edit files:  
One tool for doing this is [bump2version](https://github.com/c4urself/bump2version). For example:
```console
   $ pip install bump2version  
   # configure all the places you use your version, then, to update:
   $ bump2version --current-version 0.5.1 minor  
```   

## Manually
Updating the version number manually involves going through everywhere your version is declared and changing the version number before building your distribution. This is ***not*** recommended, you *will* eventually make mistakes and have mismatched version/metadata somewhere. In some cases this will lead to your build process failing, but it can fail silently too.
  
## Tips:
* The "best" versioning and deployment workflow is the one you will actually use!  
* Get comfortable with at least one workflow for versioning and deploying your package *otherwise, you won't do it.*

The next topic in this series is [Developer tools](./4-developer-tools.md).

## Other topics in this series:  

* [Virtual environments](./1-virtual-environments)  
* [Deploying your plugin](./2-deploying-your-plugin.md)    
* [Developer tools](./4-developer-tools.md)   
* [Survey](./5-survey.md) 
