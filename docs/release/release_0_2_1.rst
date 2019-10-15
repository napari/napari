Announcement: napari 0.2.1
==========================

We're happy to announce the release of napari v0.2.1!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-base
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

New Features
------------
- added a code of conduct
- added grid mode that allows for looking at layers side by side
- added the ability to bind custom mouse functions
- added a benchmarking suite
- added help menu with about dialog and keybindings dialog


Improvements
------------
- improved performance of the thumbnail generation for points layers
- more comprehensive testing of keybindings functionality
- now use square brackets to indicate auto-incremented names
- unify file IO and add dask lazy loading of folders


API Changes
-----------
- added the ability to pass viewer keyword arugments to our `view_*` methods


Bugfixes
--------
- fixed some pyramid data types


Deprecations
------------
- none


Pull Requests
*************
- add code of conduct (#374)
- implement custom mouse function framework (#544)
- stop vispy error catching (#551)
- faster points thumbnails (#564)
- basic grid vs stack views (#565)
- fix points updateing (#571)
- add basic benchmarks (#573)
- add layer model benchmarks (#577)
- WIP: About Menubar (#580)
- Test and display keybindings (#583)
- viewer args to view_* methods (#584)
- fix pyramid guessing (#585)
- bracketed numbers for autoincrementing (#589)
- Further unify IO and use dask for multiple files (#590)
- Bug Report Issue Template Update (#591)

4 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- Ahmet Can Solak
- Juan Nunez-Iglesias
- Kira Evans
- Nicholas Sofroniew


4 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Ahmet Can Solak
- Juan Nunez-Iglesias
- Kira Evans
- Nicholas Sofroniew
