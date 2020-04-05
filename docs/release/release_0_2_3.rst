Announcement: napari 0.2.3
==========================

We're happy to announce the release of napari v0.2.3!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-base
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

New Features
------------
- add loading from paths for images (#601)

Improvements
------------
- Add the turbo colormap (#599)
- Move markdown files to docs folder (#602)
- Standardize usage of cmaps and add cmap tests (#622)

API Changes
-----------
- None

Bugfixes
--------
- Add numpydoc to default dependencies (#598)
- fix dimensions change when no layers (#603)
- fix bracket highlighting (#606)
- fix for data overwriting during 3D rendering of float32 array (#613)
- fix ``io.magic_imread()`` in ``__main__.py`` (#626)
- Include LICENSE file in source distribution (#628)

Deprecations
------------
- None

5 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- Hector
- Juan Nunez-Iglesias
- Nicholas Sofroniew
- Talley Lambert
- Will Connell


4 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Ahmet Can Solak
- Juan Nunez-Iglesias
- Nicholas Sofroniew
- Talley Lambert
