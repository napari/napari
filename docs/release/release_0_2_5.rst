Announcement: napari 0.2.5
==========================

We're happy to announce the release of napari v0.2.5!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-base
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

New Features
------------
- Basic "play" functionality (animate a dimension) (#607)


Improvements
------------
- Add linter pre-commit hook  (#638)
- modify dims QScrollbar with scroll-to-click behavior (#664)


API Changes
-----------
- None


Bugfixes
--------
- Fix np.pad() usage (#649)
- bump vispy, remove unnecessary data.copy() (#657)
- Update Cirrus OSX image and patch windows builds (#658)
- Avoid numcodecs 0.6.4 for now (#666)


Deprecations
------------
- None

4 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- Juan Nunez-Iglesias
- Kevin Yamauchi
- Nicholas Sofroniew
- Talley Lambert


5 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Ahmet Can Solak
- Bill Little
- Juan Nunez-Iglesias
- Nicholas Sofroniew
- Talley Lambert
