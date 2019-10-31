Announcement: napari 0.2.4
==========================

We're happy to announce the release of napari v0.2.4!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-base
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

New Features
------------
- Add gamma slider (#610)

Improvements
------------
- Make _on_data_change only get called once when ndisplay changes (#629)
- fix undefined variables, remove unused imports (#633)
- Raise informative warning when no Qt loop is detected (#642)

API Changes
-----------

Bugfixes
--------
- gui_qt: Ipython command is "%gui qt" in docs (#636)
- Calculate minimum thumbnail side length so they are never zero width (#643)

Deprecations
------------
- drop add_multichannel, now add_image has `channel_axis` keyword arg (#619)


4 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- Juan Nunez-Iglesias
- Nicholas Sofroniew
- Simon Li
- Talley Lambert


4 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Ahmet Can Solak
- Juan Nunez-Iglesias
- Loic Royer
- Nicholas Sofroniew
