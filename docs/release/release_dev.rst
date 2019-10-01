Announcement: napari 0.X.0
================================

We're happy to announce the release of napari v0.X.0!

napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-base
rendering), and the scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:

https://github.com/napari/napari


New Features
------------
- Add `viewer.add_multichannel` method to rapidly add expand a multichannel
array along one particular axis with different colormaps (#528).
- Add a `Surface` layer to render already generated meshes. Support nD meshes
rendered in 2D or 3D (#503).

Improvements
------------



API Changes
-----------



Bugfixes
--------



Deprecations
------------
- Drop `napari.view` method. Replaced with `napari.view_*` methods in (#542)


Contributors to this release
----------------------------
