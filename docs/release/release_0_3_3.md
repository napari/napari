# napari 0.3.3

We're happy to announce the release of napari 0.3.3!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


This is a small bug fix PR that pins our Qt version at < 0.5.15, due to
incompatibilities with their latest release until we fix them. See #1312 for
discussion and the latest progress.

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## New Features
- Adding properties attribute to Labels layers (#1281)

## Improvements
- Make Qt window public (#1306)


## Bug Fixes
- Exit context before return on `_repr_png` (#1298)

## Build Tools
- Pin PySide2 and PyQt5 at <5.15 (#1316)


## 2 authors added to this release (alphabetical)

- [DragaDoncila](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03


## 3 reviewers added to this release (alphabetical)

- [DragaDoncila](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
