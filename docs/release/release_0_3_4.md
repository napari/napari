# napari 0.3.4

We're happy to announce the release of napari 0.3.4!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).

This is a short release that refactors our installation process to allow more
flexibility around which Qt python bindings users install (PySide2, PyQt5).
Starting with this release, running `pip install napari` will *no longer*
install a GUI backend by default. For a complete installation with a GUI
backend, users are now encouraged to use `pip install napari[all]`, which
will install the default backend (currently PyQt5).  To explicitly select
a backend, users may run either `pip install napari[pyqt5]` or
`pip install napari[pyside2]`.

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Build Tools

- Add check-manifest to CI and release workflow (#1318)
- Packaging and setup.py refactor (#1324)

## 1 author added to this release (alphabetical)

- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03

## 5 reviewers added to this release (alphabetical)

- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
