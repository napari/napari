# napari 0.2.10

We're happy to announce the release of napari 0.2.10! napari is a fast,
interactive, multi-dimensional image viewer for Python. It's designed for
browsing, annotating, and analyzing large multi-dimensional images. It's built
on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the
scientific Python stack (numpy, scipy).

This is a bug fix release to address issues that snuck through in 0.2.9.

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Bug Fixes

- remove calls to QPoint.toTuple which is invalid in PyQt5 (#866)
- fix aspect ratio press (#871)
- fix None label (#872)
- drop zarr and numcodecs dependency (#873)

## 2 authors added to this release (alphabetical)

- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn

## 2 reviewers added to this release (alphabetical)

- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
