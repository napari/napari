# napari 0.2.3

We're happy to announce the release of napari 0.2.3! napari is a fast,
interactive, multi-dimensional image viewer for Python. It's designed for
browsing, annotating, and analyzing large multi-dimensional images. It's built
on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the
scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## New Features

- add loading from paths for images (#601)

## Improvements

- Add the turbo colormap (#599)
- Move markdown files to docs folder (#602)
- Standardize usage of cmaps and add cmap tests (#622)

## Bug Fixes

- Add numpydoc to default dependencies (#598)
- fix dimensions change when no layers (#603)
- fix bracket highlighting (#606)
- fix for data overwriting during 3D rendering of float32 array (#613)
- fix `io.magic_imread()` in `__main__.py` (#626)
- Include LICENSE file in source distribution (#628)

## 5 authors added to this release (alphabetical)

- [Hector](https://github.com/napari/napari/commits?author=hectormz) - @hectormz
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Will Connell](https://github.com/napari/napari/commits?author=wconnell) - @wconnell

## 4 reviewers added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
