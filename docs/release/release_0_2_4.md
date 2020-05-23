# napari 0.2.4

We're happy to announce the release of napari 0.2.4! napari is a fast,
interactive, multi-dimensional image viewer for Python. It's designed for
browsing, annotating, and analyzing large multi-dimensional images. It's built
on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the
scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## New Features

- Add gamma slider (#610)

## Improvements

- Make `_on_data_change` only get called once when ndisplay changes (#629)
- fix undefined variables, remove unused imports (#633)
- Raise informative warning when no Qt loop is detected (#642)

## Bug Fixes

- gui_qt: Ipython command is `%gui qt` in docs (#636)
- Calculate minimum thumbnail side length so they are never zero width (#643)

## Deprecations

- napari.Viewer.add_multichannel was removed. Use `napari.Viewer.add_image(...,
  channel_axis=num)`   (#619)

## 4 authors added to this release (alphabetical)

- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Simon Li](https://github.com/napari/napari/commits?author=manics) - @manics
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03

## 4 reviewers added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Loic Royer](https://github.com/napari/napari/commits?author=royerloic) - @royerloic
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
