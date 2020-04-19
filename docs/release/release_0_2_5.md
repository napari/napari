# napari 0.2.5

We're happy to announce the release of napari 0.2.5! napari is a fast,
interactive, multi-dimensional image viewer for Python. It's designed for
browsing, annotating, and analyzing large multi-dimensional images. It's built
on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the
scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## New Features

- Basic "play" functionality (animate a dimension) (#607)

## Improvements

- Add linter pre-commit hook  (#638)
- Modify dims QScrollbar with scroll-to-click behavior (#664)

## Bug Fixes

- Fix np.pad() usage (#649)
- Bump vispy, remove unnecessary data.copy() (#657)
- Update Cirrus OSX image and patch windows builds (#658)
- Avoid numcodecs 0.6.4 for now (#666)

## 4 authors added to this release (alphabetical)

- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03

## 5 reviewers added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Bill Little](https://github.com/napari/napari/commits?author=bjlittle) - @bjlittle
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
