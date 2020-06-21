# napari 0.2.1

We're happy to announce the release of napari 0.2.1! napari is a fast,
interactive, multi-dimensional image viewer for Python. It's designed for
browsing, annotating, and analyzing large multi-dimensional images. It's built
on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the
scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## New Features

- added a code of conduct (#374)
- added grid mode that allows for looking at layers side by side (#565)
- added the ability to bind custom mouse functions (#544)
- added a benchmarking suite (#573, #577)
- added help menu with about dialog and keybindings dialog (#580, #583, #591)

## Improvements

- improved performance of the thumbnail generation for points layers (#564)
- more comprehensive testing of keybindings functionality (#583)
- now use square brackets to indicate auto-incremented names (#589)
- unify file IO and add dask lazy loading of folders (#590)

## API Changes

- added the ability to pass viewer keyword arguments to our `view_*` methods (#584)
- Allow add_points() to be called without an argument to create empty points layer (#594)

## Bug Fixes

- fixed some pyramid data types (#585)
- fix points updating (#571)
- stop vispy error catching (#551)

## 4 authors added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn

## 4 reviewers added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
