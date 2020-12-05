# napari 0.4.2

We're happy to announce the release of napari 0.4.2!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights

This is an emergency patch release to fix a regression in `0.4.1` that broke
magicgui generated dockwidgets which accepted layers as input (#1962). The
release also contains a number of other bug fixes and improvements, notably
a fix to a performance regression when adding many layers that came in the
`0.3.6` release (#1945), and the beginning of adopting evented dataclasses
for our model files, which will result in a dramatic simplification of that
part of the codebase.

## Improvements
- async-28: Misc Cleanup (#1900)
- async-29: Shared Memory Server (#1909)
- Use evented dataclass for axes (#1910)
- Use evented dataclass for scalebar (#1911)
- Use evented dataclass for cursor (#1912)
- Use evented dataclass for grid (#1913)
- Use evented dataclass for camera (#1914)
- Use evented dataclass for colormap (#1916)
- Remove add layers mixin (#1921)

## Bug Fixes
- Fix performance issues 1 - adding layers (#1945)
- Fix track labels lookup (#1946)
- Update sliders when change scale of layer (#1951)
- Fix evented dataclass for python 3.9 (#1958)
- Refine TypedMutableSequence.__getitem__ error type, add magicgui tests (#1962)

## API Changes
- ``Viewer.camera.ndisplay`` has been dropped. Instead, use
  ``Viewer.dims.ndisplay``. (#1914)

## Deprecations
- All existing deprecations have been bumped by one release (#1963)

## Build Tools and Docs
- Fix checks for running under cProfile and yappi (#1924)
- Fix missing vispy.ext.six (#1930)
- Replace calls to layer.shape in tests (#1938)
- Add pims to bundle (#1939)
- Register sync_only pytest mark to fix warning (#1941)
- Don't run app bundling in forked repos (#1953)
- Add tests for #1895 (#1961)


## 8 authors added to this release (alphabetical)

- [Alan R Lowe](https://github.com/napari/napari/commits?author=quantumjot) - @quantumjot
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [kir0ul](https://github.com/napari/napari/commits?author=kir0ul) - @kir0ul
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03


## 5 reviewers added to this release (alphabetical)

- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03

