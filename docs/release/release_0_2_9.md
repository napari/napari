# napari 0.2.9

We're happy to announce the release of napari 0.2.9! napari is a fast,
interactive, multi-dimensional image viewer for Python. It's designed for
browsing, annotating, and analyzing large multi-dimensional images. It's built
on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the
scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights

- better support for surface timeseries (#831)
- contrast limits slider popup on right click (#837)
- better isosurface rendering with colormaps (#840)
- attenuated MIP mode for better 3D rendering (#846)

## New Features

- convert layer properties to dictionary (#686)
- better support for surface timeseries (#831)
- make `contrast_limits_range` public and climSlider popup on right click (#837)
- attenuated MIP mode for better 3D rendering (#846)

## Improvements

- bump numpydoc dependency to 0_9_2 for faster startup (#830)
- better isosurface rendering with colormaps (#840)
- add nearest interpolation mode to volume rendering for better labels support (#841)
- refactor RangeSlider to accept data range and values. (#844)
- in bindings logic, check if generator, not generator function (#853)

## Bug Fixes

- fix fullscreen crash for test_viewer (#849)
- fix RangeSlider.rangeChange emit type bug (#856)

## API Changes

- `edge_color` and `face_color` now refer to colors of all points and shapes
  in layer, `current_edge_color` and `current_face_color` now refer to the
  colors currently selected in the GUI (#686)

## 5 authors added to this release (alphabetical)

- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Tony Tung](https://github.com/napari/napari/commits?author=ttung) - @ttung

## 4 reviewers added to this release (alphabetical)

- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Tony Tung](https://github.com/napari/napari/commits?author=ttung) - @ttung
