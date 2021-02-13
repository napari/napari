# napari 0.4.4

We're happy to announce the release of napari 0.4.4!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights
This release is a quick follow on from our `0.4.3` release and contains some
nice improvements to the GUI and analysis function hookspecs we experimentally
added in that release. We've expanded the API of the
`napari_experimental_provide_dock_widget` to accept new `magic_factory`-
decorated functions (available in magicgui 0.2.6+), or any callable that
returns one or more widgets, making it easier for developers who want to use
magicgui and not have to write their own qt widgets (#2143).

We have also renamed `napari_experimental_provide_function_widget` to
`napari_experimental_provide_function` and reduced its API to receive only a
function or list of functions. napari will then take care to generate an
appropriate user interface for that function. This will make it even easier to
create analysis pipelines in napari (#2158).


## Improvements
- Add example code to hook documentation (#2112)
- move Viewer import into method (#2119)
- Support for EventedList.__setitem__ with array-like items (#2120)
- Add __array__ protocol to transforms.Affine (#2137)
- Relax dock_widget_hookspec to accept callable. (#2143)
- Add name of system to napari sys_info (#2147)
- Points layer enable interactive mode in add mode, don't add point when dragging, addresses #2146 (WIP) (#2148)
- Change plugin window search from naming convention to pypi classifier (#2153)
- Add compress=1 to tifffile imsave call (#2157)
- Add informations on what to do on error in GUI (#2165)

## Documentation
- Better documentation of API changes in 0.4.4 release notes (#2171)
- Add new function and dock widget hook specifications to documentation (#2158)

## Bug Fixes
- QtAboutKeyBindings patch (#2132)
- Fix too-late registration of napari types in magicgui (#2139)
- Fix magicgui.FunctionGui deprecation warning (#2164)
- Fix show/ hide of plugin widgets (#2173)


## API Changes
- `viewer.grid_view()` has been removed, use `viewer.grid.enabled = True`
  instead (#2144)
- `viewer.stack_view()` has been removed, use `viewer.grid.enabled = False`
  instead (#2144)
- `viewer.grid_size` has been removed, use `viewer.grid.shape` instead (#2144)
- `viewer.grid_stride` has been removed, use `viewer.grid.stride` instead
  (#2144)
- Plugins are no longer discovered by naming convention alone; to appear as an
  installed plugin in napari, make sure you use the `napari.plugin`
  entrypoint in your setup.py or setup.cfg. To appear as a listed plugin on
  PyPI, be sure to use the `Framework :: napari` trove classifier. (#2152)
- The `napari_experimental_provide_function_widget` plugin hook
  specification has been removed. Plugin developers should use
  `napari_experimental_provide_function` instead for functional
  (layers/layer data in -> layers/layer data out) plugins, and
  `napari_experimental_provide_dock_widget` for more elaborate plugins
  that require custom widgets. (#2158)


## Deprecations
- `layer.status` is deprecated, to be removed in 0.4.6. Users should instead
  use `layer.get_status(position)`. (#1985)
- The position argument to `layer.get_value()` is no longer optional, and will
  be required after 0.4.6. (#1985)
- `layer.get_message()` is deprecated, to be removed in 0.4.6. Users should use
  `layer.get_status(position)` instead. (#1985)


## Build Tools and Support
- Add missed doc string in `import_resources` (#2113)
- Delay import of pkg_resources (#2121)
- Remove duplicate entry in install_requires (#2122)
- Fix typo in deprecation message (#2124)
- DOC: Formatting and Typos. (#2129)
- DOC: Rename Section to conform to numpydocs (Return->Returns) (#2130)
- Provide `make_test_viewer` as a pytest plugin, for external use. (#2131)
- Doc: fix syntax = instead of : (#2141)


## 11 authors added to this release (alphabetical)

- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Christoph Gohlke](https://github.com/napari/napari/commits?author=cgohlke) - @cgohlke
- [dongyaoli10x](https://github.com/napari/napari/commits?author=dongyaoli10x) - @dongyaoli10x
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [nhthayer](https://github.com/napari/napari/commits?author=nhthayer) - @nhthayer
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Robert Haase](https://github.com/napari/napari/commits?author=haesleinhuepf) - @haesleinhuepf
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Volker Hilsenstein](https://github.com/napari/napari/commits?author=VolkerH) - @VolkerH


## 6 reviewers added to this release (alphabetical)

- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [nhthayer](https://github.com/napari/napari/commits?author=nhthayer) - @nhthayer
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Robert Haase](https://github.com/napari/napari/commits?author=haesleinhuepf) - @haesleinhuepf
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi
