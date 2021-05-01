# napari 0.4.6

We're happy to announce the release of napari 0.4.6!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights
This release is the first that adds support for persistent settings in napari (#2212).
Right now we just store the current theme and window geometry but we will be expanding 
this to include a full set of preferences in future releases.

We've also made plugin installation from our plugin dialog more flexible, including
supporting installation from anything supported by `pip`, such as a
package name, url, or local file (#2319).


## New Features
- Add the ability to show contours of labels (#2168)
- Add initial settings management for theme and window geometry (#2212)
- More flexible plugin install (from url, name or file)  (#2319)


## Improvements
- Use pydantic for viewer model (#2066)
- Use vectorized indexing for xarray arrays in labels (#2191)
- Add basic composite tree model (#2217)
- Add Qt Tree Model and View (#2222)
- Change status_format to return an existing string as-is (#2235)
- Add translator stub and example (#2254)
- Add hook implementation function name to plugin call order sorter (#2255)
- Add `theme` parameter to `get_stylesheet` (#2263)
- Add typing to event.py (#2269)
- Add `canvas.background_color_override` (#2270)
- QColoredSVGIcon class for dynamic colorized svg icons (#2279)
- Add EventedSet class (#2280)
- Refactor icon build process (#2285)
- Add more tooltips (#2292)
- Make contour thickness settable (#2296)
- Add `_json_encode` method API for EventedModel serdes (#2297)
- Don't subclass `Layer` directly from `Image`, (adds `_ImageBase`) (#2307)
- Disable run pre release test outside napari main repository (#2331)


## Bug Fixes
- Fix QtPoll to poll when camera moves (#2227)
- Prevent garbage collection of viewer during startup. (#2262)
- Fix EventedModel signatures with PySide2 imported (#2265)
- Remove callsignature (#2266)
- Fix bug with not display points close to given layer (#2289)
- Fix plugin discovery in bundle (use certifi context for urlopen) (#2298)
- Call processEvents before nbscreenshot (#2303)
- Add handling of python <= 3.7 for translator (#2305)
- Add parent parameter to about dialog to correctly inherit and apply theme (#2308)
- Remove unneeded stylesheet on QtAbout dialog (#2312)
- Avoid importing xarray (and other array libs) for equality checkers (#2325)

## API Changes
- The ViewerModel is now a Pydantic BaseModel and so subclassing the Viewer might require
reading some of the [Pydantic BaseModel documentation](https://pydantic-docs.helpmanual.io/usage/models/).
Otherwise the API of the ViewerModel is unchanged.
- Because `Labels` no longer inherit from `Image` (#2307), magicgui function
  parameters annotated as `napari.layers.Image` will no longer include
  `napari.layers.Labels` layer types in the dropdown menu. (`napari.layers.Layer`
  may still be used to show all layer types in the magicgui-generated dropdown
  menu.)

## Build Tools and Support
- Add missing release notes 0.4.5 (#2250)
- Explicitly document ability to implement multiple of the same hook (#2256)
- Update install instructions to work with zsh default Mac shell (#2258)
- Fix broken links in README (#2275)
- Documentation typos (#2294)
- Documentation on napari types in magicgui (#2306)
- Pin pydantic < 1.8.0 (#2323)
- Fix docs build with code execution (#2324)
- Pin pydantic != 1.8.0 (#2333)


## 12 authors added to this release (alphabetical)

- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Ian Hunt-Isaak](https://github.com/napari/napari/commits?author=ianhi) - @ianhi
- [kir0ul](https://github.com/napari/napari/commits?author=kir0ul) - @kir0ul
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Neil Weisenfeld](https://github.com/napari/napari/commits?author=nweisenfeld) - @nweisenfeld
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03


## 13 reviewers added to this release (alphabetical)

- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [kir0ul](https://github.com/napari/napari/commits?author=kir0ul) - @kir0ul
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Neil Weisenfeld](https://github.com/napari/napari/commits?author=nweisenfeld) - @nweisenfeld
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [ziyangczi](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi
