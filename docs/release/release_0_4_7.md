# napari 0.4.7

We're happy to announce the release of napari 0.4.7!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights


## New Features


## Improvements


## Bug Fixes


## API Changes


## Deprecations


## Build Tools


## Other Pull Requests

- Add profiling documentation (#1998)
- ColorManager take 2 (w/ pydantic) (#2204)
- Notification Manager  (#2205)
- Add preference dialog (#2211)
- Add basic selection model (#2293)
- Tooltips2 (#2310)
- PR: Add a font size preview slider widget (#2318)
- Deprecate displayed coordinates (#2327)
- Add release note about pydantic viewermodel (#2334)
- Read multiple page from pypi (#2335)
- Use fancy indexing rather than full slices for label editing undo (#2339)
- Fix initialization of an empty Points layer (#2341)
- Broken link fixed for Handling Code of Conduct Reports (#2342)
- Consolidate polygon base object (#2344)
- Use not mutable fields on Viewer where appropriate (#2346)
- Fix octree clim (#2349)
- Fix json encoder inheritance on nested submodels for pydantic>=1.8.0 (#2357)
- Fix Shortcut to toggle grid. (#2358)
- Generic Button to toggle state. (#2359)
- Make `add_plugin_dock_widget` public, add CLI option (#2360)
- Fix rgb for octree (#2362)
- Add spiral indexing to bring in tiles at center of FOV first (#2363)
- Improve octree coverage process (#2366)
- Single scale tiled rendering (#2372)
- Add notification settings, add back console notifications (#2377)
- ipython embed example (#2378)
- Add contrast limits estimate for large plane (#2381)
- Fix deleting points data in viewer (#2383)
- discover dock widgets during get info (#2385)
- Add example of 2D, single-scale tiled rendering (#2391)
- update nbscreenshot docstring (#2395)
- Add environment flag for sparse library (#2396)
- PR: Close floating docks on close event (#2397)
- PR: Handle None settings on load (#2398)
- Preference screen size (#2399)
- Add support for octree labels (#2403)
- make structural doc changes to better align with sitemap (#2404)
- [Preference] Auto-enable gui qt when in IPython (#2406)
- Allow no vertex values to be passed to surface layer (#2408)
- Fix preference cancel (#2410)
- Fix shapes data setter (#2411)
- Fix nd text on shapes layer (#2412)
- Add base shape outline for single scale tiled (#2414)
- Change raw_to_displayed to only compute colours for all labels when required (#2415)
- fix points delete (#2419)
- Fix empty layer text (#2420)
- fix env-dependent test errors, remove warnings (#2421)
- Highlight widget (#2424)
- Remove brush shape from label UI (#2426)
- PR: Make napari strings localizable (#2429)
- Fix Labels `save_history` benchmark (#2430)
- Clarify benchmark docs (#2431)
- re-add plausible (#2433)
- PR: Remember Preferences dialog size. (#2434)
- Fix packaging (#2436)
- fix numpy warnings (#2438)
- `Selectable` mixin and `SelectableEventedList` (#2439)
- Unify evented/nestedEvented list APIs (#2440)
- Deprecate layer.position and layer.coordinates (#2443)
- pretty-settings (#2448)

## 14 authors added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [DenisSch](https://github.com/napari/napari/commits?author=DenisSch) - @DenisSch
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [ziyangczi](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi


## 12 reviewers added to this release (alphabetical)

- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Thomas A Caswell](https://github.com/napari/napari/commits?author=tacaswell) - @tacaswell
- [ziyangczi](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi

