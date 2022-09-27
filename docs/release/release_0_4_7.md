# napari 0.4.7

We're happy to announce the release of napari 0.4.7!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights

After nearly a year of planning, thanks to help from the fine folks at
[Quansight](https://labs.quansight.org/), napari now has a preferences dialog!
(#2211). It's a little sparse at the moment but we look forward to meeting all
your customization needs!

Thanks to Matthias Bussonnier and Talley Lambert, we've also improved our
notification handling and can now show tracebacks in our error notifcations in
the GUI!

You can now use PyData/Sparse arrays as labels layers, thanks to work by Draga
Doncila Pop (across both napari and sparse itself) and Juan Nunez-Iglesias.
This is useful when painting labels across large volumes/time series. As a
side-benefit of this work, undo when painting labels now works across the full
volume rather than only in the currently-visible plane. (#2339, #2396,
pydata/sparse#435 and related PRs)

We've also continued to make improvements to our experimental octree support,
which now supports single scale tiled loading (#2372, #2391). It is still
2D-only but continues to improve!

Last but not least, this is the first release since we launched our new website
at https://napari.org, powered by Jupyter Book! Check it out! Huge thanks to
Kira Evans, Talley Lambert, Lia Prins, Genevieve Buckley, and others who
contributed (and continue to contribute) to our efforts to improve our
documentation.

See below for the full list of changes.

## New Features
- Notification Manager  (#2205)
- Add preference dialog (#2211)
- Make napari strings localizable (#2429)

## Documentation
- Add profiling documentation (#1998)
- Add release note about pydantic viewermodel (#2334)
- Broken link fixed for Handling Code of Conduct Reports (#2342)
- Update nbscreenshot docstring (#2395)
- Clarify benchmark docs (#2431)
- Make structural doc changes to better align with sitemap (#2404)

## Improvements
- ColorManager take 2 (w/ pydantic) (#2204)
- Add basic selection model (#2293)
- Tooltips2 (#2310)
- Add a font size preview slider widget (#2318)
- Read multiple page from pypi (#2335)
- Use fancy indexing rather than full slices for label editing undo (#2339)
- Consolidate polygon base object (#2344)
- Use not mutable fields on Viewer where appropriate (#2346)
- Generic Button to toggle state. (#2359)
- Make `add_plugin_dock_widget` public, add CLI option (#2360)
- Add spiral indexing to bring in tiles at center of FOV first (#2363)
- Improve octree coverage process (#2366)
- Single scale tiled rendering (#2372)
- Add notification settings, add back console notifications (#2377)
- ipython embed example (#2378)
- Add contrast limits estimate for large plane (#2381)
- Discover dock widgets during get info (#2385)
- Add example of 2D, single-scale tiled rendering (#2391)
- Preference screen size (#2399)
- Add support for octree labels (#2403)
- Auto-enable gui qt when in IPython (#2406)
- Allow no vertex values to be passed to surface layer (#2408)
- Add base shape outline for single scale tiled (#2414)
- Change raw_to_displayed to only compute colours for all labels when required (#2415)
- Highlight widget (#2424)
- Remove brush shape from label UI (#2426)
- Remember Preferences dialog size. (#2434)
- `Selectable` mixin and `SelectableEventedList` (#2439)
- Unify evented/nestedEvented list APIs (#2440)
- Pretty-settings (#2448)
- Add a button to Qt Error Notification to show tracebacks (#2452)
- Update preferences dialog style to match designs (#2456)

## Bug Fixes
- Fix initialization of an empty Points layer (#2341)
- Fix octree clim (#2349)
- Fix json encoder inheritance on nested submodels for pydantic>=1.8.0 (#2357)
- Fix shortcut to toggle grid. (#2358)
- Fix rgb for octree (#2362)
- Fix deleting points data in viewer (#2383)
- Close floating docks on close event (#2397)
- Handle None settings on load (#2398)
- Fix preference cancel (#2410)
- Fix shapes data setter (#2411)
- Fix nd text on shapes layer (#2412)
- Fix overly sensitive Points layer dragging (#2413)
- Fix points delete (#2419)
- Fix empty layer text (#2420)
- Fix env-dependent test errors, remove warnings (#2421)
- Fix Labels `save_history` benchmark (#2430)
- Fix packaging (#2436)
- Fix numpy warnings (#2438)
- Fix some translation formatting (#2453)
- Remove extra remove window action from window menu and use file menu
  instead (#2454)
- Fix translation II (#2455)
- Fix for constant warning when using label brush (#2460)
- Restore QtNDisplayButton (#2464)

## API Changes

- Using `layer.get_status()` without an argument was deprecated in version
  0.4.4 and is removed in this version. Instead, use
  `layer.get_status(viewer.cursor.position, world=True)` (#2443).
- `layer.status` was deprecated in version 0.4.4 and is removed in this
  version. Instead, use `layer.get_status(viewer.cursor.position, world=True)`
  (#2443).
- Label-layers must be of integer type, float `Labels` layers are no longer allowed (see #2491)

## Deprecations

We are moving to a model in which layers don't know about the cursor or current
view, resulting in the following two deprecations:

- `layer.displayed_coordinates` is deprecated (#2327). Instead, use
  `[layer.coordinates[d] for d in viewer.dims.displayed]`
- `layer.position` and `layer.coordinates` are deprecated (#2443). Instead, use
  `viewer.cursor.position` for the position in world coordinates, and
  `layer.world_to_data(viewer.cursor.position)` for the position in data
  coordinates.

We have also deprecated the `napari.qt.QtNDisplayButton`. Instead a more general
`napari.qt.QtStateButton` is provided.

## Build Tools and Support
- Add environment flag for sparse library (#2396)
- re-add plausible (#2433)


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

