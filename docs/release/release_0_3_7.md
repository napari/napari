# napari 0.3.7

We're happy to announce the release of napari 0.3.7!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:
https://napari.org. This release contains two major new experimental features —
a standalone bundle and asynchronous rendering, see below. We deeply appreciate
any bug reports at https://github.com/napari/napari/issues/new/choose.

## Highlights

With this release, we are launching an experimental standalone app (#1289). You
can find it on our [GitHub releases
page](https://github.com/napari/napari/releases) (scroll down to "Assets" for
this release). You can install reader/writer plugins from the app itself, so
you can use the napari app to view many different kinds of datasets!

We also have an also-experimental, work-in-progress asynchronous rendering
mode (#1565, #1583) to make the viewer interactivity smoother with slow-loading
datasets, such as those backed by remote data or by dask computation. To opt
into this, set the NAPARI_ASYNC environment variable to anything other than
"0". 3D rendering and multiscale are currently not supported. If you encounter
an issue please check the [async label](https://github.com/napari/napari/labels/async)
on the repository to see if your issue is known.

Scrolling through n-dimensional datasets has become a bit more convenient:
scroll to zoom, as always, but hold Ctrl (Cmd on Mac) while scrolling to move
up and down a stack. (#1434, #1525)

This is in addition to many bug fixes and usability improvements — see below
for the full list! Thank you to everyone who contributed to this release!

## New Features
- Add briefcase app bundles to release assets (#1289)
- Evented list (#1444)
- Async rendering part 1 (#1565)
- Async rendering part 2 (#1583)
- Add SELECTED color-mode for individual label visibility (#1555)

## Improvements
- auto generate view_* methods (#978)
- Add world extents for layers (#1360)
- Reorganize _qt module (#1431)
- Add CTRL+ Mouse Scroll to scroll through last active stack (#1434)
- Perfmon System version 2 (#1453)
- Add global exception catching in Qt (#1476)
- Close napari window when KeyboardInterupt (#1494)
- Add error dialog for global error handler (#1511)
- Add simple colormap object (#1523)
- Better support for multiple translated arrays (#1539)
- Improve Shapes layer performance (#1561)
- Add support for deprecate signal in EmitterGroup (#1582)


## Bug Fixes
- Prevent napari crashing after layer is dragged into trash (#1487)
- Fix URL parsing scheme for view_path (#1515)
- Fix potentially-confusing typo in comment (#1521)
- Fix magic_name and kwargs propagation with view_path (#1522)
- Address trackpad support for dim scrolling (#1525)
- Changed class evaluation from `type` to `isinstance` (#1528)
- Add test for theme changing (#1534)
- Addressed colormap control color with same label color (#1537)
- Fix menubar raise on mac by running with framework python (#1554)
- Fix typo in perfmon patcher (#1560)
- Fix import plugin when opening dialog (#1578)
- Longer spinbox (#1580)
- Minor updates to custom mouse functionality examples (#1592)
- Do not raise viewer in IPython with pre-existing event loop (#1595)
- Restore multiple command-line arguments (#1597)
- Lock briefcase version (#1599)


## Build Tools
- Points slicing benchmark (#1435)
- Fix 0.3.6 release notes to indicate reversions (#1498)
- format imports with isort (#1505)
- Update flake8 from 3.7.9 to 3.8.3 (#1506)
- Add plausible to docs (#1526)
- Clean up all warnings in tests (#1538)
- Filter all nan axis warning on layer extent (#1540)
- pin conda in CI tests to 4.8.3 (#1541)
- CI cache reset (#1548)
- Fix perfmon for qt changes, add counters, cleanup (#1550)
- Revert "CI cache reset" (#1551)
- Add fingerprint script to pip caching (#1553)
- Add Shapes benchmarks for interactions (#1563)
- Exclude test_bundle.py from distribution (#1604)
- Skip bundle test when setup.cfg is missing (#1608)

## 11 authors added to this release (alphabetical)

- [Dan Allan](https://github.com/napari/napari/commits?author=danielballan) - @danielballan
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Hector](https://github.com/napari/napari/commits?author=hectormz) - @hectormz
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi


## 12 reviewers added to this release (alphabetical)

- [Davis Bennett](https://github.com/napari/napari/commits?author=d-v-b) - @d-v-b
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Fedor Korotkov](https://github.com/napari/napari/commits?author=fkorotkov) - @fkorotkov
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justine Larsen](https://github.com/napari/napari/commits?author=justinelarsen) - @justinelarsen
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi

