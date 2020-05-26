# napari 0.3.2

We're happy to announce the release of napari 0.3.2!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari


## New Features
- Rich jupyter display for napari screenshots (#1269)

## Improvements
- make Qt component non private module (#1122)
- add option to disable overwrite of labels during painting (#1264)
- Refactor: move dask utils from misc.py to new dask_utils.py (#1265)
- Have layer tooltip display the layer name (#1271)
- Support viewer annotation in magicgui (#1279)
- Blue colored active layer (#1284)

## Bug Fixes
- Automatically set high DPI scaling in Qt (#820)
- Use napari logo as taskbar icon on windows when run as python script (#1208)
- Remove scipy.stats import (#1250)
- Unify `_get_exent()` return tuple formatting for all layer types (#1255)
- Anti-aliasing on splash screen logo (#1260)
- Remove dupe import (#1263)
- Fix missing docstring `create_dask_cache` (#1266)
- Fix adding points with new properties  (#1274)

## Build Tools
- Add pooch to requirements/test.txt (#1249)
- Prefer rcc binary at front of path (#1261)
- Pin napari-svg to 0.1.2 (#1275)


## 9 authors added to this release (alphabetical)

- [Chris Wood](https://github.com/napari/napari/commits?author=cwood1967) - @cwood1967
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Guido Cadenazzi](https://github.com/napari/napari/commits?author=gcadenazzi) - @gcadenazzi
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [ziyangczi](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi


## 11 reviewers added to this release (alphabetical)

- [Chris Wood](https://github.com/napari/napari/commits?author=cwood1967) - @cwood1967
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Guillaume Witz](https://github.com/napari/napari/commits?author=guiwitz) - @guiwitz
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Mark Harfouche](https://github.com/napari/napari/commits?author=hmaarrfk) - @hmaarrfk
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [ziyangczi](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi
