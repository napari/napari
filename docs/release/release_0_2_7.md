# napari 0.2.7

We're happy to announce the release of napari 0.2.7! napari is a fast,
interactive, multi-dimensional image viewer for Python. It's designed for
browsing, annotating, and analyzing large multi-dimensional images. It's built
on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the
scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights

- Play button for animating axes now in the GUI
- Threshold slider for much improved isosurface rendering
- Dockable widgets (!)
- Slice information on sliders
- Dramatically improved performance with many invisible layers
- Adopted a Governance model with a mission and values statement
- Added a new Core Dev guide

## New Features

- Add governance model, mission and values, core dev guide (#655)

## Improvements

- Iso-surface threshold slider (#712)
- Add play button to GUI (#726)
- Make layers list dockable (#727)
- Add Zenodo badge to documentation (#743)
- Add a dock icon (#744)
- Show splash screen for cli launch (#745)
- Add benchmarks for setting `.data` in Image layers (#747)
- Refactor layer tests to be more parametrized (#723)
- Support opening labels layers via directly from path (#748)
- Simplify keybindings info display (#749)
- Clean up info box (#750)
- Display slice info on right of slider (#759)
- Block refresh for invisible layers (#776)
- About copy button to info display box (#798)
- Add blocked_signals context manager (#797)
- Better selected menu header background color (#813)

## Bug Fixes

- Fix StringEnum setting and errors (#757)
- scale argument now accepts array-like input (#765)
- fix `set_fps` type to float (#767)
- Add shutdown method to QtViewer that closes all resources (#769)
- Change language around windows support in readme (#779)
- Revert #784 console shutdown conditionals (#796)
- Fix window raise & inactive menubar conflict (#795)
- Change documentation on qt.py folder location (#783)
- Updating qt_console with better resource management (#784)
- Respect vispy max texture limits (#788)
- Fix (minor) deprecation warnings (#800)
- Fix FPS spin box on Qt < 5.12 (#803)
- Bumpy vispy dependency to 0.6.4 (#807)
- Set threshold for codecov failure (#806)
- Rename util to utils in MANIFEST.in (#811)
- Add `requirements/release.txt` with release dependencies (#809)

## API Changes

- Rename util to utils across repo (#808)
- Move Labels utility functions to labels_util.py (#770)
- Move Image layer utility functions to image_utils.py (#775)
- Move Layer utility functions to /napari/layers/layer_utils.py (#778)
- Refactor util.misc (#781)
- Drop ndim keyword from labels layer (#773)

## 7 authors added to this release (alphabetical)

- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Peter Boone](https://github.com/napari/napari/commits?author=boonepeter) - @boonepeter
- [Shannon Axelrod](https://github.com/napari/napari/commits?author=shanaxel42) - @shanaxel42
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03

## 7 reviewers added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Shannon Axelrod](https://github.com/napari/napari/commits?author=shanaxel42) - @shanaxel42
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
