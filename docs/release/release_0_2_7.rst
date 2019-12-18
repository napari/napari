Announcement: napari 0.2.7
==========================

We're happy to announce the release of napari v0.2.7!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-base
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

Improvements
************
- Iso-surface threshold slider (#712)
- Add play button to gui (#726)
- Make layers list dockable (#727)
- Add Zenodo badge to documentation (#743)
- Add a dock icon (#744)
- Show splash screen for cli launch (#745)
- Add benchmarks for setting .data in Image layers (#747)
- Refactor layer tests to be more parametrized (#723)
- Support opening labels layers via directly from path (#748)
- Simplify keybindings info display (#749)
- Clean up info box (#750)
- Display slice info on right of slider (#759)
- Block refresh for invisible layers (#776)
- About copy button to info display box (#798)
- Add blocked_signals context manager (#797)

Bugfixes
********
- Fix StringEnum setting and errors (#757)
- scale argument now accepts array-like input (#765)
- fix set_fps type to float (#767)
- Add shutdown method to QtViewer that closes all resources (#769)
- Change language around windows support in readme (#779)
- Revert #784 console shutdown conditionals (#796)
- Fix window raise & inactive menubar conflict (#795)
- Change documentation on qt.py folder location (#783)
- Updating qt_console with better resource management (#784)
- Respect vispy max texture limits (#788)
- Fix (minor) deprecation warnings (#800)
- Fix FPS spin box on Qt < 5.12 (#803)
- Bumpy vispy 0.6.4 (#807)
- Set threshold for codecov failure (#806)

API Changes
***********
- Rename util to utils across repo (#808)
- Move Labels utility functions to labels_util.py (#770)
- Move Image layer utility functions to image_utils.py (#775)
- Move Layer utility functions to /napari/layers/layer_utils.py (#778)
- Refactor util.misc (#781)

Deprecations
************
- drop ndim keyword from labels layer (#773)


7 authors added to this release [alphabetical by first name or login]
---------------------------------------------------------------------
- Genevieve Buckley
- Kevin Yamauchi
- Kira Evans
- Nicholas Sofroniew
- Peter Boone
- Shannon Axelrod
- Talley Lambert


7 reviewers added to this release [alphabetical by first name or login]
-----------------------------------------------------------------------
- Ahmet Can Solak
- Juan Nunez-Iglesias
- Kevin Yamauchi
- Kira Evans
- Nicholas Sofroniew
- Shannon Axelrod
- Talley Lambert
