# napari 0.4.18

We're happy to announce the release of napari 0.4.18!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights

- Add lasso tool for faster drawing of polygonal Shapes (#5555)
- Feature: support for textures and vertex colors on Surface layers (#5642)
- Zooming with the mouse wheel in any mode (#5701)
- Fix inefficient label mapping in direct color mode (10-20x speedup) (#5723)
- Efficient labels mapping for drawing in Labels (60 FPS even with 8000x8000 images) (#5732)

## New Features

- Overlays 2.0 (#4894)
- expose custom image interpolation kernels (#5130)
- Add user agent environment variable for pip installations (#5135)
- Set selection color for QListView item. (#5202)
- Add warning about set private attr when using proxy (#5209)
- Shapes interpolation (#5334)
- Add dask settings to preferences (#5490)
- Add lasso tool for faster drawing of polygonal Shapes (#5555)
- Feature: support for textures and vertex colors on Surface layers (#5642)
- Back point selection with a psygnal Selection (#5691)
- Zooming with the mouse wheel in any mode (#5701)
- Add cancellation functionality to progress (#5728)

## Improvements

- Set keyboard focus on console when opened (#5208)
- Push variables to console when instantiated (#5210)
- Tracks layer creation performance improvement (#5303)
- PERF: Event emissions and perf regression. (#5307)
- Much faster FormatStringEncoding (#5315)
- Add parent when creating layer context menu to inherit application theme and add style entry for disabled widgets and menus (#5381)
- Add correct `enablement` kwarg to `Split Stack` action, `Convert data type` submenu and `Projections` submenu (#5437)
- Apply disabled widgets style only for menus and set menus styles for `QModelMenu`  and `QMenu` instances (#5446)
- Add disabled style rule for `QComboBox` following the one for `QPushButton` (#5469)
- Allow layers control section to resize to contents (#5474)
- Allow to use `Optional` annotation in function return type for magicgui functions (#5595)
- Skip equality comparisons in EventedModel when unnecessary (#5615)
- Bugfix: improve layout of Preferences > Shortcuts tables (#5679)
- Add dev example for adding custom overlays. (#5719)
- Disable buffer swapping (#5741)
- Explicitly list valid layer names in types (#5823)
- Sort npe1 widget contributions (#5865)
- feat: add `since_version` argument of `rename_argument` decorator (#5910)
- Emit extra information with layer.events.data (#5967)

## Performance

- Return early when no slicing needed (#5239)
- Tracks layer creation performance improvement (#5303)
- PERF: Event emissions and perf regression. (#5307)
- Much faster FormatStringEncoding (#5315)
- Fix inefficient label mapping in direct color mode (10-20x speedup) (#5723)
- Efficient labels mapping for drawing in Labels (60 FPS even with 8000x8000 images) (#5732)
- Disable buffer swapping (#5741)

## Bug Fixes

- Warn instead of failing on empty or invalid alt-text (#4505)
- Fix display of order and scale combinations (#5004)
- Enforce that contrast limits must be increasing (#5036)
- Bugfix: Move Window menu to be before Help (#5093)
- Add extra garbage collection for some viewer tests (#5108)
- Connect image to plane events and expose them (#5131)
- Workaround for discover themes from plugins (#5150)
- Add missed dialogs to `qtbot` in `test_qt_notifications` to prevent segfaults (#5171)
- DOC Update docstring of `add_dock_widget` & `_add_viewer_dock_widget` (#5173)
- Fix unsortable features (#5186)
- Avoid possible divide-by-zero in Vectors layer thumbnail update (#5192)
- Disable napari-console button when launched from jupyter (#5213)
- Volume rendering updates for isosurface and attenuated MIP (#5215)
- Return early when no slicing needed (#5239)
- Check strictly increasing values when clipping contrast limits to a new range (#5258)
- UI Bugfix: Make disabled QPushButton more distinct (#5262)
- Respect background color when calculating scale bar color (#5270)
- Fix circular import in _vispy module (#5276)
- Use only data dimensions for cord in status bar (#5283)
- Prevent obsolete reports about failure of cleaning viewer instances (#5317)
- Add scikit-image[data] to install_requires, because it's required by builtins (#5329)
- Fix repeating close dialog on macOS and qt 5.12 (#5337)
- Disable napari-console if napari launched from vanilla python REPL (#5350)
- For npe2 plugin, use manifest display_name for File > Open Samples (#5351)
- Bugfix plugin display_name use (File > Open Sample, Plugin menus) (#5366)
- Fix editing shape data above 2 dimensions (#5383)
- Fix test keybinding for layer actions (#5406)
- fix theme id not being used correctly (#5412)
- Clarify layer's editable property and separate interaction with visible property (#5413)
- Fix theme reference to get image for `success_label` style (#5447)
- Bugfix: Ensure layer._fixed_vertex is set when rotating (#5449)
- Fix `_n_selected_points` in _layerlist_context.py (#5450)
- Refactor Main Window status bar to improve information presentation (#5451)
- Bugfix: Fix test_get_system_theme test for `name` to `id` change (#5456)
- Bugfix: POLL_INTERVAL_MS used in QTimer needs to be an int on python 3.10 (#5467)
- Bugfix: Add missing Enums and Flags required by PySide6 > 6.4 (#5480)
- BugFix: napari does not start with Python v3.11.1: "ValueError: A distribution name is required." (#5482)
- Fix inverted LUT and blending (#5487)
- Fix opening file dialogs in PySide (#5492)
- Handle case when QtDims play thread is partially deleted  (#5499)
- Ensure surface normals and wireframes are using Models internally (#5501)
- Recursively check for dependent property to fire events. (#5528)
- Set PYTHONEXECUTABLE as part of macos fixes on (re)startup (#5531)
- Un-set unified title and tool bar on mac (Qt property) (#5533)
- Fix key error issue of action manager (#5539)
- Bugfix: ensure Checkbox state comparisons are correct by using Qt.CheckState(state) (#5541)
- Clean dangling widget in test  (#5544)
- Fix `test_worker_with_progress` by wait on worker end (#5548)
- Fix min req  (#5560)
- Fix vispy axes labels (#5565)
- Fix colormap utils error suggestion code and add a test (#5571)
- Fix problem of missing plugin widgets after minimize napari (#5577)
- Make point size isotropic (#5582)
- Fix guard of qt import in `napari.utils.theme` (#5593)
- Fix empty shapes layer duplication and `Convert to Labels` enablement logic for selected empty shapes layers (#5594)
- Stop using removed multichannel= kwarg to skimage functions (#5596)
- Add information about `syntax_style` value in error message for theme validation (#5602)
- Remove catch_warnings in slicing (#5603)
- Incorret theme should not prevent napari from start (#5605)
- Unblock axis labels event to be emitted when slider label changes (#5631)
- Bugfix: IndexError slicing Surface with higher-dimensional vertex_values (#5635)
- Bugfix: Convert Viewer Delete button to QtViewerPushButton with action and shortcut (#5636)
- Change dim `axis_label` resize logic to set width using only displayed labels width (#5640)
- Feature: support for textures and vertex colors on Surface layers (#5642)
- Fix features issues with init param and property setter (#5646)
- Bugfix: Don't double toggle visibility for linked layers (#5656)
- Bugfix: ensure pan/zoom buttons work, along with spacebar keybinding (#5669)
- Bugfix: Add Tracks to qt_keyboard_settings (#5678)
- Fix automatic naming and GUI exposure of multiple unnamed colormaps (#5682)
- Fix mouse movement handling for `TransformBoxOverlay` (#5692)
- Update environment.yml (#5693)
- Resolve symlinks from path to environment for setting path (#5704)
- Fix tracks color-by when properties change (#5708)
- Fix Sphinx warnings (#5717)
- Do not use depth for canvas overlays; allow setting blending mode for overlays (#5720)
- Unify event behaviour for points and its qt controls (#5722)
- Fix camera 3D absolute rotation bug (#5726)
- Maint: Bump mypy (#5727)
- Style `QGroupBox` indicator (#5729)
- Fix centering of non-displayed dimensions (#5736)
- Don't attempt to use npe1 readers in napari.plugins._npe2.read (#5739)
- Prevent canvas micro-panning on point add (#5742)
- Use text opacity to signal that widget is disabled (#5745)
- Bugfix: Add the missed keyReleaseEvent method in QtViewerDockWidget (#5746)
- Update status bar on active layer change (#5754)
- Use array size directly when checking multiscale arrays to prevent overflow (#5759)
- Fix path to `check_updated_packages.py` (#5762)
- Brush cursor implementation using an overlay (#5763)
- Bugfix: force a redraw to ensure highlight shows when Points are select-all selected (#5771)
- Fix copy/paste of points (#5795)
- Fix multiple viewer example (#5796)
- Fix colormapping nD images (#5805)
- Enforce Points.selected_data type as Selection (#5813)
- Change toggle menubar visibility functionality to hide menubar and show it on mouse movement validation (#5824)
- Bugfix: Disconnect callbacks on object deletion in special functions from `event_utils`  (#5826)
- Do not blend color in QtColorBox with black using opacity (#5827)
- Don't allow negative contour values (#5830)
- Bugfixes for layer overlays: clean up when layer is removed + fix potential double creation (#5831)
- Add compatibility to PySide in file dialogs by using positional arguments (#5834)
- Bugfix: fix broken "show selected" in the Labels layer (because of caching) (#5841)
- Add tests for popup widgets and fix perspective popup slider initialization (#5848)
- [Qt6] Fix AttributeError on renaming layer (#5850)
- Bugfix: Ensure QTableWidgetItem(action.description) item is enabled (#5854)
- Add constraints file during installation of packages from pip in docs workflow (#5862)
- Bugfix: link the Labels model to the "show selected" checkbox (#5867)
- Add `__all__` to `napari/types.py` (#5894)
- Fix drawing vertical or horizontal line segments in Shapes layer (#5895)
- Disallow outside screen geometry napari window position (#5915)
- Fix `napari-svg` version parsing in `conftest.py` (#5947)
- Fix issue in utils.progress for disable=True (#5964)
- Set high DPI attributes when using PySide2 (#5968)

## API Changes

- Overlays 2.0 (#4894)
- expose custom image interpolation kernels (#5130)
- Connect image to plane events and expose them (#5131)

## Deprecations


## Build Tools

- ci(dependabot): bump styfle/cancel-workflow-action from 0.10.0 to 0.10.1 (#5158)
- ci(dependabot): bump actions/checkout from 2 to 3 (#5160)
- ci(dependabot): bump styfle/cancel-workflow-action from 0.10.1 to 0.11.0 (#5290)
- ci(dependabot): bump docker/login-action from 2.0.0 to 2.1.0 (#5291)
- ci(dependabot): bump actions/upload-artifact from 2 to 3 (#5292)
- Pin mypy version (#5310)
- MAINT: Start testing on Python 3.11 in CI. (#5439)
- Pin test dependencies (#5715)

## Documentation

- DOC Update doc contributing guide (#5114)
- Napari debugging during plugin development documentation (#5142)
- DOC Update docstring of `add_dock_widget` & `_add_viewer_dock_widget` (#5173)
- Specified that the path is to the local folder in contributing documentation guide. (#5191)
- Fixes broken links in latest docs version (#5193)
- Fixes gallery ToC (#5458)
- Fix broken link in EmitterGroup docstring (#5465)
- Fix Sphinx warnings (#5717)
- Add Fourier transform playground example (#5872)
- Improve documentation of `changed` event in EventedList (#5928)
- Set removal version in deprecation of Viewer.rounded_division (#5944)
- Pre commit fixes for 0.4.18 release branch (#5985)

## Other Pull Requests

- use app-model for view menu (#4826)
- Overlay backend refactor (#4907)
- Migrate help menu to use app model (#4922)
- Refactor layer slice/dims/view/render state (#5003)
- MAINT: increase min numpy version. (#5089)
- Refactor qt notification and its test solve problem of segfaults (#5138)
- Decouple changing viewer.theme from changing theme settings/preferences (#5143)
- [DOCS] misc invalid syntax updates. (#5176)
- MAINT: remove vendored colorconv from skimage. (#5180)
- Re-add README screenshot (#5220)
- MAINT: remove requirements.txt and cache actions based on setup.cfg. (#5234)
- Explicitly set test array data to fix a flaky test (#5245)
- Add ruff linter to pre-commit (#5275)
- Run tests on release branch (#5277)
- Vispy 0.12: per-point symbol and several bugfixes (#5312)
- Make all imports absolute (#5318)
- Fix track ids features ordering for unordered tracks (#5320)
- tests: remove private magicgui access in tests (#5331)
- Make settings and cache separate per each environment.  (#5333)
- Remove internal event connection on SelectableEventedList (#5339)
- Unset `PYTHON*` vars and use entitlements in macOS conda menu shortcut (#5354)
- Distinguish between update_dims, extent changes, and refresh (#5363)
- Add checks for pending Qt threads and timers in tests (#5373)
- Suppress color conversion warning when converting invalid LAB coordinates (#5386)
- Fix warning when fail to import qt binding. (#5388)
- Update `MANIFEST.in` to remove warning when run tox (#5393)
- [Automatic] Update albertosottile/darkdetect vendored module (#5394)
- Update citation metadata (#5398)
- [pre-commit.ci] pre-commit autoupdate (#5403)
- Fix flaky dims playback test by waiting for playing condition (#5414)
- [Automatic] Update albertosottile/darkdetect vendored module (#5416)
- [pre-commit.ci] pre-commit autoupdate (#5422)
- Avoid setting corner pixels for empty layers (#5423)
- Maint: Typing and ImportError -> ModuleNotFoundError. (#5431)
- Fix tox `passenv` setup for `DISPLAY` and `XAUTHORITY` environment variables (#5441)
- Add `error` color to themes and change application close/exit dialogs (#5442)
- Update screenshot in readme (#5452)
- Maint: Fix sporadic QtDims garbage collection failures by converting some stray references to weakrefs (#5471)
- Replace GabrielBB/xvfb-action (#5478)
- Add tags to recently added examples (#5486)
- Remove layer ndisplay event (#5491)
- MAINT: Don't format logs in log call (#5504)
- Replace flake8, isort and pyupgrade by ruff, enable additional usefull rules  (#5513)
- Second PR that enables more ruff rules.  (#5520)
- Use pytest-pretty for better log readability  (#5525)
- MAINT: Follow Nep29, bump minimum numpy. (#5532)
- [pre-commit.ci] pre-commit autoupdate (#5534)
- Move layer editable change from slicing to controls (#5546)
- update conda_menu_config.json for latest fixes in menuinst (#5564)
- Enable the `COM` and `SIM` rules in `ruff` configuration (#5566)
- Move from ubuntu 18.04 to ubuntu 20.04 in workflows (#5578)
- FIX: Fix --pre skimage that have a more precise warning message (#5580)
- Remove leftover duplicated code (#5586)
- Remove napari-hub API access code (#5587)
- Enable `ruff` rules part 4. (#5590)
- [pre-commit.ci] pre-commit autoupdate (#5592)
- Maint: ImportError -> ModuleNotFoundError. (#5628)
- [pre-commit.ci] pre-commit autoupdate (#5645)
- MAINT: Do not use mutable default for dataclass. (#5647)
- MAINT: Do not use cgi-traceback on 3.11+ (deprecated, marked for removal) (#5648)
- MAINT: Add explicit level to warn. (#5649)
- MAINT: Split test file in two to find hanging test. (#5680)
- Skip pyside6 version 6.4.3 for tests (#5683)
- Pin pydantic. (#5695)
- fix test_viewer_open_no_plugin exception message expectation (#5698)
- fix: Block PySide6==6.5.0 in tests (#5702)
- Don't resize shape after `Shift` release until mouse moves (#5707)
- Update `test_examples` job dependencies, unskip `surface_timeseries_.py` and update some examples validations (#5716)
- Add test to check basic interactions with layer controls widgets (#5757)
- test: [Automatic] Constraints upgrades: `dask`, `hypothesis`, `imageio`, `npe2`, `numpy`, `pandas`, `psutil`, `pygments`, `pytest`, `rich`, `tensorstore`, `tifffile`, `virtualenv`, `xarray` (#5776)
- [MAINT, packaging] Remove support for briefcase installers (#5804)
- Update `PIP_CONSTRAINT` value to fix failing comprehensive jobs (#5809)
- [pre-commit.ci] pre-commit autoupdate (#5836)
- [pre-commit.ci] pre-commit autoupdate (#5860)
- Fix Dev Docker Container (#5877)
- Make mypy error checking opt-out instead of opt-in (#5885)
- Update Error description when plugin not installed (#5899)
- maint: add fixture to disable throttling (#5908)
- Update upgrade dependecies and test workflows (#5919)
- [Maint] Fix comprehensive tests by skipping labels controls test on py311 pyqt6 (#5922)
- Fix typo in resources/requirements_mypy.in file name (#5924)
- Add Python 3.11 trove classifier. (#5937)
- Change license_file to license_files in setup.cfg (#5948)
- test: [Automatic] Constraints upgrades: `dask`, `fsspec`, `hypothesis`, `imageio`, `ipython`, `napari-plugin-manager`, `napari-svg`, `numpy`, `psygnal`, `pydantic`, `pyqt6`, `pytest`, `rich`, `scikit-image`, `virtualenv`, `zarr` (#5963)


## 38 authors added to this release (alphabetical)

- [Andrea Pierré](https://github.com/napari/napari/commits?author=kir0ul) - @kir0ul
- [Andrew Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Ashley Anderson](https://github.com/napari/napari/commits?author=aganders3) - @aganders3
- [Clement Caporal](https://github.com/napari/napari/commits?author=ClementCaporal) - @ClementCaporal
- [Constantin Pape](https://github.com/napari/napari/commits?author=constantinpape) - @constantinpape
- [Craig T. Russell](https://github.com/napari/napari/commits?author=ctr26) - @ctr26
- [Daniel Althviz Moré](https://github.com/napari/napari/commits?author=dalthviz) - @dalthviz
- [David Ross](https://github.com/napari/napari/commits?author=davidpross) - @davidpross
- [David Stansby](https://github.com/napari/napari/commits?author=dstansby) - @dstansby
- [Gabriel Selzer](https://github.com/napari/napari/commits?author=gselzer) - @gselzer
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Gregor Lichtner](https://github.com/napari/napari/commits?author=glichtner) - @glichtner
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Jan-Hendrik Müller](https://github.com/napari/napari/commits?author=kolibril13) - @kolibril13
- [Jannis Ahlers](https://github.com/napari/napari/commits?author=jnahlers) - @jnahlers
- [Jessy Lauer](https://github.com/napari/napari/commits?author=jeylau) - @jeylau
- [Jonas Windhager](https://github.com/napari/napari/commits?author=jwindhager) - @jwindhager
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kabilar Gunalan](https://github.com/napari/napari/commits?author=kabilar) - @kabilar
- [Katherine Hutchings](https://github.com/napari/napari/commits?author=katherine-hutchings) - @katherine-hutchings
- [Kim Pevey](https://github.com/napari/napari/commits?author=kcpevey) - @kcpevey
- [Konstantin Sofiiuk](https://github.com/napari/napari/commits?author=ksofiyuk) - @ksofiyuk
- [Kyle I. S. Harrington](https://github.com/napari/napari/commits?author=kephale) - @kephale
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [LucaMarconato](https://github.com/napari/napari/commits?author=LucaMarconato) - @LucaMarconato
- [Lucy Liu](https://github.com/napari/napari/commits?author=lucyleeow) - @lucyleeow
- [Mark Harfouche](https://github.com/napari/napari/commits?author=hmaarrfk) - @hmaarrfk
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nadalyn Miller](https://github.com/napari/napari/commits?author=Nadalyn-CZI) - @Nadalyn-CZI
- [Pam Wadhwa](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Paul Smith](https://github.com/napari/napari/commits?author=p-j-smith) - @p-j-smith
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Sean Martin](https://github.com/napari/napari/commits?author=seankmartin) - @seankmartin
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Wouter-Michiel Vierdag](https://github.com/napari/napari/commits?author=melonora) - @melonora


## 40 reviewers added to this release (alphabetical)

- [Alan R Lowe](https://github.com/napari/napari/commits?author=quantumjot) - @quantumjot
- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andrew Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Ashley Anderson](https://github.com/napari/napari/commits?author=aganders3) - @aganders3
- [Charlie Marsh](https://github.com/napari/napari/commits?author=charliermarsh) - @charliermarsh
- [Daniel Althviz Moré](https://github.com/napari/napari/commits?author=dalthviz) - @dalthviz
- [David Ross](https://github.com/napari/napari/commits?author=davidpross) - @davidpross
- [David Stansby](https://github.com/napari/napari/commits?author=dstansby) - @dstansby
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gabriel Selzer](https://github.com/napari/napari/commits?author=gselzer) - @gselzer
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Isabela Presedo-Floyd](https://github.com/napari/napari/commits?author=isabela-pf) - @isabela-pf
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Jan-Hendrik Müller](https://github.com/napari/napari/commits?author=kolibril13) - @kolibril13
- [Jessy Lauer](https://github.com/napari/napari/commits?author=jeylau) - @jeylau
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kim Pevey](https://github.com/napari/napari/commits?author=kcpevey) - @kcpevey
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Konstantin Sofiiuk](https://github.com/napari/napari/commits?author=ksofiyuk) - @ksofiyuk
- [Kyle I. S. Harrington](https://github.com/napari/napari/commits?author=kephale) - @kephale
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [LucaMarconato](https://github.com/napari/napari/commits?author=LucaMarconato) - @LucaMarconato
- [Lucy Liu](https://github.com/napari/napari/commits?author=lucyleeow) - @lucyleeow
- [Lucy Obus](https://github.com/napari/napari/commits?author=LCObus) - @LCObus
- [Mark Harfouche](https://github.com/napari/napari/commits?author=hmaarrfk) - @hmaarrfk
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nathan Clack](https://github.com/napari/napari/commits?author=nclack) - @nclack
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam Wadhwa](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Paul Smith](https://github.com/napari/napari/commits?author=p-j-smith) - @p-j-smith
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Sean Martin](https://github.com/napari/napari/commits?author=seankmartin) - @seankmartin
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Wouter-Michiel Vierdag](https://github.com/napari/napari/commits?author=melonora) - @melonora
- [Ziyang Liu](https://github.com/napari/napari/commits?author=liu-ziyang) - @liu-ziyang

