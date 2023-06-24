# napari 0.4.18

We're happy to announce the release of napari 0.4.18!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights

- Add lasso tool for faster drawing of polygonal Shapes ([#5555](https://github.com/napari/napari/pull/5555))
- Feature: support for textures and vertex colors on Surface layers ([#5642](https://github.com/napari/napari/pull/5642))
- Zooming with the mouse wheel in any mode ([#5701](https://github.com/napari/napari/pull/5701))
- Fix inefficient label mapping in direct color mode (10-20x speedup) ([#5723](https://github.com/napari/napari/pull/5723))
- Efficient labels mapping for drawing in Labels (60 FPS even with 8000x8000 images) ([#5732](https://github.com/napari/napari/pull/5732))

## New Features

- Overlays 2.0 ([#4894](https://github.com/napari/napari/pull/4894))
- expose custom image interpolation kernels ([#5130](https://github.com/napari/napari/pull/5130))
- Add user agent environment variable for pip installations ([#5135](https://github.com/napari/napari/pull/5135))
- Add option to check if plugin try to set viewer attr outside main thread ([#5195](https://github.com/napari/napari/pull/5195))
- Set selection color for QListView item. ([#5202](https://github.com/napari/napari/pull/5202))
- Add warning about set private attr when using proxy ([#5209](https://github.com/napari/napari/pull/5209))
- Shapes interpolation ([#5334](https://github.com/napari/napari/pull/5334))
- Add dask settings to preferences ([#5490](https://github.com/napari/napari/pull/5490))
- Add lasso tool for faster drawing of polygonal Shapes ([#5555](https://github.com/napari/napari/pull/5555))
- Feature: support for textures and vertex colors on Surface layers ([#5642](https://github.com/napari/napari/pull/5642))
- Back point selection with a psygnal Selection ([#5691](https://github.com/napari/napari/pull/5691))
- Zooming with the mouse wheel in any mode ([#5701](https://github.com/napari/napari/pull/5701))
- Add cancellation functionality to progress ([#5728](https://github.com/napari/napari/pull/5728))

## Improvements

- Set keyboard focus on console when opened ([#5208](https://github.com/napari/napari/pull/5208))
- Push variables to console when instantiated ([#5210](https://github.com/napari/napari/pull/5210))
- Tracks layer creation performance improvement ([#5303](https://github.com/napari/napari/pull/5303))
- PERF: Event emissions and perf regression. ([#5307](https://github.com/napari/napari/pull/5307))
- Much faster FormatStringEncoding ([#5315](https://github.com/napari/napari/pull/5315))
- Add parent when creating layer context menu to inherit application theme and add style entry for disabled widgets and menus ([#5381](https://github.com/napari/napari/pull/5381))
- Add correct `enablement` kwarg to `Split Stack` action, `Convert data type` submenu and `Projections` submenu ([#5437](https://github.com/napari/napari/pull/5437))
- Apply disabled widgets style only for menus and set menus styles for `QModelMenu`  and `QMenu` instances ([#5446](https://github.com/napari/napari/pull/5446))
- Add disabled style rule for `QComboBox` following the one for `QPushButton` ([#5469](https://github.com/napari/napari/pull/5469))
- Allow layers control section to resize to contents ([#5474](https://github.com/napari/napari/pull/5474))
- Allow to use `Optional` annotation in function return type for magicgui functions ([#5595](https://github.com/napari/napari/pull/5595))
- Skip equality comparisons in EventedModel when unnecessary ([#5615](https://github.com/napari/napari/pull/5615))
- Bugfix: improve layout of Preferences > Shortcuts tables ([#5679](https://github.com/napari/napari/pull/5679))
- Improve preferences genration ([#5696](https://github.com/napari/napari/pull/5696))
- Add dev example for adding custom overlays. ([#5719](https://github.com/napari/napari/pull/5719))
- Disable buffer swapping ([#5741](https://github.com/napari/napari/pull/5741))
- Explicitly list valid layer names in types ([#5823](https://github.com/napari/napari/pull/5823))
- Sort npe1 widget contributions ([#5865](https://github.com/napari/napari/pull/5865))
- feat: add `since_version` argument of `rename_argument` decorator ([#5910](https://github.com/napari/napari/pull/5910))
- Emit extra information with layer.events.data ([#5967](https://github.com/napari/napari/pull/5967))

## Performance

- Return early when no slicing needed ([#5239](https://github.com/napari/napari/pull/5239))
- Tracks layer creation performance improvement ([#5303](https://github.com/napari/napari/pull/5303))
- PERF: Event emissions and perf regression. ([#5307](https://github.com/napari/napari/pull/5307))
- Much faster FormatStringEncoding ([#5315](https://github.com/napari/napari/pull/5315))
- Fix inefficient label mapping in direct color mode (10-20x speedup) ([#5723](https://github.com/napari/napari/pull/5723))
- Efficient labels mapping for drawing in Labels (60 FPS even with 8000x8000 images) ([#5732](https://github.com/napari/napari/pull/5732))
- Disable buffer swapping ([#5741](https://github.com/napari/napari/pull/5741))

## Bug Fixes

- Warn instead of failing on empty or invalid alt-text ([#4505](https://github.com/napari/napari/pull/4505))
- Fix display of order and scale combinations ([#5004](https://github.com/napari/napari/pull/5004))
- Enforce that contrast limits must be increasing ([#5036](https://github.com/napari/napari/pull/5036))
- Bugfix: Move Window menu to be before Help ([#5093](https://github.com/napari/napari/pull/5093))
- Add extra garbage collection for some viewer tests ([#5108](https://github.com/napari/napari/pull/5108))
- Connect image to plane events and expose them ([#5131](https://github.com/napari/napari/pull/5131))
- Workaround for discover themes from plugins ([#5150](https://github.com/napari/napari/pull/5150))
- Add missed dialogs to `qtbot` in `test_qt_notifications` to prevent segfaults ([#5171](https://github.com/napari/napari/pull/5171))
- DOC Update docstring of `add_dock_widget` & `_add_viewer_dock_widget` ([#5173](https://github.com/napari/napari/pull/5173))
- Fix unsortable features ([#5186](https://github.com/napari/napari/pull/5186))
- Avoid possible divide-by-zero in Vectors layer thumbnail update ([#5192](https://github.com/napari/napari/pull/5192))
- Disable napari-console button when launched from jupyter ([#5213](https://github.com/napari/napari/pull/5213))
- Volume rendering updates for isosurface and attenuated MIP ([#5215](https://github.com/napari/napari/pull/5215))
- Return early when no slicing needed ([#5239](https://github.com/napari/napari/pull/5239))
- Check strictly increasing values when clipping contrast limits to a new range ([#5258](https://github.com/napari/napari/pull/5258))
- UI Bugfix: Make disabled QPushButton more distinct ([#5262](https://github.com/napari/napari/pull/5262))
- Respect background color when calculating scale bar color ([#5270](https://github.com/napari/napari/pull/5270))
- Fix circular import in _vispy module ([#5276](https://github.com/napari/napari/pull/5276))
- Use only data dimensions for cord in status bar ([#5283](https://github.com/napari/napari/pull/5283))
- Prevent obsolete reports about failure of cleaning viewer instances ([#5317](https://github.com/napari/napari/pull/5317))
- Add scikit-image[data] to install_requires, because it's required by builtins ([#5329](https://github.com/napari/napari/pull/5329))
- Fix repeating close dialog on macOS and qt 5.12 ([#5337](https://github.com/napari/napari/pull/5337))
- Disable napari-console if napari launched from vanilla python REPL ([#5350](https://github.com/napari/napari/pull/5350))
- For npe2 plugin, use manifest display_name for File > Open Samples ([#5351](https://github.com/napari/napari/pull/5351))
- Bugfix plugin display_name use (File > Open Sample, Plugin menus) ([#5366](https://github.com/napari/napari/pull/5366))
- Fix editing shape data above 2 dimensions ([#5383](https://github.com/napari/napari/pull/5383))
- Fix test keybinding for layer actions ([#5406](https://github.com/napari/napari/pull/5406))
- fix theme id not being used correctly ([#5412](https://github.com/napari/napari/pull/5412))
- Clarify layer's editable property and separate interaction with visible property ([#5413](https://github.com/napari/napari/pull/5413))
- Fix theme reference to get image for `success_label` style ([#5447](https://github.com/napari/napari/pull/5447))
- Bugfix: Ensure layer._fixed_vertex is set when rotating ([#5449](https://github.com/napari/napari/pull/5449))
- Fix `_n_selected_points` in _layerlist_context.py ([#5450](https://github.com/napari/napari/pull/5450))
- Refactor Main Window status bar to improve information presentation ([#5451](https://github.com/napari/napari/pull/5451))
- Bugfix: Fix test_get_system_theme test for `name` to `id` change ([#5456](https://github.com/napari/napari/pull/5456))
- Bugfix: POLL_INTERVAL_MS used in QTimer needs to be an int on python 3.10 ([#5467](https://github.com/napari/napari/pull/5467))
- Bugfix: Add missing Enums and Flags required by PySide6 > 6.4 ([#5480](https://github.com/napari/napari/pull/5480))
- BugFix: napari does not start with Python v3.11.1: "ValueError: A distribution name is required." ([#5482](https://github.com/napari/napari/pull/5482))
- Fix inverted LUT and blending ([#5487](https://github.com/napari/napari/pull/5487))
- Fix opening file dialogs in PySide ([#5492](https://github.com/napari/napari/pull/5492))
- Handle case when QtDims play thread is partially deleted  ([#5499](https://github.com/napari/napari/pull/5499))
- Ensure surface normals and wireframes are using Models internally ([#5501](https://github.com/napari/napari/pull/5501))
- Recursively check for dependent property to fire events. ([#5528](https://github.com/napari/napari/pull/5528))
- Set PYTHONEXECUTABLE as part of macos fixes on (re)startup ([#5531](https://github.com/napari/napari/pull/5531))
- Un-set unified title and tool bar on mac (Qt property) ([#5533](https://github.com/napari/napari/pull/5533))
- Fix key error issue of action manager ([#5539](https://github.com/napari/napari/pull/5539))
- Bugfix: ensure Checkbox state comparisons are correct by using Qt.CheckState(state) ([#5541](https://github.com/napari/napari/pull/5541))
- Clean dangling widget in test  ([#5544](https://github.com/napari/napari/pull/5544))
- Fix `test_worker_with_progress` by wait on worker end ([#5548](https://github.com/napari/napari/pull/5548))
- Fix min req  ([#5560](https://github.com/napari/napari/pull/5560))
- Fix vispy axes labels ([#5565](https://github.com/napari/napari/pull/5565))
- Fix colormap utils error suggestion code and add a test ([#5571](https://github.com/napari/napari/pull/5571))
- Fix problem of missing plugin widgets after minimize napari ([#5577](https://github.com/napari/napari/pull/5577))
- Make point size isotropic ([#5582](https://github.com/napari/napari/pull/5582))
- Fix guard of qt import in `napari.utils.theme` ([#5593](https://github.com/napari/napari/pull/5593))
- Fix empty shapes layer duplication and `Convert to Labels` enablement logic for selected empty shapes layers ([#5594](https://github.com/napari/napari/pull/5594))
- Stop using removed multichannel= kwarg to skimage functions ([#5596](https://github.com/napari/napari/pull/5596))
- Add information about `syntax_style` value in error message for theme validation ([#5602](https://github.com/napari/napari/pull/5602))
- Remove catch_warnings in slicing ([#5603](https://github.com/napari/napari/pull/5603))
- Incorret theme should not prevent napari from start ([#5605](https://github.com/napari/napari/pull/5605))
- Unblock axis labels event to be emitted when slider label changes ([#5631](https://github.com/napari/napari/pull/5631))
- Bugfix: IndexError slicing Surface with higher-dimensional vertex_values ([#5635](https://github.com/napari/napari/pull/5635))
- Bugfix: Convert Viewer Delete button to QtViewerPushButton with action and shortcut ([#5636](https://github.com/napari/napari/pull/5636))
- Change dim `axis_label` resize logic to set width using only displayed labels width ([#5640](https://github.com/napari/napari/pull/5640))
- Feature: support for textures and vertex colors on Surface layers ([#5642](https://github.com/napari/napari/pull/5642))
- Fix features issues with init param and property setter ([#5646](https://github.com/napari/napari/pull/5646))
- Bugfix: Don't double toggle visibility for linked layers ([#5656](https://github.com/napari/napari/pull/5656))
- Bugfix: ensure pan/zoom buttons work, along with spacebar keybinding ([#5669](https://github.com/napari/napari/pull/5669))
- Bugfix: Add Tracks to qt_keyboard_settings ([#5678](https://github.com/napari/napari/pull/5678))
- Fix automatic naming and GUI exposure of multiple unnamed colormaps ([#5682](https://github.com/napari/napari/pull/5682))
- Fix mouse movement handling for `TransformBoxOverlay` ([#5692](https://github.com/napari/napari/pull/5692))
- Update environment.yml ([#5693](https://github.com/napari/napari/pull/5693))
- Resolve symlinks from path to environment for setting path ([#5704](https://github.com/napari/napari/pull/5704))
- Fix tracks color-by when properties change ([#5708](https://github.com/napari/napari/pull/5708))
- Fix Sphinx warnings ([#5717](https://github.com/napari/napari/pull/5717))
- Do not use depth for canvas overlays; allow setting blending mode for overlays ([#5720](https://github.com/napari/napari/pull/5720))
- Unify event behaviour for points and its qt controls ([#5722](https://github.com/napari/napari/pull/5722))
- Fix camera 3D absolute rotation bug ([#5726](https://github.com/napari/napari/pull/5726))
- Maint: Bump mypy ([#5727](https://github.com/napari/napari/pull/5727))
- Style `QGroupBox` indicator ([#5729](https://github.com/napari/napari/pull/5729))
- Fix centering of non-displayed dimensions ([#5736](https://github.com/napari/napari/pull/5736))
- Don't attempt to use npe1 readers in napari.plugins._npe2.read ([#5739](https://github.com/napari/napari/pull/5739))
- Prevent canvas micro-panning on point add ([#5742](https://github.com/napari/napari/pull/5742))
- Use text opacity to signal that widget is disabled ([#5745](https://github.com/napari/napari/pull/5745))
- Bugfix: Add the missed keyReleaseEvent method in QtViewerDockWidget ([#5746](https://github.com/napari/napari/pull/5746))
- Update status bar on active layer change ([#5754](https://github.com/napari/napari/pull/5754))
- Use array size directly when checking multiscale arrays to prevent overflow ([#5759](https://github.com/napari/napari/pull/5759))
- Fix path to `check_updated_packages.py` ([#5762](https://github.com/napari/napari/pull/5762))
- Brush cursor implementation using an overlay ([#5763](https://github.com/napari/napari/pull/5763))
- Bugfix: force a redraw to ensure highlight shows when Points are select-all selected ([#5771](https://github.com/napari/napari/pull/5771))
- Fix copy/paste of points ([#5795](https://github.com/napari/napari/pull/5795))
- Fix multiple viewer example ([#5796](https://github.com/napari/napari/pull/5796))
- Fix colormapping nD images ([#5805](https://github.com/napari/napari/pull/5805))
- Set focus policy for mainwindow to prevent keeping focus on the axis labels (and other `QLineEdit` based widgets) when clicking outside the widget ([#5812](https://github.com/napari/napari/pull/5812))
- Enforce Points.selected_data type as Selection ([#5813](https://github.com/napari/napari/pull/5813))
- Change toggle menubar visibility functionality to hide menubar and show it on mouse movement validation ([#5824](https://github.com/napari/napari/pull/5824))
- Bugfix: Disconnect callbacks on object deletion in special functions from `event_utils`  ([#5826](https://github.com/napari/napari/pull/5826))
- Do not blend color in QtColorBox with black using opacity ([#5827](https://github.com/napari/napari/pull/5827))
- Don't allow negative contour values ([#5830](https://github.com/napari/napari/pull/5830))
- Bugfixes for layer overlays: clean up when layer is removed + fix potential double creation ([#5831](https://github.com/napari/napari/pull/5831))
- Add compatibility to PySide in file dialogs by using positional arguments ([#5834](https://github.com/napari/napari/pull/5834))
- Bugfix: fix broken "show selected" in the Labels layer (because of caching) ([#5841](https://github.com/napari/napari/pull/5841))
- Add tests for popup widgets and fix perspective popup slider initialization ([#5848](https://github.com/napari/napari/pull/5848))
- [Qt6] Fix AttributeError on renaming layer ([#5850](https://github.com/napari/napari/pull/5850))
- Bugfix: Ensure QTableWidgetItem(action.description) item is enabled ([#5854](https://github.com/napari/napari/pull/5854))
- Add constraints file during installation of packages from pip in docs workflow ([#5862](https://github.com/napari/napari/pull/5862))
- Bugfix: link the Labels model to the "show selected" checkbox ([#5867](https://github.com/napari/napari/pull/5867))
- Add `__all__` to `napari/types.py` ([#5894](https://github.com/napari/napari/pull/5894))
- Fix drawing vertical or horizontal line segments in Shapes layer ([#5895](https://github.com/napari/napari/pull/5895))
- Disallow outside screen geometry napari window position ([#5915](https://github.com/napari/napari/pull/5915))
- Fix `napari-svg` version parsing in `conftest.py` ([#5947](https://github.com/napari/napari/pull/5947))
- Fix issue in utils.progress for disable=True ([#5964](https://github.com/napari/napari/pull/5964))
- Set high DPI attributes when using PySide2 ([#5968](https://github.com/napari/napari/pull/5968))
- [0.4.18rc1] Bugfix/event proxy ([#5994](https://github.com/napari/napari/pull/5994))

## API Changes

- Overlays 2.0 ([#4894](https://github.com/napari/napari/pull/4894))
- expose custom image interpolation kernels ([#5130](https://github.com/napari/napari/pull/5130))
- Connect image to plane events and expose them ([#5131](https://github.com/napari/napari/pull/5131))

## Deprecations


## Build Tools

- ci(dependabot): bump styfle/cancel-workflow-action from 0.10.0 to 0.10.1 ([#5158](https://github.com/napari/napari/pull/5158))
- ci(dependabot): bump actions/checkout from 2 to 3 ([#5160](https://github.com/napari/napari/pull/5160))
- ci(dependabot): bump styfle/cancel-workflow-action from 0.10.1 to 0.11.0 ([#5290](https://github.com/napari/napari/pull/5290))
- ci(dependabot): bump docker/login-action from 2.0.0 to 2.1.0 ([#5291](https://github.com/napari/napari/pull/5291))
- ci(dependabot): bump actions/upload-artifact from 2 to 3 ([#5292](https://github.com/napari/napari/pull/5292))
- Pin mypy version ([#5310](https://github.com/napari/napari/pull/5310))
- MAINT: Start testing on Python 3.11 in CI. ([#5439](https://github.com/napari/napari/pull/5439))
- Pin test dependencies ([#5715](https://github.com/napari/napari/pull/5715))

## Documentation

- Fix failure on benchmark reporting ([#5083](https://github.com/napari/napari/pull/5083))
- Add NAP-5: proposal for an updated napari logo ([#5084](https://github.com/napari/napari/pull/5084))
- DOC Update doc contributing guide ([#5114](https://github.com/napari/napari/pull/5114))
- Napari debugging during plugin development documentation ([#5142](https://github.com/napari/napari/pull/5142))
- DOC Update docstring of `add_dock_widget` & `_add_viewer_dock_widget` ([#5173](https://github.com/napari/napari/pull/5173))
- Specified that the path is to the local folder in contributing documentation guide. ([#5191](https://github.com/napari/napari/pull/5191))
- Fixes broken links in latest docs version ([#5193](https://github.com/napari/napari/pull/5193))
- Fixes gallery ToC ([#5458](https://github.com/napari/napari/pull/5458))
- Fix broken link in EmitterGroup docstring ([#5465](https://github.com/napari/napari/pull/5465))
- Fix Sphinx warnings ([#5717](https://github.com/napari/napari/pull/5717))
- Add Fourier transform playground example ([#5872](https://github.com/napari/napari/pull/5872))
- Improve documentation of `changed` event in EventedList ([#5928](https://github.com/napari/napari/pull/5928))
- Set removal version in deprecation of Viewer.rounded_division ([#5944](https://github.com/napari/napari/pull/5944))
- Update docs using changes from napari/docs ([#5979](https://github.com/napari/napari/pull/5979))
- Pre commit fixes for 0.4.18 release branch ([#5985](https://github.com/napari/napari/pull/5985))
- Add favicon and configuration ([#4](https://github.com/napari/docs/pull/4))
- Docs for  5195 from main repository ([#7](https://github.com/napari/docs/pull/7))
- Use `imshow` in `getting_started` ([#9](https://github.com/napari/docs/pull/9))
- DOC Update `viewer.md` ([#11](https://github.com/napari/docs/pull/11))
- Add and/or update documentation alt text ([#12](https://github.com/napari/docs/pull/12))
- Adding documents and images from January 2022 plugin testing workshop. ([#35](https://github.com/napari/docs/pull/35))
- Add some more docs about packaging details and conda-forge releases ([#48](https://github.com/napari/docs/pull/48))
- Add documentation on using virtual environments for testing in napari based on 2022-01 workshop by Talley Lambert ([#50](https://github.com/napari/docs/pull/50))
- Added info for conda installation problems ([#51](https://github.com/napari/docs/pull/51))
- add best practices about packaging ([#52](https://github.com/napari/docs/pull/52))
- Update viewer tutorial, regarding the console button ([#53](https://github.com/napari/docs/pull/53))
- add sample database page ([#56](https://github.com/napari/docs/pull/56))
- Fix magicgui objects.inv url for intersphinx ([#58](https://github.com/napari/docs/pull/58))
- Fix broken links ([#59](https://github.com/napari/docs/pull/59))
- Add sphinx-design cards to Usage landing page ([#63](https://github.com/napari/docs/pull/63))
- Update to napari viewer tutorial. ([#65](https://github.com/napari/docs/pull/65))
- Added environment creation and doc tools install ([#72](https://github.com/napari/docs/pull/72))
- Feature: add `copy` button for code blocks using `sphinx-copybutton` ([#76](https://github.com/napari/docs/pull/76))
- Add NAP-6 - Proposal for contributable menus ([#77](https://github.com/napari/docs/pull/77))
- Update contributing docs for [dev] install change needing Qt backend install ([#78](https://github.com/napari/docs/pull/78))
- Update theme related documentation ([#81](https://github.com/napari/docs/pull/81))
- Feature: implement python version substitution in conf.py ([#84](https://github.com/napari/docs/pull/84))
- Fixes gallery ToC ([#85](https://github.com/napari/docs/pull/85))
- Clarify arm64 macOS (Apple Silicon) installation ([#89](https://github.com/napari/docs/pull/89))
- Add cards to usage landing pages ([#97](https://github.com/napari/docs/pull/97))
- Replace pip with python -m pip ([#100](https://github.com/napari/docs/pull/100))
- change blob example to be self contained ([#101](https://github.com/napari/docs/pull/101))
- Home page update, take 2 ([#102](https://github.com/napari/docs/pull/102))
- Update the 'ensuring correctness' mission clause ([#105](https://github.com/napari/docs/pull/105))
- Update steering council listing on website ([#106](https://github.com/napari/docs/pull/106))
- Update version switcher json ([#109](https://github.com/napari/docs/pull/109))
- Installation: Add libmamba solver to conda Note ([#110](https://github.com/napari/docs/pull/110))
- Update requirements and config for sphinx-favicon for 1.0 ([#116](https://github.com/napari/docs/pull/116))
- change print to f-string ([#117](https://github.com/napari/docs/pull/117))
- Replace non-breaking spaces with regular spaces ([#118](https://github.com/napari/docs/pull/118))
- Bugfix: documentation update for napari PR #5636 ([#123](https://github.com/napari/docs/pull/123))
- Add matplotlib image scraper for gallery ([#130](https://github.com/napari/docs/pull/130))
- Fix missing links and references ([#133](https://github.com/napari/docs/pull/133))
- Update URL of version switcher ([#139](https://github.com/napari/docs/pull/139))
- Harmonize release notes to new mandatory labels ([#141](https://github.com/napari/docs/pull/141))
- Update installation docs to remove briefcase bundle mentions ([#147](https://github.com/napari/docs/pull/147))
- Fix version switcher URL for the latest docs version ([#148](https://github.com/napari/docs/pull/148))
- Update Shapes How-To for new Lasso tool (napari/#5555) ([#149](https://github.com/napari/docs/pull/149))
- Fix signpost to make_napari_viewer code ([#151](https://github.com/napari/docs/pull/151))
- Docs for adding LayerData tuple to viewer ([#152](https://github.com/napari/docs/pull/152))
- Update viewer tutorial 3D mode docs ([#159](https://github.com/napari/docs/pull/159))
- Add a napari plugin debugging quick start section to the debugging guide ([#161](https://github.com/napari/docs/pull/161))
- Pin npe2 version to match installed one ([#175](https://github.com/napari/docs/pull/175))
- Add Wouter-Michiel Vierdag to list of core devs ([#181](https://github.com/napari/docs/pull/181))

## Other Pull Requests

- use app-model for view menu ([#4826](https://github.com/napari/napari/pull/4826))
- Overlay backend refactor ([#4907](https://github.com/napari/napari/pull/4907))
- Migrate help menu to use app model ([#4922](https://github.com/napari/napari/pull/4922))
- Refactor layer slice/dims/view/render state ([#5003](https://github.com/napari/napari/pull/5003))
- MAINT: increase min numpy version. ([#5089](https://github.com/napari/napari/pull/5089))
- Refactor qt notification and its test solve problem of segfaults ([#5138](https://github.com/napari/napari/pull/5138))
- Decouple changing viewer.theme from changing theme settings/preferences ([#5143](https://github.com/napari/napari/pull/5143))
- [DOCS] misc invalid syntax updates. ([#5176](https://github.com/napari/napari/pull/5176))
- MAINT: remove vendored colorconv from skimage. ([#5180](https://github.com/napari/napari/pull/5180))
- Re-add README screenshot ([#5220](https://github.com/napari/napari/pull/5220))
- MAINT: remove requirements.txt and cache actions based on setup.cfg. ([#5234](https://github.com/napari/napari/pull/5234))
- Explicitly set test array data to fix a flaky test ([#5245](https://github.com/napari/napari/pull/5245))
- Add ruff linter to pre-commit ([#5275](https://github.com/napari/napari/pull/5275))
- Run tests on release branch ([#5277](https://github.com/napari/napari/pull/5277))
- Vispy 0.12: per-point symbol and several bugfixes ([#5312](https://github.com/napari/napari/pull/5312))
- Make all imports absolute ([#5318](https://github.com/napari/napari/pull/5318))
- Fix track ids features ordering for unordered tracks ([#5320](https://github.com/napari/napari/pull/5320))
- tests: remove private magicgui access in tests ([#5331](https://github.com/napari/napari/pull/5331))
- Make settings and cache separate per each environment.  ([#5333](https://github.com/napari/napari/pull/5333))
- Remove internal event connection on SelectableEventedList ([#5339](https://github.com/napari/napari/pull/5339))
- Unset `PYTHON*` vars and use entitlements in macOS conda menu shortcut ([#5354](https://github.com/napari/napari/pull/5354))
- Distinguish between update_dims, extent changes, and refresh ([#5363](https://github.com/napari/napari/pull/5363))
- Add checks for pending Qt threads and timers in tests ([#5373](https://github.com/napari/napari/pull/5373))
- Suppress color conversion warning when converting invalid LAB coordinates ([#5386](https://github.com/napari/napari/pull/5386))
- Fix warning when fail to import qt binding. ([#5388](https://github.com/napari/napari/pull/5388))
- Update `MANIFEST.in` to remove warning when run tox ([#5393](https://github.com/napari/napari/pull/5393))
- [Automatic] Update albertosottile/darkdetect vendored module ([#5394](https://github.com/napari/napari/pull/5394))
- Update citation metadata ([#5398](https://github.com/napari/napari/pull/5398))
- Feature: making the Help menu more helpful via weblinks (re-do of #5094) ([#5399](https://github.com/napari/napari/pull/5399))
- [pre-commit.ci] pre-commit autoupdate ([#5403](https://github.com/napari/napari/pull/5403))
- Fix flaky dims playback test by waiting for playing condition ([#5414](https://github.com/napari/napari/pull/5414))
- [Automatic] Update albertosottile/darkdetect vendored module ([#5416](https://github.com/napari/napari/pull/5416))
- [pre-commit.ci] pre-commit autoupdate ([#5422](https://github.com/napari/napari/pull/5422))
- Avoid setting corner pixels for empty layers ([#5423](https://github.com/napari/napari/pull/5423))
- Maint: Typing and ImportError -> ModuleNotFoundError. ([#5431](https://github.com/napari/napari/pull/5431))
- Fix tox `passenv` setup for `DISPLAY` and `XAUTHORITY` environment variables ([#5441](https://github.com/napari/napari/pull/5441))
- Add `error` color to themes and change application close/exit dialogs ([#5442](https://github.com/napari/napari/pull/5442))
- Update screenshot in readme ([#5452](https://github.com/napari/napari/pull/5452))
- Maint: Fix sporadic QtDims garbage collection failures by converting some stray references to weakrefs ([#5471](https://github.com/napari/napari/pull/5471))
- Replace GabrielBB/xvfb-action ([#5478](https://github.com/napari/napari/pull/5478))
- Add tags to recently added examples ([#5486](https://github.com/napari/napari/pull/5486))
- Remove layer ndisplay event ([#5491](https://github.com/napari/napari/pull/5491))
- MAINT: Don't format logs in log call ([#5504](https://github.com/napari/napari/pull/5504))
- Replace flake8, isort and pyupgrade by ruff, enable additional usefull rules  ([#5513](https://github.com/napari/napari/pull/5513))
- Second PR that enables more ruff rules.  ([#5520](https://github.com/napari/napari/pull/5520))
- Use pytest-pretty for better log readability  ([#5525](https://github.com/napari/napari/pull/5525))
- MAINT: Follow Nep29, bump minimum numpy. ([#5532](https://github.com/napari/napari/pull/5532))
- [pre-commit.ci] pre-commit autoupdate ([#5534](https://github.com/napari/napari/pull/5534))
- Move layer editable change from slicing to controls ([#5546](https://github.com/napari/napari/pull/5546))
- update conda_menu_config.json for latest fixes in menuinst ([#5564](https://github.com/napari/napari/pull/5564))
- Enable the `COM` and `SIM` rules in `ruff` configuration ([#5566](https://github.com/napari/napari/pull/5566))
- Move from ubuntu 18.04 to ubuntu 20.04 in workflows ([#5578](https://github.com/napari/napari/pull/5578))
- FIX: Fix --pre skimage that have a more precise warning message ([#5580](https://github.com/napari/napari/pull/5580))
- Remove leftover duplicated code ([#5586](https://github.com/napari/napari/pull/5586))
- Remove napari-hub API access code ([#5587](https://github.com/napari/napari/pull/5587))
- Enable `ruff` rules part 4. ([#5590](https://github.com/napari/napari/pull/5590))
- [pre-commit.ci] pre-commit autoupdate ([#5592](https://github.com/napari/napari/pull/5592))
- Maint: ImportError -> ModuleNotFoundError. ([#5628](https://github.com/napari/napari/pull/5628))
- [pre-commit.ci] pre-commit autoupdate ([#5645](https://github.com/napari/napari/pull/5645))
- MAINT: Do not use mutable default for dataclass. ([#5647](https://github.com/napari/napari/pull/5647))
- MAINT: Do not use cgi-traceback on 3.11+ (deprecated, marked for removal) ([#5648](https://github.com/napari/napari/pull/5648))
- MAINT: Add explicit level to warn. ([#5649](https://github.com/napari/napari/pull/5649))
- MAINT: Split test file in two to find hanging test. ([#5680](https://github.com/napari/napari/pull/5680))
- Skip pyside6 version 6.4.3 for tests ([#5683](https://github.com/napari/napari/pull/5683))
- Pin pydantic. ([#5695](https://github.com/napari/napari/pull/5695))
- fix test_viewer_open_no_plugin exception message expectation ([#5698](https://github.com/napari/napari/pull/5698))
- fix: Block PySide6==6.5.0 in tests ([#5702](https://github.com/napari/napari/pull/5702))
- Don't resize shape after `Shift` release until mouse moves ([#5707](https://github.com/napari/napari/pull/5707))
- Update `test_examples` job dependencies, unskip `surface_timeseries_.py` and update some examples validations ([#5716](https://github.com/napari/napari/pull/5716))
- Add test to check basic interactions with layer controls widgets ([#5757](https://github.com/napari/napari/pull/5757))
- test: [Automatic] Constraints upgrades: `dask`, `hypothesis`, `imageio`, `npe2`, `numpy`, `pandas`, `psutil`, `pygments`, `pytest`, `rich`, `tensorstore`, `tifffile`, `virtualenv`, `xarray` ([#5776](https://github.com/napari/napari/pull/5776))
- [MAINT, packaging] Remove support for briefcase installers ([#5804](https://github.com/napari/napari/pull/5804))
- Update `PIP_CONSTRAINT` value to fix failing comprehensive jobs ([#5809](https://github.com/napari/napari/pull/5809))
- [pre-commit.ci] pre-commit autoupdate ([#5836](https://github.com/napari/napari/pull/5836))
- [pre-commit.ci] pre-commit autoupdate ([#5860](https://github.com/napari/napari/pull/5860))
- Fix Dev Docker Container ([#5877](https://github.com/napari/napari/pull/5877))
- Make mypy error checking opt-out instead of opt-in ([#5885](https://github.com/napari/napari/pull/5885))
- Update Error description when plugin not installed ([#5899](https://github.com/napari/napari/pull/5899))
- maint: add fixture to disable throttling ([#5908](https://github.com/napari/napari/pull/5908))
- Update upgrade dependecies and test workflows ([#5919](https://github.com/napari/napari/pull/5919))
- [Maint] Fix comprehensive tests by skipping labels controls test on py311 pyqt6 ([#5922](https://github.com/napari/napari/pull/5922))
- Fix typo in resources/requirements_mypy.in file name ([#5924](https://github.com/napari/napari/pull/5924))
- Add Python 3.11 trove classifier. ([#5937](https://github.com/napari/napari/pull/5937))
- Change license_file to license_files in setup.cfg ([#5948](https://github.com/napari/napari/pull/5948))
- test: [Automatic] Constraints upgrades: `dask`, `fsspec`, `hypothesis`, `imageio`, `ipython`, `napari-plugin-manager`, `napari-svg`, `numpy`, `psygnal`, `pydantic`, `pyqt6`, `pytest`, `rich`, `scikit-image`, `virtualenv`, `zarr` ([#5963](https://github.com/napari/napari/pull/5963))
- Update deprecation information ([#5984](https://github.com/napari/napari/pull/5984))


## 38 authors added to this release (alphabetical)

- [Andrea Pierré](https://github.com/napari/napari/commits?author=kir0ul) - @kir0ul
- [Andrew Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Ashley Anderson](https://github.com/napari/napari/commits?author=aganders3) - @aganders3
- [Clément Caporal](https://github.com/napari/napari/commits?author=ClementCaporal) - @ClementCaporal
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


## 41 reviewers added to this release (alphabetical)

- [Alan R Lowe](https://github.com/napari/napari/commits?author=quantumjot) - @quantumjot
- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andrew Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Ashley Anderson](https://github.com/napari/napari/commits?author=aganders3) - @aganders3
- [Charlie Marsh](https://github.com/napari/napari/commits?author=charliermarsh) - @charliermarsh
- [Daniel Althviz Moré](https://github.com/napari/napari/commits?author=dalthviz) - @dalthviz
- [David Ross](https://github.com/napari/napari/commits?author=davidpross) - @davidpross
- [David Stansby](https://github.com/napari/napari/commits?author=dstansby) - @dstansby
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Eric Perlman](https://github.com/napari/napari/commits?author=perlman) - @perlman
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


## 19 docs authors added to this release (alphabetical)

- [Ashley Anderson](https://github.com/napari/napari/commits?author=aganders3) - @aganders3
- [chili-chiu](https://github.com/napari/napari/commits?author=chili-chiu) - @chili-chiu
- [Christopher Nauroth-Kreß](https://github.com/napari/napari/commits?author=Chris-N-K) - @Chris-N-K
- [Curtis Rueden](https://github.com/napari/napari/commits?author=ctrueden) - @ctrueden
- [Daniel Althviz Moré](https://github.com/napari/napari/commits?author=dalthviz) - @dalthviz
- [David Stansby](https://github.com/napari/napari/commits?author=dstansby) - @dstansby
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lucy Liu](https://github.com/napari/napari/commits?author=lucyleeow) - @lucyleeow
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nadalyn Miller](https://github.com/napari/napari/commits?author=Nadalyn-CZI) - @Nadalyn-CZI
- [Oren Amsalem](https://github.com/napari/napari/commits?author=orena1) - @orena1
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Sean Martin](https://github.com/napari/napari/commits?author=seankmartin) - @seankmartin
- [Wouter-Michiel Vierdag](https://github.com/napari/napari/commits?author=melonora) - @melonora


## 20 docs reviewers added to this release (alphabetical)

- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andrew Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Ashley Anderson](https://github.com/napari/napari/commits?author=aganders3) - @aganders3
- [Christopher Nauroth-Kreß](https://github.com/napari/napari/commits?author=Chris-N-K) - @Chris-N-K
- [David Stansby](https://github.com/napari/napari/commits?author=dstansby) - @dstansby
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lucy Liu](https://github.com/napari/napari/commits?author=lucyleeow) - @lucyleeow
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nadalyn Miller](https://github.com/napari/napari/commits?author=Nadalyn-CZI) - @Nadalyn-CZI
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Sean Martin](https://github.com/napari/napari/commits?author=seankmartin) - @seankmartin
- [Wouter-Michiel Vierdag](https://github.com/napari/napari/commits?author=melonora) - @melonora

