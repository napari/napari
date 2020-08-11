# napari 0.3.6

We're happy to announce the release of napari 0.3.6!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights
This release contains the long awaited addition of text to both the points and
shapes layers (#1374). Checkout our `examples/*_with_text.py` for simple usage
and this [segmentation annotation tutorial](https://napari.org/tutorials/applications/annotate_segmentation)
for a more real-world use case.

We've added support for a circular paintbrush for easier labels painting,
and moved more of our contrast limits and gamma setting to the GPU for faster
rendering and interactivity with 3D rendered datasets. As always this release
contains various bug fixes and other improvements, including some automated
reformatting and fixes to our docstrings.

This release also contains a number of contributions from new authors thanks
to the SciPy conference sprints. We’re delighted to welcome new contributors to
the codebase. If you want help contributing to napari, reach out to us on our chat
room at https://napari.zulipchat.com!


## New Features
- Functions to split/combine multiple layers along an axis (#1322)
- Add text to shapes and points via TextManager (#1374)
- Add circle/spherical brush to Labels paint brush (#1429)


## Improvements
- Add ability to run napari script with/without gui_qt from CLI (#1373)
- Event handler refactor for image layer  "reverted by (#1416)" (#1376)
- Add ndim as a keyword argument to shapes to support creating empty layers (#1379)
- Add helpful error on multichannel IndexError (#1381)
- Use qlistwidget for QtLayerList "reverted by (#1416)" (#1391)
- Event handler surface layer  "reverted by (#1416)" (#1396)
- Revert "event handler refactors (#1376), (#1391), (#1396)" (#1416)
- Move contrast limits and gamma to the shader by vendoring vispy code (#1456)
- Reduce attenuation default and range (#1460)


## Bug Fixes
- Revert "remove scipy.stats import (#1250)" (#1371)
- Fix vispy volume colormap changing (#1402)
- Fix vertical alignment of QLabel in QtDimSliderWidget (#1415)
- Fix adding single shape duplicating properties (#1427)
- Fix viewing properties for multiscale labels layers (#1433)
- Fix trim of layer number (#1439)
- Hide status bar line on windows (#1440)
- Handle nested zarr path (#1441)
- Toggle image/ volume nodes on ndisplay change (#1445)
- Fix attenuation setting (#1454)
- Fix toggle ndisplay z-order points (#1463)
- Fix edge color select bug (#1464)
- Proper recognition tiff files with ".TIF" extension (#1472)


## Build Tools
- Updates plugin dev docs to encourage github topic (#1366)
- Refactor all tests to hide GUI (#1372)
- Fix two remaining tests that try to show the viewer. (#1375)
- Fix windows tests (#1377)
- Dedicated testing doc (#1378)
- Rename mark to avoid warning on pytest ordering package (#1383)
- Rename viewer_factory -> make_test_viewer, don't return view (#1386)
- Fix inconsistency in docs (#1420)
- Docreformat (#1428)
- DOC: autoreformat of more docstrings. (#1437)
- Fix typos using codespell (#1438)
- Preserve tests from EVH revert (#1452)
- Fix typos (#1468)
- DOC: minor doc reformatting. (#1469)
- DOC: update param names to match function signature. (#1479)
- Fix events docstring type (#1481)
- Don't look for release notes in pre-releases (#1483)


## 14 authors added to this release (alphabetical)

- [Cameron Lloyd](https://github.com/napari/napari/commits?author=camlloyd) - @camlloyd
- [Chris Wood](https://github.com/napari/napari/commits?author=cwood1967) - @cwood1967
- [Draga Doncila](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Forrest Li](https://github.com/napari/napari/commits?author=floryst) - @floryst
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Hector Munoz](https://github.com/napari/napari/commits?author=hectormz) - @hectormz
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justin Kiggins](https://github.com/napari/napari/commits?author=neuromusic) - @neuromusic
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Trevor Manz](https://github.com/napari/napari/commits?author=manzt) - @manzt
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi


## 12 reviewers added to this release (alphabetical)

- [Chris Wood](https://github.com/napari/napari/commits?author=cwood1967) - @cwood1967
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Hector Munoz](https://github.com/napari/napari/commits?author=hectormz) - @hectormz
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justin Kiggins](https://github.com/napari/napari/commits?author=neuromusic) - @neuromusic
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Trevor Manz](https://github.com/napari/napari/commits?author=manzt) - @manzt
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi
