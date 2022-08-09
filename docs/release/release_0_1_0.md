# napari 0.1.0

We're happy to announce the release of napari 0.1.0! napari is a fast,
interactive, multi-dimensional image viewer for Python. It's designed for
browsing, annotating, and analyzing large multi-dimensional images. It's built
on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the
scientific Python stack (numpy, scipy).

This is our first minor release, timed for the 2019 SciPy Conference in Austin.
It marks our transition from pre-alpha to alpha, and establishes a reasonable
API for adding images, shapes, and other basic layer types to an interactive
viewer. It supports launching a viewer with python scripting or from Jupyter
notebooks.

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Pull Requests

- Add shapes ({pr}`100`)
- Vectors Layer (#129)
- setup css basics (#167)
- Update shields on README (#169)
- Add dimension sliders (#171)
- New more convenient ViewerApp API (#172)
- WIP: Labels layer (#175)
- automate running/testing of examples (#176)
- fix error with nD markers (#183)
- fix sphinx-apidoc command on CONTRIBUTING.MD (#187)
- Allow other array-like data (#188)
- Add example using a Zarr array (#191)
- readme updates (#192)
- fix empty markers (#194)
- Interactive labels (#195)
- improve layer selection / name editing (#196)
- Rasterize shapes (#197)
- Fix color map to work with unlimited labels. (#203)
- Make clim range an input argument (#205)
- Use triangles for vectors (#215)
- support clipping appropriately (#216)
- fix nD fill (#217)
- fix nD paint (#218)
- vectors layer speed up (#219)
- Generate svg from shapes layer (#220)
- allow for other arrays used with labels (#228)
- fix drawing lines in shapes layer (#230)
- fix empty polygons in shapes layer (#231)
- add support for custom key bindings (#232)
- Deduplicate marker symbols (#233)
- switch to qtpy (#235)
- new styles (#236)
- add support for settable viewer title (#237)
- fix escape selecting shape error (#242)
- updated screenshots (#244)
- unify titlebar (#245)
- refactor layers list (#246)
- remove app (#247)
- fix range slider imports (#248)
- Refactor layer indices and coords (#249)
- remove example data utils file (#250)
- theme setting (#253)
- layer active when only one selected (#257)
- tiny theme related fixes (#258)
- Viewer to svg (#259)
- Fix svg canvas (#264)
- remove async utils (#267)
- refactor draggable layers (#271)
- fix bbox call on new markers (#272)
- fix default selection logic (#273)
- add layer viewer update events (#274)
- flip markers (#275)
- fix markers sizing (#276)
- fix blending update (#277)
- fix int clim value (#278)
- remove viewer from individual layer object (#279)
- Black formatter PR (#282)
- revert layers list (#284)
- fix image dims update (#288)
- fix status updates on dims changes (#289)
- Refactor viewer syntax (#290)
- Simplify the colormaps list and add the single-color colormaps (#291)
- Remove qtviewer from viewer (#292)
- refactor theme setting (#293)
- Pyramid layer (#295)
- Nd shapes (#297)
- WIP: Stop using add_to_viewer syntax for basic layers (#303)
- fix click on layer list (#306)
- use stylesheet for styling of range slider (#307)
- Unify layer mode Enums (#311)
- Thumbnails (#314)
- fix remove layer (#315)
- Blending Enum (#317)
- Change Image.interpolation to Enum (#319)
- Reformatting whole repo with Black (#322)
- Nd pyramids (#323)
- Expand button (#324)
- Revert "Reformatting whole repo with Black" (#327)
- Black Format CI task (#329)
- Refactor layer qt properties and controls (#330)
- black format pyramid examples (#333)
- fix labels colormap (#340)
- fix black ignore (#341)
- Nd vectors (#343)
- remove broadcast from shapes (#345)
- fix layer select styling (#347)
- Change layers.Markers to layers.Points (#348)
- add points thumbnail (#352)
- fix resource compiling instructions (#353)
- Improve resource building contrib (#354)
- Add menubar to napari main window (#356)
- fix selected default (#361)
- Add dims test and fix 5D images (#362)
- Shape thumbnails (#364)
- [FIX] setting remote upstream in contributing guidelines  (#366)
- Refactor thumbnail type conversion (#370)
- Selectable points (#371)
- Test layers list model and view (#373)
- vectors thumbnails (#377)
- add drag and drop (#378)
- standardize keybindings framework (#389)
- Refactor directory structure (#390)
- Test image and pyramid layers (#391)
- Rename app_context gui_qt (#392)
- Test labels layer (#393)
- Test points layer (#394)
- Test vectors (#396)
- modified multiple images overlaid figure (#399)
- Test shapes (#400)
- add viewer model tests (#401)
- update readme for alpha release (#402)

## 12 authors added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Bryant Chhun](https://github.com/napari/napari/commits?author=bryantChhun) - @bryantChhun
- [Eric Perlman](https://github.com/napari/napari/commits?author=perlman) - @perlman
- [John Kirkham](https://github.com/napari/napari/commits?author=jakirkham) - @jakirkham
- [Jeremy Freeman](https://github.com/napari/napari/commits?author=freeman-lab) - @freeman-lab
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Loic Royer](https://github.com/napari/napari/commits?author=royerloic) - @royerloic
- [Mars Huang](https://github.com/napari/napari/commits?author=marshuang80) - @marshuang80
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pranathi Vemuri](https://github.com/napari/napari/commits?author=pranathivemuri) - @pranathivemuri

## 10 reviewers added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Bryant Chhun](https://github.com/napari/napari/commits?author=bryantChhun) - @bryantChhun
- [Charlotte Weaver](https://github.com/napari/napari/commits?author=csweaver) - @csweaver
- [Jeremy Freeman](https://github.com/napari/napari/commits?author=freeman-lab) - @freeman-lab
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Loic Royer](https://github.com/napari/napari/commits?author=royerloic) - @royerloic
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Shannon Axelrod](https://github.com/napari/napari/commits?author=shanaxel42) - @shanaxel42
