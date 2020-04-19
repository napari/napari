# napari 0.2.0

We're happy to announce the release of napari 0.2.0! napari is a fast,
interactive, multi-dimensional image viewer for Python. It's designed for
browsing, annotating, and analyzing large multi-dimensional images. It's built
on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the
scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## New Features

- **Improved UI**, unifying buttons from controls, icons for layers,
  and more understandable dimensions sliders
- Add support for **3D rendering** for all our layer types
- Add a `Surface` layer to render already generated meshes. Support nD meshes
  rendered in 2D or 3D.
- Add `viewer.add_multichannel` method to rapidly add expand a multichannel
  array along one particular axis with different colormaps (#528).
- Add basic **undo / redo** functionality to the labels layer

## Deprecations

- Drop `napari.view` method. Replaced with `napari.view_*` methods in for all
  our layer types.
- Drop `Pyramid` layer. Pyramid functionality now integrated into both the
  labels and image layer.

## Pull Requests

- Tutorials (#395)
- fix import in cli (#403)
- 3D volume viewer - volume layer (#405)
- remove vispy backport (#406)
- Fix axis shape one (#409)
- Xarray example (#410)
- fix clim setter (#411)
- switch to pyside2 (#412)
- fix delete markers (#413)
- [FIX] paint color inidicator update when shuffle color (#416)
- QT returns a warning instead of an error (#418)
- Fix Crash with stacked binary tiffs. (#422)
- cleanup shape classes (#423)
- move tutorials to napari-tutorials repo (#425)
- fix vispy 0.6.0 colormap bug (#426)
- fix points keypress (#427)
- minimal vispy 0.6 colormap fix (#430)
- Fix dims sliders (#431)
- add `_vispy` init (#433)
- Expose args for blending, visible, opacity (#434)
- more dims fixes (#435)
- fix screenshot (#437)
- fix dims mixing (#438)
- test add_* signatures and improve docstring testing (#439)
- add qt console (#443)
- adapt existing keybindings to use new system (#444)
- fix aspect ratio (#446)
- Swappable dimensions (#451)
- use __init_subclass__ in keymap mixin to create empty class keymap (#452)
- use pytest-qt (#453)
- use codecov (#455)
- expose scaling factor for volume (#463)
- fix size policy on layers list (#466)
- Allow out of range float images (#468)
- add viewer keybindings (#472)
- fix windows ci build (#479)
- fix OSX CI (#482)
- remove vispy backport (#483)
- clean up black pre-commit hook & exclusion pattern (#484)
- remove vispy code from layer models (#485)
- host docs (#486)
- Fix keybindings (#487)
- layer views (#488)
- Include requirements/default.txt in sdist (#491)
- Integrate 3D rendering with layers (#493)
- revert "layer views (#488)" (#494)
- support more image dtypes (#498)
- rename clim (#499)
- fix cursor position (#501)
- add surface layer (#503)
- don't ignore errors in events (#505)
- fix contributing guidelines (#506)
- create release guide (#508)
- fix node ordering (#509)
- fix call signature to work with keyword-only arguments (#510)
- prevent selected label from being reduced below 0 (#512)
- fix typos in release guidelines (#522)
- clip rgba images (#524)
- DOC: specify that IPython needs to be started with `gui=qt` (#525)
- add multichannel (#528)
- enable `python -m napari` (#529)
- support 3D rendering shapes layer (#532)
- add undo/redo to labels layer (#533)
- unified layer ui (#536)
- add `view_*` methods at napari level (#542)
- Merge pyramid layer into image (#545)
- Add release notes (#546)
- Labels pyramid (#548)
- fix 3d point rendering (#549)
- make dims sliders bars (#550)
- fix menubar focus mac (#553)
- move zarr, xarray, dask from examples to tests (#555)
- fix pyramid guessing (#556)
- Update NumPy pad call for 1.16.4 (#559)
- WIP Unify IO between different modalities (#560)

## 11 authors added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Alexandre de Siqueira](https://github.com/napari/napari/commits?author=alexdesiqueira) - @alexdesiqueira
- [Ariel Rokem](https://github.com/napari/napari/commits?author=arokem) - @arokem
- [Christoph Gohlke](https://github.com/napari/napari/commits?author=cgohlke) - @cgohlke
- [Jan Eglinger](https://github.com/napari/napari/commits?author=imagejan) - @imagejan
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Mars Huang](https://github.com/napari/napari/commits?author=marshuang80) - @marshuang80
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pranathi Vemuri](https://github.com/napari/napari/commits?author=pranathivemuri) - @pranathivemuri

## 6 reviewers added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Loic Royer](https://github.com/napari/napari/commits?author=royerloic) - @royerloic
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pranathi Vemuri](https://github.com/napari/napari/commits?author=pranathivemuri) - @pranathivemuri
