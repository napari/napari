# napari 0.1.5

We're happy to announce the release of napari 0.1.5! napari is a fast,
interactive, multi-dimensional image viewer for Python. It's designed for
browsing, annotating, and analyzing large multi-dimensional images. It's built
on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the
scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## New Features

- Support for swappable dimensions
- Support for 3D rendering for more layer types

## Pull Requests

- Expose args for blending, visible, opacity (#434)
- test `add_*` signatures and improve docstring testing (#439)
- add qt console (#443)
- adapt existing keybindings to use new system (#444)
- fix aspect ratio (#446)
- Swappable dimensions (#451)
- use `__init_subclass__` in keymap mixin to create empty class keymap (#452)
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
- don't ignore errors in events (#505)
- fix contributing guidelines (#506)
- create release guide (#508)
- fix node ordering (#509)
- fix call signature to work with keyword-only arguments (#510)
- prevent selected label from being reduced below 0 (#512)

## 4 authors added to this release (alphabetical)

- [Christoph Gohlke](https://github.com/napari/napari/commits?author=cgohlke) - @cgohlke
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn

## 5 reviewers added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Loic Royer](https://github.com/napari/napari/commits?author=royerloic) - @royerloic
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
