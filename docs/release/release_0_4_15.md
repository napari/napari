# napari 0.4.15

```{note}
These are preliminary release notes until 0.4.15 is released. Currently this is
slated to occur on 2022-03-08.
```

We're happy to announce the release of napari 0.4.15!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).

Following NumPy's [NEP-29 deprecation
policy](https://numpy.org/neps/nep-0029-deprecation_policy.html), this release
drops support for Python 3.7, which enables us to simplify many code paths.
From 0.4.15, napari supports Python 3.8, 3.9, and 3.10.

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights

This release is focused on documentation improvements and bug fixes, with few
changes to napari's API. The napari documentation is now entirely contained,
built, and deployed within the napari repository (#4047, #4147, #4154). This
and ongoing work will make it easier for the community to contribute
documentation to the project and help others use napari.

We have added completely optional, opt-in error reporting via the package
[napari-error-reporter](https://github.com/tlambert03/napari-error-reporter/).
If users install this third-party package, they will be presented with a
dialog asking whether to opt-in to report errors to select napari core
developers, which will help us prioritise and fix bugs that we might otherwise
miss. See the project's
[README](https://github.com/tlambert03/napari-error-reporter/#readme) for
details.

## New Features

- Add edge_width per-point and edge_width_is_relative (#3999)
- Support error reporting via napari-error-reporter (#4055)

## Improvements

- avoid repeated computation of layers.extent when adding a new Labels layer (#4037)
- Add style encoding for strings (#4051)
- Cache layerlist expensive properties (#4054)
- Prevent errors in qt/test_plugin_widgets from installed plugins (#4058)
- Add a `recurse` option to `EventedModel.update` (#4062)
- Only add progress widget on worker start (#4068)
- minor refactor of Layer._slice_indices (#4072)
- Improve missing widget name error for npe2 plugins (#4080)
- Update shapes color mapping when changing property (#4081)
- reduce NumPy overhead in ScaleTranslate and Affine transform calls (#4094)
- Remove remaining references to requirements.txt (#4105)
- Try to standardise API as list of strings. (#4107)

## Documentation

- Use cells3d image in README example (#3823)
- update event tables in the in-depth guides (#4030)
- add quick start (#4041)
- New documentation infrastructure (#4047)
- Add Grzegorz to 0.4.14 release notes (#4061)
- remove `--checkout npe2` instructions for cookiecutter in documentation (#4092)
- Clarify that napari should be installed (#4101)
- Move workshop out of tutorials into usage (#4123)
- Refactor layers content (#4124)
- Organize tutorials sections (#4137)
- Add release management docs (#4139)
- Use napari-sphinx-theme in docs build (#4147)
- Add new core developers (#4157)
- Move most examples to use features instead of properties (#4162)
- improve examples/add_points_on_nD_shapes.py camera/dims setup (#4161)

## Bug Fixes

- Fix single colormap create (#4067)
- Fix ampersand in sample menu (#4070)
- Fix json usage when setting preferences from nested environment variables (#4078)
- Fix import in plugin_manager that is causing typing fails (#4106)
- Contrast limits popup fix (#4122)
- Prevent deepcopy of layer._source (#4128)
- Always pass list of strings to npe2. (#4130)
- Don't validate defaults on `EventedModel` (#4138)
- support conda-style (pyside2-)rcc locations (#4144)
- Fix style encoding tests on main (#4150)
- Fix bug in continuous autocontrast button, add tests (#4152)
- Tracks graph shader buffer order update (#4165)

## API Changes


## Deprecations


## Build Tools

- Don't use git+https to install on CI (#4116)
- Add markupsafe pin to bundle build (#4118)
- Don't pip install from github in comprehensive tests (avoid another LFS leak) (#4141)
- fix make docs for non editable installs (#4156)
- Deploy docs workflow (#4154)

## Other Pull Requests

- [pre-commit.ci] pre-commit autoupdate (#4032)
- remove leftover comments on `Points.n_dimensional` (#4052)
- Drop python 3.7 (#4063)
- avoid scikit-image deprecation warnings in test suite (#4071)
- bump magicgui dep (fix `_normalize_slot` AttributeError and minreqs test) (#4082)
- Refactor dummy text data for vispy (#4097)
- Add stateful StyleEncoding tests (#4113)


## 17 authors added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [chili-chiu](https://github.com/napari/napari/commits?author=chili-chiu) - @chili-chiu
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gregory Lee](https://github.com/napari/napari/commits?author=grlee77) - @grlee77
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Jeremy Asuncion](https://github.com/napari/napari/commits?author=codemonkey800) - @codemonkey800
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [pre-commit-ci[bot]](https://github.com/napari/napari/commits?author=pre-commit-ci[bot]) - @pre-commit-ci[bot]
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03


## 20 reviewers added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [chili-chiu](https://github.com/napari/napari/commits?author=chili-chiu) - @chili-chiu
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gregory Lee](https://github.com/napari/napari/commits?author=grlee77) - @grlee77
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jackson Maxfield Brown](https://github.com/napari/napari/commits?author=JacksonMaxfield) - @JacksonMaxfield
- [Jeremy Asuncion](https://github.com/napari/napari/commits?author=codemonkey800) - @codemonkey800
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justin Kiggins](https://github.com/napari/napari/commits?author=neuromusic) - @neuromusic
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nathan Clack](https://github.com/napari/napari/commits?author=nclack) - @nclack
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [shahidhaider](https://github.com/napari/napari/commits?author=shahidhaider) - @shahidhaider
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03

