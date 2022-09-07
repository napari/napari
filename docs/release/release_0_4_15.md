# napari 0.4.15

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
built, and deployed from the napari repository (#4047, #4147, #4154, #4176,
#4183). This and ongoing work will make it easier for the community to
contribute documentation to the project and help others use napari. We have a
new [documentation contributing guide](docs_contributing_guide) to help
community members contribute documentation pages.

Our volume display layers (`Image` and `Labels`) have gained a new slicing
plane mode and user interface to view slices of the data embedded within a
larger volume (#3759). This had been previously available as an experimental
API, and is now much improved, with new user interface controls. See the
[volume plane rendering](https://github.com/napari/napari/blob/e1ebbc20ccd3136dee1a7f1c051ea65d020b429c/examples/volume_plane_rendering.py)
example for details.

We have added completely optional, opt-in error reporting via the package
[napari-error-reporter](https://github.com/tlambert03/napari-error-reporter/).
If users install this third-party package, they will be presented with a
dialog asking whether to opt-in to report errors to select napari core
developers, which will help us prioritise and fix bugs that we might otherwise
miss. See the project's
[README](https://github.com/tlambert03/napari-error-reporter/#readme) for
details.

Finally, we have made some updates to how our application works. We now provide
a bundle based on conda, which should help with plugin installation on
different platforms (#3555). Additionally, the plugin listing inside the app is
now based on the napari hub API, which allows plugin authors to distribute
plugins on PyPI but have them not be listed in the app (#4074).

Keep reading below for the full list of changes!

## New Features

- Add edge_width per-point and edge_width_is_relative (#3999)
- Support error reporting via napari-error-reporter (#4055)
- volume plane interactive controls (#3759)

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
- Prototype a conda-based bundle (#3555)
- Use napari hub api to list plugins (#4074)

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
- Add docs contributor's guide and docs template (#4168)
- Adds calendar to docs  (#4176)
- add licensing page (#4185)
- Update release notes for 0.4.15rc1 (#4193)
- Fix plausible not added to docs (#4199)
- Documentation for scale in layers documentation (#4204)
- Fix EULA/licensing/signing issues post-merge (#4210)
- Add PRs made after 0.4.15rc1 to release notes (#4217)

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
- block points add mode in 3D when layer.ndim == 2 (#4184)
- fix bugs in setting plane orientation with nD data (#4197)
- Add value method to extension2reader widget (#4179)
- Add hook implementation name to order sorting (#4180)
- Add dependencies helper and fix pyside issue when filtering plugins (#4214)
- Add plugin dialog initial tests (#4216)
- Update napari pin for conda/mamba install in constructor bundle (#4220)
- Adjust conditions that control package signing and artifact uploads (constructor installers) (#4221)
- Fix name normalization of already installed plugins (#4223)
- Use conda instead of mamba in windows conda-forge bundled app (#4225)
- Remove access to private npe2 plugin manager attributes (#4236)

## API Changes


## Deprecations


## Build Tools

- Don't use git+https to install on CI (#4116)
- Add markupsafe pin to bundle build (#4118)
- Don't pip install from github in comprehensive tests (avoid another LFS leak) (#4141)
- fix make docs for non editable installs (#4156)
- Deploy docs workflow (#4154)
- Allow the calendar to build correctly in PR build docs (#4181)
- Deploy to napari.github.io (#4183)

## Other Pull Requests

- [pre-commit.ci] pre-commit autoupdate (#4032)
- remove leftover comments on `Points.n_dimensional` (#4052)
- Drop python 3.7 (#4063)
- avoid scikit-image deprecation warnings in test suite (#4071)
- bump magicgui dep (fix `_normalize_slot` AttributeError and minreqs test) (#4082)
- Refactor dummy text data for vispy (#4097)
- Add stateful StyleEncoding tests (#4113)
- Remove print statement (#4178)
- Adding tests missing from PR #4165 (#4170)

## 19 authors added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [chili-chiu](https://github.com/napari/napari/commits?author=chili-chiu) - @chili-chiu
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Gregory Lee](https://github.com/napari/napari/commits?author=grlee77) - @grlee77
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Jeremy Asuncion](https://github.com/napari/napari/commits?author=codemonkey800) - @codemonkey800
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [pre-commit-ci[bot]](https://github.com/napari/napari/commits?author=pre-commit-ci[bot]) - @pre-commit-ci[bot]
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03

## 25 reviewers added to this release (alphabetical)

- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [chili-chiu](https://github.com/napari/napari/commits?author=chili-chiu) - @chili-chiu
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Gregory Lee](https://github.com/napari/napari/commits?author=grlee77) - @grlee77
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jackson Maxfield Brown](https://github.com/napari/napari/commits?author=JacksonMaxfield) - @JacksonMaxfield
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Jan-Hendrik Müller](https://github.com/napari/napari/commits?author=kolibril13) - @kolibril13
- [Jeremy Asuncion](https://github.com/napari/napari/commits?author=codemonkey800) - @codemonkey800
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justin Kiggins](https://github.com/napari/napari/commits?author=neuromusic) - @neuromusic
- [Justine Larsen](https://github.com/napari/napari/commits?author=justinelarsen) - @justinelarsen
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
- [Ziyang Liu](https://github.com/napari/napari/commits?author=potating-potato) - @potating-potato

