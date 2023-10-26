# napari 0.4.19

We're happy to announce the release of napari 0.4.19!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights

This release is mostly a bug fix release. We aim for the next release to be 0.5.0, with many new exciting features!

Highlights in `v0.4.19`:

The removal of public access to `viewer.window.qt_viewer` was postponed to `v0.6.0` ([napari/napari/#6283](https://github.com/napari/napari/pull/6283)). If you're interested in some feature that's now hidden behind private or deprecated API, please make an issue on the napari repo requesting the feature to be exposed!

Napari is now pydantic 2 compatible. At this moment we are using `pydantic.v1` to achieve this. In this release the bundle will be shipped with pydantic v1, but
we expect that in the next release we will ship the bundle with pydantic v2.
Please, if you are a plugin developer and your plugin uses pydantic, ensure 
that it is compatible with pydantic v2 ([napari/napari/#6358](https://github.com/napari/napari/pull/6358)).

Performance of the `Labels` layer is drastically improved for many labels by moving part of calculations to the GPU ([napari/napari/#3308](https://github.com/napari/napari/pull/3308)).

If you have any questions or suggestions regarding napari core, for example on how to adapt your plugin to be compatible with future napari versions, please get in touch! We have weekly community meetings, for which you can find the schedule [here](https://napari.org/stable/community/meeting_schedule.html). We would like to hear from you.

- Use a shader for low discrepancy label conversion ([napari/napari/#3308](https://github.com/napari/napari/pull/3308))
- Postpone qt_viewer deprecation to 0.6.0 ([napari/napari/#6283](https://github.com/napari/napari/pull/6283))
- Pydantic 2 compatibility using `pydantic.v1`  ([napari/napari/#6358](https://github.com/napari/napari/pull/6358))

## New Features

- Automatic recognition of hex colour strings in layer data ([napari/napari/#6102](https://github.com/napari/napari/pull/6102))

## Improvements

- Bump vispy to 0.13 ([napari/napari/#6025](https://github.com/napari/napari/pull/6025))
- Extend "No Qt bindings found" error message with details about conda ([napari/napari/#6095](https://github.com/napari/napari/pull/6095))
- Implement direct color calculation in shaders for Labels auto color mode ([napari/napari/#6179](https://github.com/napari/napari/pull/6179))
- Update "toggle ndview" text ([napari/napari/#6192](https://github.com/napari/napari/pull/6192))
- Add collision check when set colors for labels layer ([napari/napari/#6193](https://github.com/napari/napari/pull/6193))
- Add numpy as `np` to console predefined variables ([napari/napari/#6314](https://github.com/napari/napari/pull/6314))
- Pydantic 2 compatibility using `pydantic.v1`  ([napari/napari/#6358](https://github.com/napari/napari/pull/6358))

## Performance

- Use a shader for low discrepancy label conversion ([napari/napari/#3308](https://github.com/napari/napari/pull/3308))

## Bug Fixes

- Use a shader for low discrepancy label conversion ([napari/napari/#3308](https://github.com/napari/napari/pull/3308))
- Workaround Qt bug on Windows with fullscreen mode in some screen resolutions/scaling configurations ([napari/napari/#5401](https://github.com/napari/napari/pull/5401))
- Fix taskbar icon grouping in Windows bundle (add `app_user_model_id` to bundle shortcut) ([napari/napari/#6056](https://github.com/napari/napari/pull/6056))
- Add basic tests for the `ScreenshotDialog` widget and fixes ([napari/napari/#6057](https://github.com/napari/napari/pull/6057))
- Install napari from repository in docker image ([napari/napari/#6097](https://github.com/napari/napari/pull/6097))
- Fix automatic selection of points when setting data ([napari/napari/#6098](https://github.com/napari/napari/pull/6098))
- Fix exception raised on empty pattern in search plugin in preferences ([napari/napari/#6107](https://github.com/napari/napari/pull/6107))
- Ensure visual is updated when painting into zarr array ([napari/napari/#6112](https://github.com/napari/napari/pull/6112))
- Emit event from Points data setter ([napari/napari/#6117](https://github.com/napari/napari/pull/6117))
- Emit event from Shapes data setter ([napari/napari/#6134](https://github.com/napari/napari/pull/6134))
- Fix oblique button by chekcing if action is generator ([napari/napari/#6145](https://github.com/napari/napari/pull/6145))
- Fix bug in `examples/multiple_viewer_widget.py` copy layer logic ([napari/napari/#6162](https://github.com/napari/napari/pull/6162))
- Fix split logic in shortcut editor ([napari/napari/#6163](https://github.com/napari/napari/pull/6163))
- Layer data events before and after ([napari/napari/#6178](https://github.com/napari/napari/pull/6178))
- Implement direct color calculation in shaders for Labels auto color mode ([napari/napari/#6179](https://github.com/napari/napari/pull/6179))
- Update `QLabeledRangeSlider` style rule to prevent labels from being cut off ([napari/napari/#6180](https://github.com/napari/napari/pull/6180))
- Update color texture build to reduce collisions, and fix collision handling ([napari/napari/#6182](https://github.com/napari/napari/pull/6182))
- Prevent layer controls buttons changing layout while taking screenshots with flash effect on ([napari/napari/#6194](https://github.com/napari/napari/pull/6194))
- Vispy 0.14 ([napari/napari/#6214](https://github.com/napari/napari/pull/6214))
- Fix `ShapeList.outline` validations for `int`/list like argument and add a test ([napari/napari/#6215](https://github.com/napari/napari/pull/6215))
- Ensure pandas Series is initialized with a list as data ([napari/napari/#6226](https://github.com/napari/napari/pull/6226))
- Fix Python 3.11 StrEnum Compatibility ([napari/napari/#6242](https://github.com/napari/napari/pull/6242))
- FIX add `changing` event to `EventedDict` ([napari/napari/#6268](https://github.com/napari/napari/pull/6268))
- Restore default color support for direct color mode in Labels layer ([napari/napari/#6311](https://github.com/napari/napari/pull/6311))
- Update example scripts (magicgui with threads) ([napari/napari/#6353](https://github.com/napari/napari/pull/6353))
- Exclude the loaded property when linking two layers ([napari/napari/#6377](https://github.com/napari/napari/pull/6377))

## API Changes

- Layer data events before and after ([napari/napari/#6178](https://github.com/napari/napari/pull/6178))

## Deprecations

- Postpone qt_viewer deprecation to 0.6.0 ([napari/napari/#6283](https://github.com/napari/napari/pull/6283))

## Build Tools

- Vispy 0.14 ([napari/napari/#6214](https://github.com/napari/napari/pull/6214))

## Documentation

- Add HIP workshop to documentation/workshops ([napari/napari/#5117](https://github.com/napari/napari/pull/5117))
- Update README.md for conda install change ([napari/napari/#6123](https://github.com/napari/napari/pull/6123))
- Add 0.4.19 release notes ([napari/napari/#6376](https://github.com/napari/napari/pull/6376))
- Update docs contribution guide for two-repo setup ([napari/docs/#5](https://github.com/napari/docs/pull/5))
- add foundation grant onboarding workshop ([napari/docs/#55](https://github.com/napari/docs/pull/55))
- Fixes formatting for the contributing documentation section ([napari/docs/#79](https://github.com/napari/docs/pull/79))
- Move napari workshop template link to top of page ([napari/docs/#90](https://github.com/napari/docs/pull/90))
- Add instructions to build napari docs on Windows ([napari/docs/#158](https://github.com/napari/docs/pull/158))
- Fix typo of points instead of shapes ([napari/docs/#195](https://github.com/napari/docs/pull/195))
- Improve titles of fundamentals tutorials ([napari/docs/#196](https://github.com/napari/docs/pull/196))
- NAP 7: Key Binding Dispatch ([napari/docs/#200](https://github.com/napari/docs/pull/200))
- make Talley emeritus SC ([napari/docs/#204](https://github.com/napari/docs/pull/204))
- Use napari_scraper instead of qtgallery ([napari/docs/#207](https://github.com/napari/docs/pull/207))
- Move contributing resources to top-level navbar ([napari/docs/#208](https://github.com/napari/docs/pull/208))
- Add roadmap board link to Roadmaps page ([napari/docs/#212](https://github.com/napari/docs/pull/212))
- Fix getting started in napari linking to the unittest getting started page ([napari/docs/#217](https://github.com/napari/docs/pull/217))
- Update core developer list ([napari/docs/#219](https://github.com/napari/docs/pull/219))
- Add note on milestones for PRs ([napari/docs/#221](https://github.com/napari/docs/pull/221))
- Fix titles on Getting Started section of user guide ([napari/docs/#228](https://github.com/napari/docs/pull/228))
- Update Kyle's tag on the core devs page ([napari/docs/#232](https://github.com/napari/docs/pull/232))
- Update ndisplay title ([napari/docs/#235](https://github.com/napari/docs/pull/235))
- Remove sub-sub section heading from rendering guide ([napari/docs/#236](https://github.com/napari/docs/pull/236))
- Update selection instructions in Points tutorial for Shift-A keybinding ([napari/docs/#238](https://github.com/napari/docs/pull/238))
- fix outdated dimension sliders documentation ([napari/docs/#241](https://github.com/napari/docs/pull/241))
- Update napari-workshops.md ([napari/docs/#243](https://github.com/napari/docs/pull/243))

## Other Pull Requests

- test: [Automatic] Constraints upgrades: `certifi`, `dask`, `fsspec`, `hypothesis`, `imageio`, `ipython`, `pint`, `qtconsole`, `rich`, `virtualenv` ([napari/napari/#5788](https://github.com/napari/napari/pull/5788))
- test: [Automatic] Constraints upgrades: `dask`, `hypothesis`, `torch` ([napari/napari/#5835](https://github.com/napari/napari/pull/5835))
- [Auto] Constraints upgrades: ipykernel, lit, setuptools, xarray ([napari/napari/#5857](https://github.com/napari/napari/pull/5857))
- Clean up shapes mouse test for clarity ([napari/napari/#5917](https://github.com/napari/napari/pull/5917))
- test: [Automatic] Constraints upgrades: `app-model`, `certifi`, `dask`, `hypothesis`, `jsonschema`, `npe2`, `pydantic`, `pyqt6`, `pyyaml`, `tifffile`, `virtualenv`, `xarray`, `zarr` ([napari/napari/#6007](https://github.com/napari/napari/pull/6007))
- test: [Automatic] Constraints upgrades: `rich` ([napari/napari/#6105](https://github.com/napari/napari/pull/6105))
- [Automatic] Constraints upgrades: `dask`, `hypothesis`, `jsonschema`, `numpy`, `pygments`, `rich`, `superqt` ([napari/napari/#6124](https://github.com/napari/napari/pull/6124))
- [pre-commit.ci] pre-commit autoupdate ([napari/napari/#6128](https://github.com/napari/napari/pull/6128))
- Fix headless test ([napari/napari/#6161](https://github.com/napari/napari/pull/6161))
- Stop using temporary directory for store array for paint test ([napari/napari/#6191](https://github.com/napari/napari/pull/6191))
- Use class name for object that does not have qt name ([napari/napari/#6222](https://github.com/napari/napari/pull/6222))
- Fix labeler by adding permissions ([napari/napari/#6289](https://github.com/napari/napari/pull/6289))
- Update pre-commit and constraints and minor fixes for 0.4.19 release ([napari/napari/#6340](https://github.com/napari/napari/pull/6340))
- Ensure conda workflow runs with proper permissions ([napari/napari/#6378](https://github.com/napari/napari/pull/6378))
-  Remove sphinx dependency from defaults dependecies ([napari/napari/#6380](https://github.com/napari/napari/pull/6380))
- Fix `test_link_layers_with_images_then_loaded_not_linked` test ([napari/napari/#6385](https://github.com/napari/napari/pull/6385))
- Update docs to suggest python 3.10 install ([napari/docs/#246](https://github.com/napari/docs/pull/246))


## 17 authors added to this release (alphabetical)

- [akuten1298](https://github.com/napari/napari/commits?author=akuten1298) - @akuten1298
- [Andrew Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Daniel Althviz Moré](https://github.com/napari/napari/commits?author=dalthviz) - @dalthviz
- [David Stansby](https://github.com/napari/napari/commits?author=dstansby) - @dstansby
- [Egor Zindy](https://github.com/napari/napari/commits?author=zindy) - @zindy
- [Elena Pascal](https://github.com/napari/napari/commits?author=elena-pascal) - @elena-pascal
- [Eric Perlman](https://github.com/napari/napari/commits?author=perlman) - @perlman
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lucy Liu](https://github.com/napari/napari/commits?author=lucyleeow) - @lucyleeow
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Robert Haase](https://github.com/napari/napari/commits?author=haesleinhuepf) - @haesleinhuepf
- [Wouter-Michiel Vierdag](https://github.com/napari/napari/commits?author=melonora) - @melonora


## 14 reviewers added to this release (alphabetical)

- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andrew Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Daniel Althviz Moré](https://github.com/napari/napari/commits?author=dalthviz) - @dalthviz
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Egor Zindy](https://github.com/napari/napari/commits?author=zindy) - @zindy
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Wouter-Michiel Vierdag](https://github.com/napari/napari/commits?author=melonora) - @melonora


## 11 docs authors added to this release (alphabetical)

- [Ashley Anderson](https://github.com/napari/docs/commits?author=aganders3) - @aganders3
- [chili-chiu](https://github.com/napari/docs/commits?author=chili-chiu) - @chili-chiu
- [David Stansby](https://github.com/napari/docs/commits?author=dstansby) - @dstansby
- [dgmccart](https://github.com/napari/docs/commits?author=dgmccart) - @dgmccart
- [Juan Nunez-Iglesias](https://github.com/napari/docs/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/docs/commits?author=kne42) - @kne42
- [Melissa Weber Mendonça](https://github.com/napari/docs/commits?author=melissawm) - @melissawm
- [Peter Sobolewski](https://github.com/napari/docs/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Sean Martin](https://github.com/napari/docs/commits?author=seankmartin) - @seankmartin
- [Talley Lambert](https://github.com/napari/docs/commits?author=tlambert03) - @tlambert03
- [Wouter-Michiel Vierdag](https://github.com/napari/docs/commits?author=melonora) - @melonora


## 13 docs reviewers added to this release (alphabetical)

- [Andrew Sweet](https://github.com/napari/docs/commits?author=andy-sweet) - @andy-sweet
- [David Stansby](https://github.com/napari/docs/commits?author=dstansby) - @dstansby
- [Draga Doncila Pop](https://github.com/napari/docs/commits?author=DragaDoncila) - @DragaDoncila
- [Grzegorz Bokota](https://github.com/napari/docs/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/docs/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/docs/commits?author=kne42) - @kne42
- [Lorenzo Gaifas](https://github.com/napari/docs/commits?author=brisvag) - @brisvag
- [Lucy Liu](https://github.com/napari/docs/commits?author=lucyleeow) - @lucyleeow
- [Matthias Bussonnier](https://github.com/napari/docs/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/docs/commits?author=melissawm) - @melissawm
- [Peter Sobolewski](https://github.com/napari/docs/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Sean Martin](https://github.com/napari/docs/commits?author=seankmartin) - @seankmartin
- [Wouter-Michiel Vierdag](https://github.com/napari/docs/commits?author=melonora) - @melonora

## New Contributors

There are 4 new contributors for this release:

- akuten1298 [napari](https://github.com/napari/napari/commits?author=akuten1298) - @akuten1298
- dgmccart [docs](https://github.com/napari/docs/commits?author=dgmccart) - @dgmccart
- Egor Zindy [napari](https://github.com/napari/napari/commits?author=zindy) - @zindy
- Elena Pascal [napari](https://github.com/napari/napari/commits?author=elena-pascal) - @elena-pascal
