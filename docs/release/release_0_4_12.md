# napari 0.4.12

We're happy to announce the release of napari 0.4.12!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights

This is a bug fix release with many minor improvements and bug fixes. The user
experience for users of dask arrays might be significantly improved by a new
approach to setting the contrast limits based on the current slice (#3425).

A progress bar will now display when opening multiple files (#3355).
Thanks to first-time contributor @tibuch the data type of labels layers can now
be converted from a context menu on the layer list (#3402).

See the full list of merged pull requests below for further delails!

## New Features
- Add progress bar when opening list of files (#3355)
- Add right-click context menu to convert label data type. (#3402)


## Improvements
- Support for `Future` return type in magicgui widget  (#2581)
- Don't register dask cache globally - but do use cache as context manager when slicing (#3285)
- Hide or Destroy dock widgets (#3331)
- `_vispy` module cleanup (#3333)
- Add expressions API (will eventually support `when` expressions for internal & plugin usage) (#3350)
- Experimental npe2 support (#3354)
- Disable save options if no layers available (#3363)
- Add cancel and cancel all actions to plugin dialog and improve UI (#3369)
- Add an option to change the theme to match the system one (#3370)
- Add toggle visibility to layer actions (#3372)
- Remove 0.4.9 deprecations (#3377)
- Move `progress` outside of `qt` and eliminate need for `qt` imports in headless mode  (#3379)
- Normalize_dtype() when setting contrast limits. (#3380)
- Add translucent no depth blending mode (#3398)
- Center data within points thumbnail (#3406)
- Add back support for big-endian NumPy dtypes in get_dtype_limits (#3424)
- Schedule contrast limits calculation for dask arrays after first set_view_slice (#3425)
- Minor refactor of get_active_layer_dtype (#3434)
- Add event filter to convert tooltips to richtext (#3442)
- Remove duplicate test parameter (#3443)
- Check if proper function name is found when connect to EventEmmiter (#3445)
- Prevent shapes removal while creating (#3451)
- Bug fix for individual shape selection with shift modifier (#3456)
- Multiscale slicing extends beyond shape (#3460)
- Added float16 support to dtype normalization (#3463)
- Updates for magicgui 0.3.0 (#3465)
- Finds layer_controls based on layer's MRO (#3471)
- Use `ensure_main_thread` instead of custom thread propagation mechanism in NapariQtNotification (#3473)
- Drop pythonw patch in windows bundle (#3479)
- Revert "drop pythonw patch (#3479)" (#3501)


## Bug Fixes
- Fix `_old_size` attribute error in main window (#3329)
- Fix problem with local function signal binding (#3352)
- Fix __getattr__ in WorkerBase (#3368)
- Fix off-by-one bug in extent of Image and Labels layers (#3381)
- Try to fix bundle building. (#3403)
- Fix to_labels default output shape (#3412)
- Fix lazy load console (#3419)
- Fix teardown of menus to prevent widget test leaks (#3433)
- Fix off-by one error in Dims.range for non pixel-based layers (#3444)
- Fix naming inconsistency for windows bundle (#3476)
- Add ability to provide empty data to vectors layer (#2995)


## API Changes


## Deprecations


## Documentation
- Fix docs order in _toc.yml for 0.4.11 (#3330)
- Add example with data of mixed dimensionality (#3392)


## Build Tools and Support
- Split windows pyside/pyqt into own GitHub action check (#2989)
- Attempt to cache tox virtualenv. (#2996)
- Update references to master to point to main (#3351)
- Fix docs build from main branch (#3365)
- Turn on testing examples on CI again (#3367)
- Add certifi to dependencies (#3386)
- Update README.md (#3396)
- Fix CI failure from example (#3397)
- Attempt to speedup tests around theme. (#3405)
- Add global test timeout to avoid regression. (#3407)
- Add missing rendering documentation (#3436)
- Try to skip test on windows. (#3438)
- Add wheel to bundle run dependecies (#3450)
- Clarify existing behavior of `TextManager` with new tests and benchmarks (#3452)
- Move 8s tiemout to github actions only (#3457)
- Move icons to package src (#3462)
- Fix image display for plugin installation page. (#3480)
- Fix broken links in documentation (#3481)
- Update formatting in README.md (#3482)
- Fix formatting of Layers docstrings (#3483)
- Add links to napari repo in README (#3484)
- Bump minimum NumPy requirement to 1.18 (as per NEP29) (#3485)
- Remove make_napari_viewer in vispy tests. (#3486)
- Update ubuntu image to 18.04 (#3348)
- Bundle: canonicalize arch names (#3349)
- Fix min_req text matrix by skiping test (#3496)
- Bundle: make fork multiprocessing default in macos again (#3498)

## 22 authors added to this release (alphabetical)

- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Eric Perlman](https://github.com/napari/napari/commits?author=perlman) - @perlman
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Gregory R. Lee](https://github.com/napari/napari/commits?author=grlee77) - @grlee77
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Guillaume Gay](https://github.com/napari/napari/commits?author=glyg) - @glyg
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Jan-Hendrik Müller](https://github.com/napari/napari/commits?author=kolibril13) - @kolibril13
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lukas Vasadi](https://github.com/napari/napari/commits?author=lukasvasadi) - @lukasvasadi
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Tim-Oliver Buchholz](https://github.com/napari/napari/commits?author=tibuch) - @tibuch
- [Ziyang Liu](https://github.com/napari/napari/commits?author=potating-potato) - @potating-potato


## 20 reviewers added to this release (alphabetical)

- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Gregory R. Lee](https://github.com/napari/napari/commits?author=grlee77) - @grlee77
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lukasz Migas](https://github.com/napari/napari/commits?author=lukasz-migas) - @lukasz-migas
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nathan Clack](https://github.com/napari/napari/commits?author=nclack) - @nclack
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [psobolewskiPhD](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Tim-Oliver Buchholz](https://github.com/napari/napari/commits?author=tibuch) - @tibuch
- [Ziyang Liu](https://github.com/napari/napari/commits?author=potating-potato) - @potating-potato

