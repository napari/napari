# napari 0.4.13

We're happy to announce the release of napari 0.4.13!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).

This is a major release with many new features, so don't hesitate to raise any
issues at https://github.com/napari/napari/issues! Thank you to the 27 authors
and 30 reviewers who contributed to this release!

This release contains a new "spherical shading" mode for points (#3430), which
allows napari to be used for molecular visualizations, among other use cases.
This is currently only available via the API using the keyword argument
`shading='spherical'`.

3D interactivity is further improved with a new 3D click-and-drag API (#3205)
and box selection of points in 3D (#3840).

This is the first release supporting npe2, the second iteration of napari
plugin engine, which will soon enable more powerful actions from plugins.
New plugin authors should start using npe2, while existing authors can migrate
their plugins using our
[migration guide](https://napari.org/plugins/stable/npe2_migration_guide.html).

For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

Complete list of changes below:

## Highlights
- Spherical Points (#3430)
- Point selection in 3d (#3508)
- Surface normals and wireframe (#3689)
- Add dialog for selecting reader plugin when dragging & dropping a file (#3799)


## New Features
- 3D click + drag interactivity API (#3205)
- Initial npe2 writer support (#3426)
- Add glossary (#3569)
- Use pandas DataFrame to store layer features (#3730)
- Add `experimental_canvas_size_limits` argument to points (#3734)
- 3D point selection with drag box (#3840)


## Improvements
- Raise AttributeError when labels.contrast_limits are set to anything other than (0, 1) (#2573)
- Allow magicgui to return a worker object (#2593)
- Proposition of save as_dict in get_theme but with change default behaviour in future (#3429)
- Refactor `progress` into an evented model (#3439)
- Allow to connect method to event without positional arguments (#3449)
- Bump vispy (#3494)
- Make text param consistent with text setter using `EventedModel.update` (#3503)
- Rename autoscale to auto-contrast (#3509)
- Add Context object, and LayerListContextKeys (#3513)
- Add class method to viewers to simplify closing all of them. (#3516)
- Ndisplay async (#3517)
- Minor LayerList refactor to remove unneeded NumPy<=1.17 fallback code (#3522)
- Establish LayerDataProtocol and basic Multiscale wrapper (resuscitate #2683) (#3560)
- Add checked `QpushButton` qss (#3561)
- Allow disconnect from EventEmiter using object (#3566)
- InteractionBox as EventedModel in Viewer (#3577)
- Minimal update to plan for qtpy 2.0 and pyqt6. (#3582)
- Add a method for setting `Camera.view_direction` and `Camera.up_direction` (#3586)
- Support RGB(A) array as colormap input type (#3587)
- Integrate more updates from npe2 (#3590)
- Initial support for npe2 widget (#3591)
- Cache layer extents (#3606)
- Make points add mode cursor a crosshair (#3607)
- Ommit RuntimeError during `remove_flash_animation` (#3609)
- Transform mode for image layer (#3611)
- Add better error messages for bad colormaps (#3613)
- Turn off vispy auto connect magic (#3615)
- Remove 1-pixel border from vispy widgets (#3621)
- Npe2 samples (#3635)
- Use more weakrefs in test. (#3639)
- Cleanup console namespace in test to avoid leaking viewer. (#3640)
- Add support for negative labels (#3647)
- Use a Weak Counter to track blocked events. (#3649)
- Update vendored modules. (#3655)
- Coerce vector length type (#3660)
- Float/unfloat plugin dockwidgets on first show (#3663)
- Try to close dask threads after usage. (#3665)
- Consolidate npe2 stuff into one module (#3666)
- User threadworker from superqt (#3668)
- Use in-memory IPython history during testing (#3671)
- Apply filter after yielding search results (#3675)
- Add setter for modifying multiple dims.range elements (#3677)
- Cleanum QtPopup code (#3687)
- Remove buttons and context from action manager, fix multi-viewer (#3692)
- Add npe2 plugins to plugins installed list  (#3694)
- Force layer name to be string (#3699)
- Add sequence support to more Dims setters (#3709)
- Add tooltip implementation for points (#3710)
- Add support for property.setters to EventedModel (#3711)
- Add a button linking to the plugin homepage in the plugin installer dialog (#3712)
- More efficient computation of displayed dimension order (#3714)
- Initialize viewer point at data center (#3718)
- Use public-only object proxy for magicgui and plugin dock_widgets (#3736)
- Remove float64 from GPU-accepted dtypes (#3739)
- Privatise translate_grid and expose data_to_world (#3754)
- Add icons for severity handling in notification manager (#3760)
- Added popup to manually order the viewer axes. (#3766)
- Add test for new large labels when in contour mode (#3768)
- Don't run qapplication when testing examples (#3773)
- Refactor npe2 widget name to display_name (#3789)
- Don't use `pytest.warns(None)` (avoids DeprecationWarning with pytest 7)  (#3793)
- Integrate npe2 schema updates  (#3798)
- Feature defaults property (#3801)
- Add event debugging tool (#3802)
- napari cli: enable --with option for npe2 widgets (#3807)
- Add buttons removed in #3692 with added deprecation (#3808)
- Dont pass app to QtToolTipEventFilter (#3817)
- npe2 tooltip and icon (#3824)
- Add warning to npe2 docs that they are in-progress (#3829)
- Remove `needs: code` from test_comprehensive workflow (#3841)
- Remove excess fields from plugin manifest (#3843)
- Remove napari from installed plugins list (#3845)
- QtViewer viewer argument type annotation fix (#3846)
- Changes so viewer is centered only when first layer is added.  (#3859)
- npe2 icon on plugin install dialog (#3860)
- Protect against setting corner_pixels with extent 0 for multiscale images (#3866)
- Properly order IPython-related logic. (#3870)
- Indicate how to enable test in warning message (#3871)
- Properly raise exception in test to set __traceback__ (#3873)
- Replace timeout in test by duration assertion. (#3887)
- Do not leak QtViewer between tests. (#3890)
- `_track_colors`  cleanup (#3891)
- Add error message if local file doesn't exist. (#3900)
- Better error message on failed window import (#3901)
- Update typing test on CI (#3903)
- Features implementation refactor (#3904)
- Add npe2 to install requires (#3906)
- Rename feature manager attribute to table (#3931)
- Better Notification __str__ method (#3933)
- Fix ndisplay button highlight (#3935)


## Bug Fixes
- Fix removing selected points with derived text (#3505)
- Fix for wrong bounding box definition (#3511)
- Fix missing class annotations from stubgen (#3514)
- Fix corner_pixels set from QtViewer.on_draw when layers have mixed numbers of dimensions (#3519)
- Fix test import (#3588)
- Fix bug with inspecting dock widget __init__  (#3612)
- Fix screenshot test on hi dpi screen. (#3638)
- Fix HiDPI test. (#3641)
- Fix workflow yml syntax (#3643)
- Fix 3D image interpolation breaking remove-and-add layer (#3670)
- Fix calc_data_range for large, 1D data (#3683)
- Fix event emission order on base layer (#3684)
- Fix custom color with contour in Labels layer (#3697)
- Use GPU scaled textures (fix NaN) (#3701)
- Fixes several sphinx warnings. (#3706)
- Fix tracks id (#3729)
- Fix 3D labels iso rendering (#3738)
- Fix napari.yaml manifest for npe2 (#3743)
- Fix QtModeRadioButton strong reference causing test failure (#3776)
- Fix getattr obj __module__ is None in PublicOnlyProxy (#3786)
- Fix dims order for clipping planes (#3790)
- Fix `_get_tooltip_text` signature, add docstrings (#3804)
- Fix tests for dask 2012.12.0 (#3806)
- Fix lack of enters in properties tooltip (#3809)
- Fix reference to missing q in npe2 getting started guide (#3847)
- Hack sys.argv to fix #2389 on macos (#3853)
- Fix edge width factor of 2 (#3857)
- Fix surface wireframe (#3879)
- Fix get_value() when called before canvas has been interacted with (#3881)
- Fix surface layer control layout (#3883)
- Fix contrast limit slider for small range (#3895)
- Fix tracks instantiation with floating point time values (#3909)
- Fix cleaning of resources in function contextmanagers (#3918)
- Fix magicgui layer combobox not populated when adding to viewer (#3938)
- Fix setting attribute to None for single slices of image stacks on stack creation (#3941)
- Fix too many entries in available plugins list (#3943)
- Block some duplications in Plugins menu (#3957)
- Fix close_progress_bar with no current_pbar (#3961)
- Switch append to concat (#3963)
- Update plugin docs index (#3964)


## API Changes


## Deprecations
- Deprecate public `window.qt_viewer` (remove in 0.5.0) (#3748)
- Deprecate `qt_viewer.screenshot` & `clipboard` (#3765)
- Restrict PublicOnlyProxy to napari namespace, allow napari-internal private usage (#3777)
- Change PublicOnlyProxy deprecation expiry to 0.5.0 (#3788)
- Remove deprecation of sceenshot in qt_viewer (#3937)


## Build Tools and Docs
- Auto generate event reference docs (#2750)
- Try to run docs under xvfb-run. (#3497)
- Patch plugin manager in `plugins.io` during test (#3515)
- Add docs on Working Groups to the napari.org docs (#3558)
- Fix headings on "Hooking up your own events" (#3563)
- Docs on contexts and expressions (#3571)
- Documentation: Fix capitalization, duplicated ToCs and translate rst to md files (#3594)
- Fix SSL Cert error for external libraries used in the bundle (#3595)
- Add sentinel file to detect whether bundled or not (#3599)
- Add example computing dynamic projections with dask and magicgui (#3604)
- Bump tensorstore dependency (#3614)
- Skip dask example test (#3616)
- Move vendored check to cron job (#3626)
- Try to autoupdate vendored code. (#3627)
- Misc docs fixes (#3628)
- Proper PR title on vendored code autoupdate (#3645)
- Exclude new buggy version of imageio. (#3648)
- Don't warn about pythonw on Big Sur and later (#3656)
- Add docs on best coding practices for plugin developers. (#3657)
- Update translation strings checks (#3662)
- Test on macos-11, py39 (#3673)
- Restore all mac-ci-skipped tests. (#3674)
- Reduce macos full testing. (#3676)
- Update python version in install instructions in readme (#3678)
- Try to automatically open/update an issues for missing translations (#3682)
- Support python 3.10 (#3702)
- Remove @slow on a number of tests (#3703)
- Remove slow decorator (#3705)
- Fix link to myst-parser documentation (#3716)
- Remove code check step from CI workflow (#3717)
- Test some typing on PR (#3722)
- Add `napari/plugins` and `utils/context` to typechecks (#3728)
- Remove main push from pre test (#3740)
- Update best_practices.md (#3744)
- Pin scikit image (!=0.19.0) in test suite (#3749)
- Moved files to agree with ToC reorganization. (#3751)
- [pre-commit.ci] pre-commit autoupdate (#3756)
- Fix outdated API in eventloop docs (#3772)
- Add glossary to copy-docs list (#3791)
- Update glossary.md links (#3797)
- update npe2 migration guide (#3832)
- Remove excess installation details from README (#3833)
- Add matplotlib to test dependencies (#3837)
- Update jupyter-book version (#3858)
- Readme install update (#3862)
- Link "development status" button in readme to explanation of "Alpha", "Beta",... (#3885)
- Use packaging.version, DeprecationWarnings since setuptools 59+  (#3894)
- [pre-commit.ci] pre-commit autoupdate (#3902)
- npe2 doc fix: remove outdated text (#3905)
- Fix docs calendar (#3912)
- Update plugin docs (#3916)
- Convert all line-ending to unix for the 3 files with CRLF (#3919)
- Try to fix OpenGL errors installing in CI (github deprecated git://) (#3920)
- FIX: git:// is deprecated (#3921)
- Fix plugin docs, ToC and links (#3929)
- Fix docs build on CI (#3946)
- Trying to fix published docs (#3949)
- Fix path to prep_docs in make_docs workflow (#3950)
- Add descriptive information to assertion checking if QtViewer is cleaned properly (#3960)
- Fix preference docs (#3967)


## 28 authors added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Chi-li Chiu](https://github.com/napari/napari/commits?author=chili-chiu) - @chili-chiu
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Eric Perlman](https://github.com/napari/napari/commits?author=perlman) - @perlman
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Gregory Lee](https://github.com/napari/napari/commits?author=grlee77) - @grlee77
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Guillaume Witz](https://github.com/napari/napari/commits?author=guiwitz) - @guiwitz
- [Isabela Presedo-Floyd](https://github.com/napari/napari/commits?author=isabela-pf) - @isabela-pf
- [Jacob Czech](https://github.com/napari/napari/commits?author=jczech) - @jczech
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Jeremy Asuncion](https://github.com/napari/napari/commits?author=codemonkey800) - @codemonkey800
- [Johannes Elferich](https://github.com/napari/napari/commits?author=jojoelfe) - @jojoelfe
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nathan Clack](https://github.com/napari/napari/commits?author=nclack) - @nclack
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Robert Haase](https://github.com/napari/napari/commits?author=haesleinhuepf) - @haesleinhuepf
- [Subhashree Mishra](https://github.com/napari/napari/commits?author=Mishrasubha) - @Mishrasubha
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03


## 30 reviewers added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Chi-li Chiu](https://github.com/napari/napari/commits?author=chili-chiu) - @chili-chiu
- [David Hoese](https://github.com/napari/napari/commits?author=djhoese) - @djhoese
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Eric Perlman](https://github.com/napari/napari/commits?author=perlman) - @perlman
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Gregory Lee](https://github.com/napari/napari/commits?author=grlee77) - @grlee77
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Isabela Presedo-Floyd](https://github.com/napari/napari/commits?author=isabela-pf) - @isabela-pf
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Johannes Elferich](https://github.com/napari/napari/commits?author=jojoelfe) - @jojoelfe
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justin Kiggins](https://github.com/napari/napari/commits?author=neuromusic) - @neuromusic
- [Justine Larsen](https://github.com/napari/napari/commits?author=justinelarsen) - @justinelarsen
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lucy Obus](https://github.com/napari/napari/commits?author=LCObus) - @LCObus
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nathan Clack](https://github.com/napari/napari/commits?author=nclack) - @nclack
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Robert Haase](https://github.com/napari/napari/commits?author=haesleinhuepf) - @haesleinhuepf
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=potating-potato) - @potating-potato

