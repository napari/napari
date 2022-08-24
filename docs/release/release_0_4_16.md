# napari 0.4.16

We're happy to announce the release of napari 0.4.16!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our
[website](https://napari.org).

## File Opening Changes in 0.4.16

Prior to `npe2`, file opening with plugins worked through a cascade of function calls trying different readers until one worked, or all failed, in which case an error would be raised. Preferences for readers could be set by reordering hook implementations in the Call Order preference dialog.

This behavior was slow, confusing, and often led to unexpected results. You can see more discussion on this in issue [#4000](https://github.com/napari/napari/issues/4000). `npe2` supports readers declaring a list of accepted filename patterns, and PR [#3799](https://github.com/napari/napari/pull/3799) added a dialog for users to select a plugin to read their file (if more than one was available), and save a preference for that file extension.

Before removing plugin call order, we want to make sure that file opening behavior across the GUI and command line is predictable, reproducible and explicit.

After discussion in [#4102](https://github.com/napari/napari/pull/4102), [#4111](https://github.com/napari/napari/discussions/4111) and [this zulip thread](https://napari.zulipchat.com/#narrow/stream/212875-general/topic/.60viewer.2Eopen.60.20.26.20multiple.20plugins), we decided that as a guiding principle, calling `viewer.open` should not infer a plugin choice for you, and any inference behavior should be opt in.

This has led to the following API and GUI changes

-  `builtins` is now the default value for the `plugin` argument in `viewer.open`. This means 
    - you should **always** explicitly pass a plugin to `viewer.open`, if you don't want to use `builtins` (and we encourage you to pass the argument anyway).

        - To specify a plugin in a Python script:

            ```python
            import napari

            viewer = napari.Viewer()
            viewer.open('my-path.tif') # this will throw MultipleReaderError if napari_tifffile is installed as both it and builtins could open the file
            viewer.open('my-path.tif', plugin='napari_tifffile') # this won't
            ```

    - `viewer.open` will **not** inspect your file extension preferences, and will not choose among available plugins
    - if you wish to opt into the "gui-like" behavior where your preferences are respected and we infer a plugin if just one is compatible with your file path, you must explicitly use `plugin=None`

        - To opt into plugin inference behavior:

            ```python
            import napari

            viewer = napari.Viewer()
            viewer.open('my-path.nd2', plugin=None)
            ```
        - If multiple plugins could read your file, you will see a `MultipleReaderError`
        - A preferred reader missing from current plugins will trigger a warning, but the preference will be otherwise ignored
        - A preferred reader failing to read your file will result in an error e.g. if you saved `napari_tifffile` as a preference for TIFFs but then tried to open a broken file

        - To save a preference for a file pattern in Python, use:

            ```python
            from napari.settings import get_settings
            get_settings().plugins.extension2reader['*.tif'] = 'napari_tifffile'
            get_settings().plugins.extension2reader['*.zarr'] = 'napari-ome-zarr'
            ```

- When opening a file through a GUI pathway (drag & drop, File -> Open, Open Sample) with no preferences saved, you are provided with a dialog allowing you to choose among the various plugins that are compatible with your file
    - This dialog also allows you to save a preference for files and folders with extensions
    - This dialog also pops up if a preferred reader fails to open your file
    - This dialog does not pop up if only one plugin can open your file
- Running `napari path` in the shell will also provide the reader dialog. You can still pass through a plugin choice, or layer keyword arguments
    - To specify a plugin at the command line, use:
    
    ```sh
    napari my-path.tif --plugin napari_tifffile
    ```
- Preference saving for file reading is now supported for filename patterns accepted by `npe2` readers, rather than strictly file extensions
    - Existing preferences for file extensions will be automatically updated e.g. `.tif` will become `*.tif`
- Reader preferences for filename patterns can be saved in the GUI via the preference dialog
    - Reader preferences for folders are not yet supported in the GUI preference dialog - use the Python method above
    - This will be addressed by the next release

We have thought carefully about these choices, but there are still some open questions to address, and features to implement. Some of these are captured across the issues listed below, and we'd love to hear any feedback you have about the new behavior!

- How can we support selecting an individual reader within plugins that offer multiple [#4391](https://github.com/napari/napari/issues/4391)
- If two plugins can read a file, and one is builtins, should we use the other plugin as it's likely more bespoke [#4389](https://github.com/napari/napari/issues/4389)
- Provide a way to "force" the reader dialog to open regardless of saved preferences [#4388](https://github.com/napari/napari/issues/4388)
- Add filename pattern support for folders [npe2 #155](https://github.com/napari/npe2/issues/155)

## Highlights

- Added sphinx-gallery (#4288)
- Add NAP process for major proposals (#4299)
- Add ColorEncoding privately with tests (#4357)
- Implement `TextManager` with `StringEncoding` (#4198)
- Add NAP1: institutional and funding partners (#4446)

## New Features

- Add alt-text to nbscreenshot output HTML images (#3825)
- Support of transformation parameters for the interaction box (#4301)
- Add function to show error in notification manager (#4369)

## Improvements

- Faster 2D shape layer creation (#3867)
- Npe2 enable/disable support (#4086)
- Use QFormLayout for layer control grid (#4195)
- Implement `TextManager` with `StringEncoding` (#4198)
- Add size argument to Viewer.screenshot() (#4201)
- fix error message when no reader available (#4254)
- Allow remote .tiff files to be loaded (#4284)
- refactor shape resizing logic and bugfix for #4262 (#4291)
- Accept None for scale (#4295)
- Rewrite ellipse discretization from scratch (#4330)
- Add ColorEncoding privately with tests (#4357)
- Update TextManager benchmarks to use string/features (#4364)
- add is_diagonal utility and Transform property (#4370)
- Add points size slider tooltip. (#4393)
- Split_channel makes base channel translucent, rest additive (#4394)
- Vispy 0.10 (#4401)
- Use syntax highlighter when printing stacktrace in GUI (#4414)
- Accelerate adding large numbers of points (#4549)
- use mip minip cutoff (#4556)
- Warn user when preferred plugin for a file is missing (#4545)
- Add preference saving from dialog for folders with extensions (#4535)
- Add filename pattern to reader associations to preference dialog (#4459)
- use imageio v2 api (#4537)

## Bug Fixes

- Fix erroneous point deletion when pressing delete key on layer (#4259)
- Bugfix: Divide by zero error making empty shapes layer (#4267)
- Bugfix: Conversion between Label and Image with original scaling (#4272)
- Address concurrent refresh in plugin list during multiple (un)installs (#4283)
- Delay import of _npe2 module in napari.__main__ to prevent duplicate discovery of plugins (#4311)
- Fix black line ellipse (#4312)
- Fix Labels.fill when Labels.data is an xarray.DataArray (#4314)
- Fix image and label layer values reported in GUI status bar when display is 3D (#4315)
- Quick fix for colormap updates not updating QtColorBox. (#4321)
- Update `black` version because of break of private API in its dependency (#4327)
- Fix progress update signature (#4333)
- move pixel center offset code into _ImageBase (#4352)
- Fix TextManager to work with vispy when using string constants (#4362)
- Fix format string encoding for all numeric features    (#4363)
- Bugfix/broadcast projections by reducing number of axes (keepdims=False) (#4376)
- Correctly order vispy layers on insertion (#4433)
- napari --info: list npe2 plugins (#4445)
- Bugfix/Add affine to base_dict via _get_base_state() (#4453)
- Fix layer control pop-up issue (#4460)
- fix Re-setting shapes data to initial data fails, but only in 3D (#4550)
- Make sure we pass plugin through if opening file as stack (#4515)
- Fix update of plugins and disable update button if not available on conda forge (for bundle) (#4512)
- Connect napari events first to EventEmitter (#4480)
- Fix AttributeError: 'LayerList' object has no attribute 'name' (#4276)
- Fix _BaseEventedItemModel.flags (#4558)
- Bug fix: blending multichannel images and 3D points (#4567)
- Fix checkable menu entries when using PySide2 backend (#4581)

## Documentation

- New Example: Creating reproducible screenshots with a viewer loop (#3947)
- add workshops (#4188)
- Replace image pyramid with multiscale image in the docs. (#4202)
- Uniform install instructions. (#4206)
- Use features instead of properties in `bbox_annotator` example (#4218)
- DOC: pep on python.org have moved. (#4237)
- Fix quick start links (#4239)
- Add napari.yaml to first plugin file layout (#4243)
- Improve "index" pages content (#4251)
- Fix links in docs (#4257)
- Bring back example notebook from back in time. (#4261)
- Fix README links Contributing Guide, Mission&Values, Code of Conduct, & GovModel (#4269)
- Minor copy update: Usage page (#4278)
- Minor copy update: Segmentation tutorial page (#4279)
- Minor copy update: Annotations tutorial page (#4280)
- Minor copy update: Tracking tutorial page  (#4282)
- Add napari.utils.notifications to the API docs (#4286)
- Added sphinx-gallery (#4288)
- Add NAP process for major proposals (#4299)
- Update best_practices.md (#4305)
- Fix broken link and adds packaging page to toc (#4335)
- Add napari.utils.events to API doc (#4338)
- add alt text workshop (#4373)
- Add and/or update documentation alt text (#4375)
- Add napari.window to API docs (#4379)
- Convert remaining .gifs to .webm (#4392)
- Add naps to the TOC (#4407)
- DOC Fix Broken links in the governance section of README (#4408)
- DOC Fix error in Using the image layer > A simple example (#4411)
- DOC Small fixes in 'Using the image layer' (#4418)
- Fix docs warnings related to NAPs (#4429)
- Add parser for Events section in docstrings (#4430)
- Fixes several sphinx warnings. (#4432)
- DOC Fix typo in 'Using the shapes layer' (#4438)
- Fix events rendering in docs for components.LayerList (#4442)
- Add NAP1: institutional and funding partners (#4446)
- Update to the documentation: add viewer.dims.current_step tips (#4454)
- Add information about new file opening behaviour (#4516)

## API Changes

- Update file opening behavior to ensure consistency across command line and GUI. (#4347)
- Warn user when preferred plugin for a file is missing (#4545)
- Make `builtins` default plugin for `viewer.open` (#4574)

## UI Changes

- Hide console toggle button and ignore corresponding keybinding for ipython
  (#4240) (Note: previously, this button was present but opened an empty/broken
  console, so this is strictly an improvement!)
- Allow resizing left dock widgets (#4368)
- Add filename pattern to reader associations to preference dialog (#4459)
- Add preference saving from dialog for folders with extensions #4535
- Make sure npe2 and npe1 builtins are available in dialogs (#4575)
- Open reader dialog when running napari from shell (#4569)

## Deprecations


## Build Tools

- singularity and docker container images from CI (#3965)
- Test bundle installation in CI (#4307)
- Use conda-forge/napari-feedstock source on main (#4309)
- add project_urls to setup.cfg metadata to improve project metadata on PyPI (#4317)
- Fix minreq test take 3. (#4329)
- `bundle_conda`: ignore unlink errors on cleanup (#4387)
- Move nap flowchart to lfs (#4403)
- Use installer version instead of napari version for default paths (#4444)
- add custom final condarc to bundle (#4447)
- Add doc specific Makefile (#4452)
- Set `TMP` on Windows+Mamba subprocesses if not set (#4462)
- Update test_typing.yml (#4475)
- Fix make-typestubs: use union for type hint instead of '|' (#4476)
- [conda] rework how plugin install/remove subprocesses receive the parent environment (#4520)
- [conda] revert default installation path (#4525)
- Pin vispy to <0.11 to prevent future breakages (#4594)

## Other Pull Requests

- adds citation file (#3470)
- Add tests for _npe2.py (#4103)
- Decrease LFS size, gif -> webm. (#4207)
- Run PNG crush on all Pngs. (#4208)
- Refactor toward fixing local value capturing. (#4212)
- Minor error message improvement. (#4219)
- Bump npe2 to 0.2.0 and fix typing tests (#4241)
- Remove headless test ignore, move orient_plane_normal test (#4245)
- [pre-commit.ci] pre-commit autoupdate (#4255)
- catch elementwise comparison warning that now shows frequently on layer creation (#4256)
- fix octree imports (#4264)
- Raise error when binding a button to a generator function (#4265)
- MAINT: coverage lines +1 (#4297)
- bump scipy minimum requirement from 1.4.0 to 1.4.1 (#4310)
- MAINT: separate ImportError from ModuleNotFoundError (#4339)
- [pre-commit.ci] pre-commit autoupdate (#4354)
- Remove 'of' from 'in this example of we will' (#4356)
- Fix npe2 import according to 0.3.0 deprecation warning (#4367)
- [pre-commit.ci] pre-commit autoupdate (#4378)
- add test for generate_3D_edge_meshes (#4416)
- Fix mypy error in CI (#4439)
- Make npe2 writer test more lenient (#4457)

## 33 authors added to this release (alphabetical)

- [aeisenbarth](https://github.com/napari/napari/commits?author=aeisenbarth) - @aeisenbarth
- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andrey Aristov](https://github.com/napari/napari/commits?author=aaristov) - @aaristov
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [chili-chiu](https://github.com/napari/napari/commits?author=chili-chiu) - @chili-chiu
- [Chris Wood](https://github.com/napari/napari/commits?author=cwood1967) - @cwood1967
- [David Stansby](https://github.com/napari/napari/commits?author=dstansby) - @dstansby
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Eric Perlman](https://github.com/napari/napari/commits?author=perlman) - @perlman
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Gregory Lee](https://github.com/napari/napari/commits?author=grlee77) - @grlee77
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Isabela Presedo-Floyd](https://github.com/napari/napari/commits?author=isabela-pf) - @isabela-pf
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Jan-Hendrik Müller](https://github.com/napari/napari/commits?author=kolibril13) - @kolibril13
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justin Kiggins](https://github.com/napari/napari/commits?author=neuromusic) - @neuromusic
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lucy Liu](https://github.com/napari/napari/commits?author=lucyleeow) - @lucyleeow
- [Marc Boucsein](https://github.com/napari/napari/commits?author=MBPhys) - @MBPhys
- [Marcelo Zoccoler](https://github.com/napari/napari/commits?author=zoccoler) - @zoccoler
- [Martin Weigert](https://github.com/napari/napari/commits?author=maweigert) - @maweigert
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [pre-commit-ci[bot]](https://github.com/napari/napari/commits?author=pre-commit-ci[bot]) - @pre-commit-ci[bot]
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Tom di Mino](https://github.com/napari/napari/commits?author=tdimino) - @tdimino
- [Tru Huynh](https://github.com/napari/napari/commits?author=truatpasteurdotfr) - @truatpasteurdotfr
- [Yuki Mochizuki](https://github.com/napari/napari/commits?author=2dx) - @2dx
- [Ziyang Liu](https://github.com/napari/napari/commits?author=potating-potato) - @potating-potato

## 42 reviewers added to this release (alphabetical)

- [Alan R Lowe](https://github.com/napari/napari/commits?author=quantumjot) - @quantumjot
- [alisterburt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andrea PIERRÉ](https://github.com/napari/napari/commits?author=kir0ul) - @kir0ul
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [chili-chiu](https://github.com/napari/napari/commits?author=chili-chiu) - @chili-chiu
- [David Stansby](https://github.com/napari/napari/commits?author=dstansby) - @dstansby
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Eric Perlman](https://github.com/napari/napari/commits?author=perlman) - @perlman
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Gregory Lee](https://github.com/napari/napari/commits?author=grlee77) - @grlee77
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Hagai Har-Gil](https://github.com/napari/napari/commits?author=HagaiHargil) - @HagaiHargil
- [Hector](https://github.com/napari/napari/commits?author=hectormz) - @hectormz
- [Isabela Presedo-Floyd](https://github.com/napari/napari/commits?author=isabela-pf) - @isabela-pf
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Jan-Hendrik Müller](https://github.com/napari/napari/commits?author=kolibril13) - @kolibril13
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justin Kiggins](https://github.com/napari/napari/commits?author=neuromusic) - @neuromusic
- [Justine Larsen](https://github.com/napari/napari/commits?author=justinelarsen) - @justinelarsen
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lucy Liu](https://github.com/napari/napari/commits?author=lucyleeow) - @lucyleeow
- [Lucy Obus](https://github.com/napari/napari/commits?author=LCObus) - @LCObus
- [Lukasz Migas](https://github.com/napari/napari/commits?author=lukasz-migas) - @lukasz-migas
- [Marc Boucsein](https://github.com/napari/napari/commits?author=MBPhys) - @MBPhys
- [Marcelo Zoccoler](https://github.com/napari/napari/commits?author=zoccoler) - @zoccoler
- [Martin Weigert](https://github.com/napari/napari/commits?author=maweigert) - @maweigert
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Nathan Clack](https://github.com/napari/napari/commits?author=nclack) - @nclack
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Peter Boone](https://github.com/napari/napari/commits?author=boonepeter) - @boonepeter
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Robert Haase](https://github.com/napari/napari/commits?author=haesleinhuepf) - @haesleinhuepf
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Tru Huynh](https://github.com/napari/napari/commits?author=truatpasteurdotfr) - @truatpasteurdotfr
- [Volker Hilsenstein](https://github.com/napari/napari/commits?author=VolkerH) - @VolkerH
- [Ziyang Liu](https://github.com/napari/napari/commits?author=potating-potato) - @potating-potato

