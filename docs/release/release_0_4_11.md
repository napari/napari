# napari 0.4.11

We're happy to announce the release of napari 0.4.11!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights
We've known for a while that 3D interactivity is one of the areas where napari needed improvement (#515). This release introduces ways to use the mouse to interact with data in 3D (#3037). Features like label picking (#3074) and label painting/erasing (#3108) in 3D are just the beginning and we're excited to see where this takes us! For more details, please see the documentation at [https://napari.org/guides/stable/3D_interactivity.html](https://napari.org/guides/stable/3D_interactivity.html).

Our volume rendering functionality has been significantly improved and now includes the ability to render arbitrary planes through volumes (#3023) and add clipping planes to restrict rendering to a region of interest (#3140). For now, these features are marked as `experimental` and the API around their use is likely to change in future versions of napari.

Last but not least, some common operations are now much more accessible from the GUI thanks to a new context menu on the layer list (#2556 and #3028) and buttons for controlling image contrast limit scaling (#3022).

## New Features
- Add context menu on layer list, introduce `QtActionContextMenu`. (#2556)
- Add activity dialog and style progress bars (#2656)
- Add playback options to settings (#2933)
- Refactor settings manager to allow setting all preferences with env vars and context managers (#2932)
- Add autoscale modes to image layer model, and buttons to GUI (#3022)
- Arbitrary plane rendering prototype (#3023)
- Add view ray and labels selection in 3D (#3037)
- Add `add_<shape_type>` method for each shape type (#3076)
- Grid mode popup (#3084)
- Fix stubgen and package stubs in wheel/sdists (#3105)
- Add 3D fill, "mill", and "print" on top of #3074 (#3108)
- Add positive tail length to tracks layer (#3138)
- Arbitrary clipping planes for volumes in the image layer (#3140)
- Mask image from points layer (#3151)
- Add projections to layer context menu, allow grouping and nesting of menu items (#3028)

## Improvements
- Add `assign_[plugin]_to_extension` methods on plugin_manager.  (#2695)
- Use QDoubleRangeSlider from superqt package (#2752)
- Use labeled sliders from superqt (#2753)
- Shortcuts UI (#2864)
- Convert TextManager to an EventedModel (#2885)
- Make maximum brush size flexible (#2901)
- Allow layer to register action on double clicks. (#2907)
- Reduce numpy array traceback print (#2910)
- Provide manual deepcopy implementation for translations strings. (#2913)
- Make Points construction with properties consistent with setting properties (#2916)
- Add search field to plugin dialog  (#2923)
- Add initital support to install from conda/mamba (#2943)
- Shape Mouse refactor (#2950)
- Make handling properties more consistent across some layer types (#2957)
- Labels paintbrush now takes anisotropy into account (#2962)
- Remove mode guards for selection interactions in points (#2982)
- Emit data event when moving, adding or removing shapes or points (#2992)
- Add TypedMutableMapping and EventedDict (#2994)
- Add isosurface rendering to Labels (#3006)
- Remove mentions of _mode_history (2987) (#3008)
- Change opacity slider to float slider (#3016)
- Refactor the Point, Label and Shape Layer Mode logic. (#3050)
- Make flash effect feel more instant (#3060)
- Use enum values internally for settings. (#3063)
- Update vendored volume visual from vispy (#3064)
- Allow for multiple installs and update buttons to reflect state (#3067)
- Unify plugin wording and installer dialog to display only package/plugins (#3071)
- 3D label picking and label ID in status bar (#3074)
- Store unmaximized size if napari closes maximized (#3075)
- Change shapes default edge color to middle gray (#3113)
- Change default text overlay color to mid grey (#3114)
- Add 3D get_value to Shapes (#3117)
- Replace custom signals to accept/reject (#3120)
- Remove old utils/settings/constants file (#3122)
- NAPARI_CATCH_ERRORS disable notification manager (#3126)
- Save files after launch from ipython (#3130)
- Don't try to read unknown formats in builtin reader plugin (#3145)
- Make current viewer accessible from the napari module (#3149)
- Rename `Layer.plane` to `Layer.slicing_plane` (#3150)
- Update new label action to work with tensorstore arrays (#3153)
- Always raise PluginErrors (#3157)
- Establish better YAML encoding for settings (fix enum encoding issue). (#3163)
- Move `get_color` call to after `all_vals` have been cleared (#3173)
- Prevent highlight widget from emitting constant signals (#3175)
- Refactor preferences dialog to take advantage of evented settings methods (#3178)
- Use QElidingLabel from superqt (#3188)
- Use `QLargeIntSpinBox` from superqt, remove internal one (#3191)
- Catch and prune connections to stale Qt callbacks in events.py (#3193)
- Add checkbox to handle global plugin enable/disabled state (#3194)
- Print warning if error formatting in the console fails instead of ignoring errors. (#3201)
- Ensure we save a copy of existing value for undo (#3203)

## Bug Fixes
- Fix notification manager threading test (#2892)
- Pycharm blocking fix (#2905)
- Fix windows 37 test (#2909)
- Fix docstring, and type annotate. (#2912)
- Don't raise exception when failing to save qt resources. (#2919)
- Dix invalid yaml for docs workflow (#2920)
- Fix use of `default_factory` in settings (#2930)
- Close Qt progress bars when viewer is closed (#2931)
- Degrade gracefully to default when colormap is not recognized (#2936)
- Fix magicgui registration and circular imports (#2949)
- Fix error in `Viewer.reset_view()` with vispy 0.7 (#2958)
- Addressing case where argument to get_default_shape_type is empty list, addresses issue #2960 (#2961)
- Fix nD anisotropic painting when scale is negative (#2966)
- Ensure new height of expanded notification is larger than current height (#2981)
- Gracefully handle properties with `object` dtype (#2986)
- Fix scale decomp with composite (#2990)
- Fix behavior of return/escape on preferences dialog to accept/cancel (#2998)
- Fix EventedDict (#3011)
- Use compression=('zlib', 1) for new tifffile (#3040)
- Fix saving preferences (#3041)
- Use non-deprecated colormap in viewer cmap test (#3043)
- Fix Labels layer controls checkbox labels (#3046)
- Fix Layer.affine assignment and broadcasting (#3056)
- Fix problem with assigning affine with negative entries to  pyramids (#3088)
- Fix stubgen and package stubs in wheel/sdists (#3105)
- Fix opacity slider on shapes (#3109)
- Fix empty points layer with color cycle (#3110)
- Fix point deletion bug (#3119)
- Fix for get_value() with mixed dims (#3121)
- Fix settings reset breaking connections (creating a new instance of nested sub-models) (#3123)
- Fix plane serialisation (#3143)
- Bugfix in labels erasing (#3146)
- Bug fix for undo history in 3D painting (#3154)
- Don't clear blocked plugins when closing preferences dialog (#3164)
- Revert `Points` `remove_selected` always overwriting `self._value` to `None` (#3165)
- Fix window geometry loading bug, and make `ApplicationSettings` types more accurate (#3182)
- Fix missing import in napari.__init__.pyi (#3183)
- Fix incorrect window position storage (#3196)
- Fix incorrect use of dims_order when 3D painting (#3202)


## API Changes
- Remove brush shape (#3047)
- Enforce layer.metadata as dict (#3020)
- Use enum objects in EventedModel (#3112)


## UI Changes
- Remove keybindings dialog from help menu (#3048)
- Remove plugin sorter from plugin install dialog (#3069)
- Update Labels layer keybindings to be more ergonomic (#3072)


## Deprecations


## Build Tools, Tests, Documentation, and other Tasks
- Add imagecodecs to the bundle to open additional tiffs (#2895)
- Make ordering of releases manual (#2921)
- Add alister burt to team page (#2937)
- Use briefcase 0.3.1 on all platforms (#2980)
- Speedup one of the slowest test. (#2997)
- Update plugin guide with references and instructions for napari-hub (#3055)
- Skip progress indicator test when viewer is not shown (#3065)
- Add missing libraries in docker file and entrypoint (#3081)
- Update documentation regarding the hub (#3091)
- Block showing dialog in nongui test (#3127)
- Update about page (#3134)
- Adding new design issues template (#3142)
- Fix emoji for design template (#3161)
- Update design_related.md (#3162)
- Try to fix CI, change perfmon test strategy (#3167)
- Fix comprehensive tests (#3168)
- Fix `make_docs` action (#3169)
- Remove convert_app (#3171)
- Update team.md (#3176)
- Misc Doc format fixing (#3179)
- Add public meeting calendar to the docs (#3192)
- Don't start gui qt event loop when building docs (#3207)
- Add note detailing current octree support (#3208)
- Add napari_write_tracks to hook spec reference (#3209)
- Add 3d interactivity docs (#3210)
- Fix docs build again (#3211)


## 21 authors added to this release (alphabetical)

- [Abigail McGovern](https://github.com/napari/napari/commits?author=AbigailMcGovern) - @AbigailMcGovern
- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Lia Prins](https://github.com/napari/napari/commits?author=liaprins-czi) - @liaprins-czi
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lukasz Migas](https://github.com/napari/napari/commits?author=lukasz-migas) - @lukasz-migas
- [Marlene Elisa Da Vitoria Lobo](https://github.com/napari/napari/commits?author=marlene09) - @marlene09
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nathan Clack](https://github.com/napari/napari/commits?author=nclack) - @nclack
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Volker Hilsenstein](https://github.com/napari/napari/commits?author=VolkerH) - @VolkerH
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi


## 19 reviewers added to this release (alphabetical)

- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andy Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Jordão Bragantini](https://github.com/napari/napari/commits?author=JoOkuma) - @JoOkuma
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Lia Prins](https://github.com/napari/napari/commits?author=liaprins-czi) - @liaprins-czi
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lucy Obus](https://github.com/napari/napari/commits?author=LCObus) - @LCObus
- [Lukasz Migas](https://github.com/napari/napari/commits?author=lukasz-migas) - @lukasz-migas
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Nathan Clack](https://github.com/napari/napari/commits?author=nclack) - @nclack
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Pam](https://github.com/napari/napari/commits?author=ppwadhwa) - @ppwadhwa
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi

