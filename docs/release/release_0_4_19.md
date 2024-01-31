# napari 0.4.19

We're happy to announce the release of napari 0.4.19!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://napari.org/stable/

## Highlights

This release mostly contains a lot of bug fixes and performance improvements.
But look out for 0.5.0, coming to a software repository near you — we expect to
release a lot of new features then!

### BIG improvements to the Labels layer

[#3308](https://github.com/napari/napari/pull/3308) closed *many* long-standing
bugs in the handling of colors in the Labels layer: the color swatch in the
Labels controls now always matches the color on the canvas, direct color
mapping with thousands of colors works fine, and shuffling colors (when two
touching labels coincidentally had the same color) is now much more likely to
map them to different colors. 🎨🚀🚀🚀

Unfortunately, the fix turned out to have rather
terrible consequences for the rendering performance of 3D Labels, and each time
we fixed one thing something else reappeared in a game of bug fix whack-a-mole
that any programmer would recognize. A great many
fixes later (
[#6411](https://github.com/napari/napari/pull/6411),
[#6439](https://github.com/napari/napari/pull/6439),
[#6459](https://github.com/napari/napari/pull/6459),
[#6460](https://github.com/napari/napari/pull/6460),
[#6461](https://github.com/napari/napari/pull/6461),
[#6467](https://github.com/napari/napari/pull/6467),
[#6479](https://github.com/napari/napari/pull/6479),
[#6520](https://github.com/napari/napari/pull/6520),
[#6540](https://github.com/napari/napari/pull/6540),
[#6571](https://github.com/napari/napari/pull/6571),
[#6580](https://github.com/napari/napari/pull/6580),
[#6596](https://github.com/napari/napari/pull/6596),
[#6602](https://github.com/napari/napari/pull/6602),
[#6607](https://github.com/napari/napari/pull/6607),
[#6616](https://github.com/napari/napari/pull/6616),
[#6618](https://github.com/napari/napari/pull/6618)
) — thank you for your patience! 😅 — Labels are faster than ever
*and* color accurate. *But*, to get the best performance, if you can use
8- or 16-bit integers as your data type, you should do so, and if not, you
should install *numba*, a just-in-time compiler for numerical code in Python.
(Ultimately, the data sent to the GPU will be 8- or 16-bit, so if you use a
larger data type, you will pay some conversion cost.)

These improvements in color handling are accompanied by updates to the Labels
API. You can now easily set a specific color cycle for Labels data. For
example, to use the famous [Glasbey look-up table/color
cycle](https://onlinelibrary.wiley.com/doi/10.1002/col.20327), you can combine
the [`glasbey`](https://pypi.org/project/glasbey/) Python package with the new
`CyclicLabelColormap` API:

```python
import glasbey
from napari.utils.colormaps import CyclicLabelColormap
# ...
labels_layer = viewer.add_labels(
    segmentation, colormap=CyclicLabelColormap(glasbey.create_palette(256))
)
```

See the ["Deprecations" section below](#Deprecations) for more information on
the new API. ([#6542](https://github.com/napari/napari/pull/6542))

### More on colormaps!

Yes, this is the colors update! 🌈

Making image layers with linear colormaps using custom colors is easier than
ever! You can just do `viewer.add_image(data, color='turquoise')` to get a
black-to-turquoise linear colormap for that image. For the full list of colors
available, see the
[VisPy color dict](https://github.com/vispy/vispy/blob/269ed1ac4d8126421fd5a7eb06a2996d63f46b17/vispy/color/_color_dict.py#L181)
([#6102](https://github.com/napari/napari/pull/6102)). You can also pass in an
RGB hex color prefixed with `#`, as in
`napari.imshow(data, colormap=`#88ff1a`)`.

(For an amusing side note, though, check out the [API Changes](#api-changes)
note related to this PR. 😅)

### Some technical stuff

If you were worried about those pesky "public access to `qt_viewer` will be
removed" warnings, fret not! Its removal has been postponed until at least
0.6.0! We want to spend more time working with the community to ensure your
use case is supported before pulling out the rug. 🤝 If you are using the
`qt_viewer` because we don't have another public API to do what you need,
please [raise an issue](https://github.com/napari/napari/issues/new) so we can
make sure your use case is supported before we remove it.

Finally, although we still use pydantic 1.0 objects internally, napari
installs correctly with both pydantic v1 and pydantic v2. If you wanted to
upgrade your napari library or plugin to use Pydantic 2, now you can!

Note though, for 0.4.19, the napari bundled app still ships with Pydantic 1.x.
However, we will bundle v2 starting with 0.5.0, so if you use Pydantic
internally, now is a good time to check that you are compatible either v2 or
both v1 and v2
([#6358](https://github.com/napari/napari/pull/6358)).

### Onwards!

As always, napari is developed by the community, for the community. We want to
hear from you and help you get your napari visualization and/or plugin use done
faster! And if napari is missing something you need, we can help you add it! So
please remember to ask [questions on
image.sc](https://forum.image.sc/tag/napari), join our [Zulip chat
room](https://napari.zulipchat.com/), come to our [community
meetings](https://napari.org/stable/community/meeting_schedule.html), or [tag
us on Mastodon](https://fosstodon.org/@napari)!

Read on for the full list of changes that went into this release.

- Postpone qt_viewer deprecation to 0.6.0 ([napari/napari/#6283](https://github.com/napari/napari/pull/6283))

## New Features

- Add show_debug notification ([napari/napari/#5101](https://github.com/napari/napari/pull/5101))
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
- Fix lagging 3d view for big data in auto color mode ([napari/napari/#6411](https://github.com/napari/napari/pull/6411))
- Fix cycle in _update_draw/_set_highlight for Points and Shapes (high CPU background usage) ([napari/napari/#6425](https://github.com/napari/napari/pull/6425))
- Update performance and reduce memory usage for big Labels layer in direct color mode ([napari/napari/#6439](https://github.com/napari/napari/pull/6439))
- Add _data_to_texture method to LabelColormap and remove caching of (u)int8 and (uint16) ([napari/napari/#6602](https://github.com/napari/napari/pull/6602))

## Bug Fixes

- Use a shader for low discrepancy label conversion ([napari/napari/#3308](https://github.com/napari/napari/pull/3308))
- Workaround Qt bug on Windows with fullscreen mode in some screen resolutions/scaling configurations ([napari/napari/#5401](https://github.com/napari/napari/pull/5401))
- Fix point selection highlight ([napari/napari/#5737](https://github.com/napari/napari/pull/5737))
- Fix shapes interactivity with scale != 1 (selection, rotate/resize) ([napari/napari/#5802](https://github.com/napari/napari/pull/5802))
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
- Bugfix: Account for multiscale for labels in 3d ([napari/napari/#6317](https://github.com/napari/napari/pull/6317))
- Update example scripts (magicgui with threads) ([napari/napari/#6353](https://github.com/napari/napari/pull/6353))
- Exclude the loaded property when linking two layers ([napari/napari/#6377](https://github.com/napari/napari/pull/6377))
- Fix problem with transform box of multiscale image  ([napari/napari/#6390](https://github.com/napari/napari/pull/6390))
- Fix cycle in _update_draw/_set_highlight for Points and Shapes (high CPU background usage) ([napari/napari/#6425](https://github.com/napari/napari/pull/6425))
- Do not run macos process renaming if debugger is loaded ([napari/napari/#6437](https://github.com/napari/napari/pull/6437))
- Fix bounding box transforms when multiscale layer corner goes below zero ([napari/napari/#6438](https://github.com/napari/napari/pull/6438))
- Update performance and reduce memory usage for big Labels layer in direct color mode ([napari/napari/#6439](https://github.com/napari/napari/pull/6439))
- Fix `breakpoint()` function after console import ([napari/napari/#6443](https://github.com/napari/napari/pull/6443))
- fix problem with alligned overlay ([napari/napari/#6445](https://github.com/napari/napari/pull/6445))
- Fix guess_continuous to not assign continuous for string categories ([napari/napari/#6452](https://github.com/napari/napari/pull/6452))
- Fix casting uint32 to vispy dtype for image layers ([napari/napari/#6456](https://github.com/napari/napari/pull/6456))
- Fix thumbnail for auto color mode in labels ([napari/napari/#6459](https://github.com/napari/napari/pull/6459))
- Fix label color shuffling by also updating colormap size ([napari/napari/#6460](https://github.com/napari/napari/pull/6460))
- Fix direct colormap ([napari/napari/#6461](https://github.com/napari/napari/pull/6461))
- Use views rather than CPU-based hashing for 8- and 16-bit Labels data ([napari/napari/#6467](https://github.com/napari/napari/pull/6467))
- remove main window from instances list only if close event is accepted ([napari/napari/#6468](https://github.com/napari/napari/pull/6468))
- Update allowed selected_label range when data dtype changes ([napari/napari/#6479](https://github.com/napari/napari/pull/6479))
- Run test suite with optional dependencies and fix tests when `triangle` is installed ([napari/napari/#6488](https://github.com/napari/napari/pull/6488))
- Do not use native dialog for reset shortcuts ([napari/napari/#6493](https://github.com/napari/napari/pull/6493))
- Pass key event from Main window to our internal mechanism v0.4.19 ([napari/napari/#6507](https://github.com/napari/napari/pull/6507))
- Fix problem with invalidate cache  ([napari/napari/#6520](https://github.com/napari/napari/pull/6520))
- Reset single step and decimals on reset range slider in popup ([napari/napari/#6523](https://github.com/napari/napari/pull/6523))
- Fix label direct mode for installation without numba ([napari/napari/#6571](https://github.com/napari/napari/pull/6571))
- Fix labels mapping cache by filling it with background, not 0 ([napari/napari/#6580](https://github.com/napari/napari/pull/6580))
- Initialize ndim value for Points layer using data shape if available ([napari/napari/#6593](https://github.com/napari/napari/pull/6593))
- Fix wrong working interpolation of labels in 3d ([napari/napari/#6596](https://github.com/napari/napari/pull/6596))
- Fix points `changed` event emission ([napari/napari/#6611](https://github.com/napari/napari/pull/6611))
- Fix `Labels.data_setitem` setting of view by taking dims order into account ([napari/napari/#6616](https://github.com/napari/napari/pull/6616))
- Fixing data_setitem function of label layer ([napari/napari/#6618](https://github.com/napari/napari/pull/6618))
- Fix points selection ([napari/napari/#6621](https://github.com/napari/napari/pull/6621))
- Select only linked layers already present in layer list ([napari/napari/#6622](https://github.com/napari/napari/pull/6622))
- Bug fixes to multiple issues with linked layers ([napari/napari/#6623](https://github.com/napari/napari/pull/6623))
- Fix rendering of vertexes of shape layers with small scale ([napari/napari/#6628](https://github.com/napari/napari/pull/6628))

## API Changes


[#6102](https://github.com/napari/napari/pull/6102) added the ability to set
linear colormaps from black to a named color by using `colormap="color-name"`
syntax. It turns out that "orange" is both the name of a
white-to-orange colormap in VisPy, and one of the color names in the color
dictionary, therefore implicitly a *black-to-orange* colormap! We decided to use
the new, color-name behavior in this update. So if you are wondering why your
`imshow(data, colormap='orange')` calls look different — this is why.

In [#6178](https://github.com/napari/napari/pull/6178), the "action" type of
events emitted when editing Shapes or Points was changed to be more granular.
The types are no longer "add", "remove", and "change", but "adding", "added",
"removing", "removed", "changing", and "changed". This gives listeners more
control over when to take action in response to an event.


## Deprecations

[#6542](https://github.com/napari/napari/pull/6542) made a number of
deprecations to the Labels API to simplify it. Rather than having color-related
properties strewn all over the layer, color control is moved strictly to the
layer's `colormap`. Here is the full list of deprecated attributes and their
replacements:

- num_colors: `layer.num_colors` becomes `len(layer.colormap)`.
  `layer.num_colors = n` becomes `layer.colormap = label_colormap(n)`.
- `napari.utils.colormaps.LabelColormap` is deprecated and has been renamed to
  `napari.utils.colormaps.CyclicLabelColormap`.
- color: `layer.color` becomes `layer.colormap.color_dict`.
  `layer.color = color_dict` becomes
  `layer.colormap = DirectLabelColormap(color_dict)`.
- _background_label: `layer._background_label` is now at
  `layer.colormap.background_value`.
- color_mode: `layer.color_mode` is set by setting the colormap using the
  corresponding colormap type (`CyclicLabelColormap` or `DirectLabelColormap`;
  these classes can be imported from `napari.utils.colormaps`.).
- `seed`: was only used for shifting labels around in [0, 1]. It is
  superseded by `layer.new_colormap()` which was implemented in
  [#6460](https://github.com/napari/napari/pull/6460).

- Postpone qt_viewer deprecation to 0.6.0 ([napari/napari/#6283](https://github.com/napari/napari/pull/6283))

## Build Tools

- Vispy 0.14 ([napari/napari/#6214](https://github.com/napari/napari/pull/6214))

## Documentation

- Add HIP workshop to documentation/workshops ([napari/napari/#5117](https://github.com/napari/napari/pull/5117))
- Update README.md for conda install change ([napari/napari/#6123](https://github.com/napari/napari/pull/6123))
- Cherry-pick docs for 0.4.19 release  ([napari/napari/#6384](https://github.com/napari/napari/pull/6384))
- Add 0.4.19 release notes ([napari/napari/#6376](https://github.com/napari/napari/pull/6376))
- Update docs contribution guide for two-repo setup ([napari/docs/#5](https://github.com/napari/docs/pull/5))
- Improve makefile ([napari/docs/#41](https://github.com/napari/docs/pull/41))
- add foundation grant onboarding workshop ([napari/docs/#55](https://github.com/napari/docs/pull/55))
- Add 'html-live' make action to support live reload workflows ([napari/docs/#75](https://github.com/napari/docs/pull/75))
- Fixes formatting for the contributing documentation section ([napari/docs/#79](https://github.com/napari/docs/pull/79))
- Move napari workshop template link to top of page ([napari/docs/#90](https://github.com/napari/docs/pull/90))
- Explain how to add new examples to the gallery ([napari/docs/#137](https://github.com/napari/docs/pull/137))
- Add instructions on how to use docs-xvfb ([napari/docs/#138](https://github.com/napari/docs/pull/138))
- Update annotate_points.md ([napari/docs/#145](https://github.com/napari/docs/pull/145))
- Add more information to the documentation contribution guide ([napari/docs/#157](https://github.com/napari/docs/pull/157))
- Add instructions to build napari docs on Windows ([napari/docs/#158](https://github.com/napari/docs/pull/158))
- Add information about constraint usage to install older napari release ([napari/docs/#193](https://github.com/napari/docs/pull/193))
- Fix typo of points instead of shapes ([napari/docs/#195](https://github.com/napari/docs/pull/195))
- Improve titles of fundamentals tutorials ([napari/docs/#196](https://github.com/napari/docs/pull/196))
- NAP 7: Key Binding Dispatch ([napari/docs/#200](https://github.com/napari/docs/pull/200))
- Update installation instructions for the new conda-forge packages and other changes in our packaging infra ([napari/docs/#202](https://github.com/napari/docs/pull/202))
- make Talley emeritus SC ([napari/docs/#204](https://github.com/napari/docs/pull/204))
- Use napari_scraper instead of qtgallery ([napari/docs/#207](https://github.com/napari/docs/pull/207))
- Move contributing resources to top-level navbar ([napari/docs/#208](https://github.com/napari/docs/pull/208))
- Add roadmap board link to Roadmaps page ([napari/docs/#212](https://github.com/napari/docs/pull/212))
- Fix getting started in napari linking to the unittest getting started page ([napari/docs/#217](https://github.com/napari/docs/pull/217))
- Update core developer list ([napari/docs/#219](https://github.com/napari/docs/pull/219))
- Add note on milestones for PRs ([napari/docs/#221](https://github.com/napari/docs/pull/221))
- Add nap for telemetry ([napari/docs/#226](https://github.com/napari/docs/pull/226))
- Fix titles on Getting Started section of user guide ([napari/docs/#228](https://github.com/napari/docs/pull/228))
- Update Kyle's tag on the core devs page ([napari/docs/#232](https://github.com/napari/docs/pull/232))
- DOC Minor fixes to nap 6 doc ([napari/docs/#233](https://github.com/napari/docs/pull/233))
- Update ndisplay title ([napari/docs/#235](https://github.com/napari/docs/pull/235))
- Remove sub-sub section heading from rendering guide ([napari/docs/#236](https://github.com/napari/docs/pull/236))
- Update selection instructions in Points tutorial for Shift-A keybinding ([napari/docs/#238](https://github.com/napari/docs/pull/238))
- fix outdated dimension sliders documentation ([napari/docs/#241](https://github.com/napari/docs/pull/241))
- Update napari-workshops.md ([napari/docs/#243](https://github.com/napari/docs/pull/243))
- Add NAP for multicanvas ([napari/docs/#249](https://github.com/napari/docs/pull/249))
- [Fix error] Image layers can't have converted data type using contextual menu, only Labels ([napari/docs/#252](https://github.com/napari/docs/pull/252))
- Installation guide: Mention slow first launch time ([napari/docs/#253](https://github.com/napari/docs/pull/253))
- Update to use napari-sphinx-theme 0.3.0 ([napari/docs/#267](https://github.com/napari/docs/pull/267))
- [NAP-8] delete :orphan: ([napari/docs/#269](https://github.com/napari/docs/pull/269))
- add link to video from EMBO workshop ([napari/docs/#273](https://github.com/napari/docs/pull/273))
- Improve flow of install page ([napari/docs/#274](https://github.com/napari/docs/pull/274))
- Add Kyle to steering council, make Talley emeritus ([napari/docs/#322](https://github.com/napari/docs/pull/322))
- move self to emeritus ([napari/docs/#323](https://github.com/napari/docs/pull/323))
- Update working groups leads ([napari/docs/#327](https://github.com/napari/docs/pull/327))
- Close calendar event popover when clicking outside it ([napari/docs/#337](https://github.com/napari/docs/pull/337))
- Move Nick and Loic to emeritus, sort emeritus core devs ([napari/docs/#339](https://github.com/napari/docs/pull/339))
- Update conf.py to display the announcement banner ([napari/docs/#342](https://github.com/napari/docs/pull/342))
- Rename `myModal` to `eventDetailBackground` in meetings schedule ([napari/docs/#343](https://github.com/napari/docs/pull/343))
- add melissa to core devs ([napari/docs/#345](https://github.com/napari/docs/pull/345))

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
- Do not repeat warnings in GUI ([napari/napari/#6396](https://github.com/napari/napari/pull/6396))
- Fix drawing timer ([napari/napari/#6400](https://github.com/napari/napari/pull/6400))
- Reraise warnings in proxy ([napari/napari/#6408](https://github.com/napari/napari/pull/6408))
- Bump napari console to ensure users get latest bug fixes ([napari/napari/#6442](https://github.com/napari/napari/pull/6442))
- [maint] update Dockerfile with current installation of Xpra ([napari/napari/#6463](https://github.com/napari/napari/pull/6463))
- [Maint, v0.4.19] Use python 3.11 for manifest check ([napari/napari/#6497](https://github.com/napari/napari/pull/6497))
- Add copy operator to fix memory benchmarks ([napari/napari/#6530](https://github.com/napari/napari/pull/6530))
- Check in LabelColormap that fewer than 2**16 colors are requested ([napari/napari/#6540](https://github.com/napari/napari/pull/6540))
- [Maint] Update build_docs workflow to match napari/docs ([napari/napari/#6547](https://github.com/napari/napari/pull/6547))
- Moving IntensityVisualizationMixin from _ImageBase to Image ([napari/napari/#6548](https://github.com/napari/napari/pull/6548))
- test: [Automatic] Constraints upgrades: `app-model`, `babel`, `certifi`, `dask`, `fsspec`, `hypothesis`, `imageio`, `ipython`, `jsonschema`, `lxml`, `magicgui`, `matplotlib`, `numba`, `numpy`, `pandas`, `pillow`, `pint`, `psutil`, `psygnal`, `pydantic`, `pygments`, `pyqt6`, `pytest-qt`, `qtconsole`, `rich`, `scipy`, `tensorstore`, `tifffile`, `torch`, `virtualenv`, `wrapt`, `xarray` ([napari/napari/#6559](https://github.com/napari/napari/pull/6559))
- Do not require triangle on macos arm ([napari/napari/#6603](https://github.com/napari/napari/pull/6603))
- No-cache fast painting ([napari/napari/#6607](https://github.com/napari/napari/pull/6607))
- test: [Automatic] Constraints upgrades: `dask`, `hypothesis`, `ipython`, `jsonschema`, `lxml`, `npe2`, `numpy`, `pillow`, `psutil`, `pytest`, `scipy`, `tensorstore`, `toolz`, `xarray` ([napari/napari/#6608](https://github.com/napari/napari/pull/6608))
- Ignore pandas pyarrow warning ([napari/napari/#6609](https://github.com/napari/napari/pull/6609))
- test: [Automatic] Constraints upgrades: `hypothesis`, `pydantic`, `tifffile` ([napari/napari/#6630](https://github.com/napari/napari/pull/6630))
- Update docs to suggest python 3.10 install ([napari/docs/#246](https://github.com/napari/docs/pull/246))


## 20 authors added to this release (alphabetical)

- [akuten1298](https://github.com/napari/napari/commits?author=akuten1298) - @akuten1298
- [Andrew Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Ashley Anderson](https://github.com/napari/napari/commits?author=aganders3) - @aganders3
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
- [Leopold Franz](https://github.com/napari/napari/commits?author=leopold-franz) - @leopold-franz
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Lucy Liu](https://github.com/napari/napari/commits?author=lucyleeow) - @lucyleeow
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Robert Haase](https://github.com/napari/napari/commits?author=haesleinhuepf) - @haesleinhuepf
- [Sean Martin](https://github.com/napari/napari/commits?author=seankmartin) - @seankmartin
- [Wouter-Michiel Vierdag](https://github.com/napari/napari/commits?author=melonora) - @melonora


## 17 reviewers added to this release (alphabetical)

- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Andrew Sweet](https://github.com/napari/napari/commits?author=andy-sweet) - @andy-sweet
- [Ashley Anderson](https://github.com/napari/napari/commits?author=aganders3) - @aganders3
- [Daniel Althviz Moré](https://github.com/napari/napari/commits?author=dalthviz) - @dalthviz
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Egor Zindy](https://github.com/napari/napari/commits?author=zindy) - @zindy
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Gonzalo Peña-Castellanos](https://github.com/napari/napari/commits?author=goanpeca) - @goanpeca
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/napari/commits?author=jaimergp) - @jaimergp
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Lorenzo Gaifas](https://github.com/napari/napari/commits?author=brisvag) - @brisvag
- [Matthias Bussonnier](https://github.com/napari/napari/commits?author=Carreau) - @Carreau
- [Melissa Weber Mendonça](https://github.com/napari/napari/commits?author=melissawm) - @melissawm
- [Peter Sobolewski](https://github.com/napari/napari/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Wouter-Michiel Vierdag](https://github.com/napari/napari/commits?author=melonora) - @melonora


## 18 docs authors added to this release (alphabetical)

- [Alister Burt](https://github.com/napari/docs/commits?author=alisterburt) - @alisterburt
- [Ashley Anderson](https://github.com/napari/docs/commits?author=aganders3) - @aganders3
- [chili-chiu](https://github.com/napari/docs/commits?author=chili-chiu) - @chili-chiu
- [Constantin Pape](https://github.com/napari/docs/commits?author=constantinpape) - @constantinpape
- [David Stansby](https://github.com/napari/docs/commits?author=dstansby) - @dstansby
- [dgmccart](https://github.com/napari/docs/commits?author=dgmccart) - @dgmccart
- [Grzegorz Bokota](https://github.com/napari/docs/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/docs/commits?author=jaimergp) - @jaimergp
- [Juan Nunez-Iglesias](https://github.com/napari/docs/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/docs/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/docs/commits?author=kne42) - @kne42
- [Lucy Liu](https://github.com/napari/docs/commits?author=lucyleeow) - @lucyleeow
- [Melissa Weber Mendonça](https://github.com/napari/docs/commits?author=melissawm) - @melissawm
- [Peter Sobolewski](https://github.com/napari/docs/commits?author=psobolewskiPhD) - @psobolewskiPhD
- [Robert Haase](https://github.com/napari/docs/commits?author=haesleinhuepf) - @haesleinhuepf
- [Sean Martin](https://github.com/napari/docs/commits?author=seankmartin) - @seankmartin
- [Talley Lambert](https://github.com/napari/docs/commits?author=tlambert03) - @tlambert03
- [Wouter-Michiel Vierdag](https://github.com/napari/docs/commits?author=melonora) - @melonora


## 18 docs reviewers added to this release (alphabetical)

- [Andrew Sweet](https://github.com/napari/docs/commits?author=andy-sweet) - @andy-sweet
- [Ashley Anderson](https://github.com/napari/docs/commits?author=aganders3) - @aganders3
- [Constantin Pape](https://github.com/napari/docs/commits?author=constantinpape) - @constantinpape
- [David Stansby](https://github.com/napari/docs/commits?author=dstansby) - @dstansby
- [Draga Doncila Pop](https://github.com/napari/docs/commits?author=DragaDoncila) - @DragaDoncila
- [Egor Panfilov](https://github.com/napari/docs/commits?author=soupault) - @soupault
- [Genevieve Buckley](https://github.com/napari/docs/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/docs/commits?author=Czaki) - @Czaki
- [Jaime Rodríguez-Guerra](https://github.com/napari/docs/commits?author=jaimergp) - @jaimergp
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

There are 5 new contributors for this release:

- akuten1298 [napari](https://github.com/napari/napari/commits?author=akuten1298) - @akuten1298
- dgmccart [docs](https://github.com/napari/docs/commits?author=dgmccart) - @dgmccart
- Egor Zindy [napari](https://github.com/napari/napari/commits?author=zindy) - @zindy
- Elena Pascal [napari](https://github.com/napari/napari/commits?author=elena-pascal) - @elena-pascal
- Leopold Franz [napari](https://github.com/napari/napari/commits?author=leopold-franz) - @leopold-franz
