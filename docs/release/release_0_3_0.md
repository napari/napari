# napari 0.3.0

We're happy to announce the release of napari 0.3.0!

napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant
GPU-based rendering), and the scientific Python stack (numpy, scipy).

This is the first "major" release since 0.2.0 and is the culmination of 6
months of work by our developer community. We have made a small number of
breaking changes to the API, and added several new capabilities to napari, so
we encourage you to read on for more details.

For more information, examples, and documentation, please visit our website:
https://napari.org

## Highlights

### Community and governance

After the 0.2.5 release, which was our first publicly-announced release, we
worked to rapidly turn napari into a mature library in the scientific Python
ecosystem. We added a [code of
conduct](https://napari.org/community/code_of_conduct.html), a [mission
and values
document](https://napari.org/docs/community/mission_and_values.html), and
adopted a [community governance
model](https://napari.org/docs/community/governance.html) (based on
scikit-image's, and since adopted with modifications by zarr). These are
accessible from our [developer resources
page](https://napari.org/developers/index.html), together with a [public
roadmap](https://napari.org/roadmaps/0_3.html) explaining where
the core team will devote effort in the coming months.

We are still humbled by the enthusiasm of the community response to napari and
we hope that the above documents will continue to encourage potential users to
join our community. We welcome contributions of all kinds and encourage you to
get in touch with us if you don't see your most wanted feature in our roadmap,
or as an issue in our [issue tracker](https://github.com/napari/napari/issues).

### Getting in touch

We joined the [image.sc forum](https://forum.image.sc) and actively monitor it
([under the "napari" tag](https://forum.image.sc/tags/napari)) to help users
with any issues they might have using napari and have discussions about
exciting ways to use napari. If you're new to napari and getting started this
is the first place you should go for help.

If you've found a napari bug or have a specific feature request, please let us
know in our [GitHub issues](https://github.com/napari/napari/issues).

### IO plugins

Contributors can now easily extend napari to open and save in a variety of file
formats, both local and remote, through our plugin architecture. The same file
formats as before are available to read (TIFF, most image file formats
supported by imageio, and zarr).  However, we can now *write* to all these
formats, and read and write point and shape annotations in .csv format.
Additionally, we have made it possible for anyone to create packages for napari
to read and write in any other formats through plugins. You can read about our
plugin architecture [here](https://napari.org/plugins/stable/index.html).

Want to drag and drop your favorite file format into napari and have it load
automatically? See [this
guide](https://napari.org/plugins/stable/for_plugin_developers.html) to
understand how to write your own plugin, see Jackson Brown's
[napari-aicsimageio](https://github.com/AllenCellModeling/napari-aicsimageio)
for an exemplar plugin, and get started with Talley's [cookiecutter napari
plugin](https://github.com/napari/cookiecutter-napari-plugin)!

Many thanks to Talley Lambert for driving this effort!

### Dockable widgets and magicgui

Another brainchild of Talley is our dockable widget architecture, which allows
you to pop out the napari UI elements from the main window, enabling, for
example, those on multi-monitor setups to have the toolbars on one monitor and
the main window in full-screen on another.

Even better, we have released a side package called
[magicgui](https://github.com/napari/magicgui) to allow you to create your own
dockable widgets with which to interact with napari without writing GUI code.
We are still working on standard models of interaction here (see our
[roadmap](https://napari.org/docs/developers/ROADMAP_0_3.html)), but you should
be able to get started creating useful user interfaces right now. [This
image.sc
post](https://forum.image.sc/t/integration-of-napari-module-subclass-plugin/36018/2)
by Talley provides a good overview of how to create interaction with napari
right now, and [this GitHub
answer](https://github.com/napari/napari/issues/1165#issuecomment-618013894)
explains how to embed a matplotlib plot within napari.

### Multiscale image handling

napari is now much better at handling large datasets. Viewing a large dataset
will no longer trigger automatic — but very slow, and often unnecessary —
generation of an image pyramid. Instead, we direct users to our [tutorial on
how to generate your own
pyramid](https://scikit-image.org/docs/dev/auto_examples/transform/plot_pyramid.html)
from `scikit-image`. For large arrays, users may want to look at the
[`dask.array.coarsen`
documentation](https://docs.dask.org/en/stable/generated/dask.array.coarsen.html)

If you submit an image pyramid, napari will automatically detect it as such. To
turn off automatic detection, you can now pass the `multiscale=True/False`
parameter to `add_image`. This replaces the `is_pyramid` parameter which has
now been removed. In the future, we aim to add multiscale capabilities to all
our layers.

We have also fixed the bug where too small a tile was shown to fill the entire
canvas.

### Points with properties

Points are no longer generic coordinates floating in space! Each point can have
its own personality and character :-). Specifically, each point can have an
arbitrary number of properties, and attributes such as size, face color, and
edge color can be determined by those properties. This makes it easier to
annotate multiple types of points in an image, such as different cell types. To
assign properties to points you can pass a dictionary as the `properties`
parameter to `add_points`.

### API changes and improvements

We are taking the opportunity of this major release to update a few APIs. We
hope that the number of users impacted by these changes will be small. In each
case, we provide an equivalent API for the same functionality.

- `viewer.add_path` has been renamed `viewer.open` and gained the ability to
  read to any layer type.
- `add_image(path=...)` and `add_labels(path=...)` have been removed. Users
  should use `viewer.open(...)` instead with the `layer_type` parameter to
  force the added data to be a particular layer type, e.g.
  `layer_type='image'` or `layer_type='labels'`.
- Image pyramids are no longer automatically generated when a dataset is large.
  This should not affect API compatibility but might affect performance. For
  most users, this should result in faster startup for large images.
- `add_image(..., is_pyramid=False)` is now `add_image(..., multiscale=False)`.
  This will allow us to use a consistent keyword argument when we add
  multiscale support for other layer types.
- `layer.to_svg()` has been removed. This functionality is now implemented with
  `viewer.save('path/to/layer.svg', layer, plugin='svg')` through our plugin
  architecture.

### And one more thing...

Thanks to the ever-creative Kira Evans, napari will now populate layer names
based on the name of the variables being visualized:

```python
import napari
from skimage import data

camera = data.camera()
with napari.gui_qt():
    viewer = napari.view_image(camera)
    print(viewer.layers[0].name)  # prints "camera"!
```

In Python 3.8, the name will even be visible if you are using the assignment
expression a.k.a. the walrus operator:

```python
import napari
from skimage import data

with napari.gui_qt():
    viewer = napari.view_image(camera := data.camera())
    print(viewer.layers[0].name)  # prints "camera"!
```

## New Features
- Hook up reader plugins (#937)
- Support for magicgui (#981)
- Writer plugins (#1104)


## Improvements
- Generalize keybindings (#791)
- Points view data refactor (#951)
- Add magic name guessing (#1008)
- Refactor labels layer mouse bindings (#1010)
- Reduce code duplication for `_on_scale_change` & `_on_translate_change` (#1015)
- Style refactor (#1017)
- Add ScaleTranslate transform (#1018)
- Add docstrings for all the Qt classees (#1022)
- Sorting and disabling of hook implementations (#1023)
- Plugin exception storage and developer notification GUI (#1024)
- Refactor points layer mouse bindings (#1033)
- Add chains of transforms (#1042)
- Shapes layer internal renaming (#1046)
- Internal utils / keybindings / layer accessories renaming (#1047)
- Shapes layer mouse bindings refactor (#1051)
- Make console loading lazy (#1055)
- Remove scikit-image dependency (#1061)
- All args to add_image() accept sequences when channel_axis != None (#1092)
- Added documentation for label painting (#1094)
- Finish layer mouse bindings refactor (#1121)
- Enable volume rendering interpolation control (#1127)
- Change image icon (#1128)
- Refactor Vectors colors (#1130)
- Reduce code duplication in Points (#1140)
- Add keybinding to toggle selected layer visibility (#1147)
- Only block in gui_qt or quit QApp when necessary (#1148)
- Plugin `_HookCaller` and registration updates (#1153)
- Reduce code duplication in points tests (#1154)
- Color cycles properties return array (#1163)
- Allow last added point to be deleted with backspace keybinding (#1164)
- Extract plugin code to napari-plugin-engine (#1169)
- Turn caching on and fusion off when adding a dask array (#1173)
- Save all/selected layers Qt dialogs (#1185)
- Magic layer name guessing is always on (#1186)
- Warn when nothing saved (#1188)
- Add builtin shapes writer and reader (#1193)
- Bump svg-dep, add builtin write_labels (#1200)
- Change screenshot hotkey and open menubar names (#1201)

## Bug Fixes
- Refactor cleanup, prevent leaked widgets, add viewer.close method (#1014)
- Allow StringEnum to accept instances of self as argument (#1027)
- Close the canvas used to retrieve maximum texture size. (#1028)
- Fix points color cycle refresh (#1034)
- Fix styles for Qt < 5.12.  Fix styles in manifest for pip install (#1037)
- Fix label styles in contrast limit popup  (#1039)
- Fix pyramid clipping (#1053)
- Fix resources/build_icons:build_resources_qrc (#1060)
- Add error raise in `Viewer._add_layer_from_data` (#1063)
- Fix empty points with properties (#1069)
- Fix image format (#1076)
- Lazy console fixes (#1081)
- Fix function and signature to match with code (#1083)
- Fix LayerList and ListModel type checking (#1088)
- Fix changing translate order when changing data dims (#1102)
- Fix add_points_with_properties example (#1126)
- Use mode='constant' in numpy.pad usage (#1150)
- Fix canvas none after layer deletion (#1158)
- Prevent crash when viewing 3D pyramids (#1179)
- Add ensure_colormap utility function to standardize colormap getting/setting (#1180)
- Fix small plugin error report bug (#1181)
- Fix multichannel implicit multiscale (#1192)
- One more plugin error reporting fix (#1194)
- Normalize paths handed to reader plugins (#1195)
- Fix singleton dims display after toggle (#1196)
- Fix resize axis labels on show (#1197)
- Only assert that dask config returns to original value in test (#1202)

## Breaking API Changes
- Allow add_path() to accept any layer-specific kwarg and rename to open() (#1111)
- Remove path arg from add_image and add_labels (#1149)
- Drop pyramid autogeneration (#1159)
- Replace pyramid with multiscale (#1170)
- Add napari-svg plugin support (#1171)


## Support
- Publish developer resources (#967)
- Fix recursive include in manifest.in (#1003)
- Fix pip install with older versions of pip (#1011)
- Plugin docs (for plugin developers) (#1030)
- Add github links to contributor list (#1043)
- Check if PyQt5 is installed and then not install PySide2 (#1059)
- Use pip instead of invoking setup.py (#1072)
- pyenv with PyQt5 in other environment workaround (#1080)
- Roadmap for 0.3 releases (#1095)
- Add installed plugins to sys_info() (#1096)
- Avoid pillow 7.1.0 (#1099)
- Pin pillow at <= 7.0.0 (#1108)
- Fix a number of sphinxdocs errors (#1113)
- Fix miniconda download in CI (#1119)
- Convert old release notes to md (#1135)
- Automate release process (#1138)
- Open up Pillow again after 7.1.* (#1146)
- Fix black consistency (#1152)
- Fix sphinx (#1155)
- Change release notes source file to use .md ext (#1156)
- Rename requirements test (#1160)
- Update Pillow dependency pin (#1172)
- Update napari-svg to 0.1.1 (#1182)
- Update manifest.in for plugin code removal (#1187)
- Fix pip-missing-reqs step (#1189)


## 13 authors added to this release (alphabetical)

- [Bhavya Chopra](https://github.com/napari/napari/commits?author=BhavyaC16) - @BhavyaC16
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Hector](https://github.com/napari/napari/commits?author=hectormz) - @hectormz
- [Hugo van Kemenade](https://github.com/napari/napari/commits?author=hugovk) - @hugovk
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Luigi Petrucco](https://github.com/napari/napari/commits?author=vigji) - @vigji
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Peter Boone](https://github.com/napari/napari/commits?author=boonepeter) - @boonepeter
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Tony Tung](https://github.com/napari/napari/commits?author=ttung) - @ttung


## 13 reviewers added to this release (alphabetical)

- [Ahmet Can Solak](https://github.com/napari/napari/commits?author=AhmetCanSolak) - @AhmetCanSolak
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Hector](https://github.com/napari/napari/commits?author=hectormz) - @hectormz
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justine Larsen](https://github.com/napari/napari/commits?author=justinelarsen) - @justinelarsen
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Luigi Petrucco](https://github.com/napari/napari/commits?author=vigji) - @vigji
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Shannon Axelrod](https://github.com/napari/napari/commits?author=shanaxel42) - @shanaxel42
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Tony Tung](https://github.com/napari/napari/commits?author=ttung) - @ttung
