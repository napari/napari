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
conduct](https://napari.org/docs/developers/CODE_OF_CONDUCT.html), a [mission
and values
document](https://napari.org/docs/developers/MISSION_AND_VALUES.html), and
adopted a [community governance
model](https://napari.org/docs/developers/GOVERNANCE.html) (based on
scikit-image's, and since adopted with modifications by zarr). These are
accessible from our [developer resources
page](https://napari.org/docs/developers.html), together with a [public
roadmap](https://napari.org/docs/developers/ROADMAP_0_3.html) explaining
where the core team will devote effort in the coming months.

We are still overwhelmed by the community response to napari and we hope that
the above documents will continue to encourage potential users to join our
community. We continue to welcome contributions of all kinds and encourage
you to get in touch with us if you don't see your most wanted feature in our
roadmap, or as an issue in our [issue
tracker](https://github.com/napari/napari/issues).

### Getting in touch

We have also created a [Zulip chat room](https://napari.zulipchat.com) and
worked with the [image.sc forum](https://image.sc) (use the "napari" tag) to
allow users to get help with any issues they might have using napari.

### IO plugins


### Dockable widgets and magicgui


### Multiscale image handling


### Points with properties

### API changes and improvements

- magic name guessing


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


## Breaking API Changes
- Allow add_path() to accept any layer-specific kwarg and rename to open_path() (#1111)
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
