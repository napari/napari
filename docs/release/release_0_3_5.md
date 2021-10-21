# napari 0.3.5

We're happy to announce the release of napari 0.3.5!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights
This release contains a number of bug fixes on various platforms. For those
interested in napari performance, we have added a new performance monitoring
mode, that can be activated by the `NAPARI_PERFMON` environment variable, see
(#1262) for details. We have also added a page in the explanations section of
our docs on napari's [rendering](https://napari.org/guides/stable/rendering.html)
including plans for the future.


## New Features
- Allow using of custom color dictionary in labels layer (#1339 and #1362)
- Allow Shapes face and edge colors to be mapped to properties (#1342)
- Add performance monitoring widget (#1262)


## Improvements
- Factor out ImageSlice and ImageView from Image (#1343)

## Bug Fixes
- Fix warning for python 3.8 (#1335)
- Fix range slider position (#1344)
- Fix Linux and Windows key hold detection (#1350)
- Fix crash when selecting all points (#1358)
- Fix deleting layers changing dims (#1359)
- Revert "remove scipy.stats import (#1250)" (#1371)


## Build Tools and Support
- Remove broken link from BENCHMARKS.md (#1236)
- New documentation on rendering (#1328)
- Remove incorrect dashes in cirrus push_docs task (#1330)
- Use correct pyqt version in tests (#1331)
- Fix docs version, reformat, and add explanations (#1368)


## 8 authors added to this release (alphabetical)

- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi


## 8 reviewers added to this release (alphabetical)

- [Davis Bennett](https://github.com/napari/napari/commits?author=d-v-b) - @d-v-b
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi
