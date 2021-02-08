# napari 0.3.8

We're happy to announce the release of napari 0.3.8!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website:
https://napari.org


## Highlights
This release is mainly a bug fix release, with a number of small improvements
including around our contrast limits updates (#1622) and points coloring (#1641)
and (#1643). This will also be our last release supporting Python3.6.


## Improvements
- Async-2.5: Vispy Changes (#1607)
- Increase screenshot performance (#1615)
- Speed up points selection and selection display (#1648)

## Bug Fixes
- Remove silence overwritte during screenshot (#1567)
- Fix usage nan_to_num to work with numpy 1.16 (#1613)
- Update colortransform when building texture (#1622)
- Fix points layer coloring (#1623)
- Fix painting by creating a new slice every time (#1641)
- Fix new color point off (#1643)
- Prevent bundle fail when name is not napari (#1647)


## Build Tools and Docs
- Add test matrix entry to test minimum requirements (#1617)
- Change bundle deps approach (#1619)
- Use pyside2-rcc if pyrcc5 fail (#1626)
- Big update of rendering explanation doc (#1632)
- Do not ship bundle.py in source distribution (#1633)
- Remove Python version requirement for pre-commit (#1645)
- Add github retry action to try and fix flaky app bundling (#1649)


## 6 authors added to this release (alphabetical)

- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Matthias Wagner](https://github.com/napari/napari/commits?author=matthias-us) - @matthias-us
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi


## 5 reviewers added to this release (alphabetical)

- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
