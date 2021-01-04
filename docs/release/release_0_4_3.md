# napari 0.4.3

We're happy to announce the release of napari 0.4.3!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights
This release is the first release to have direct support for generating widgets from functions using [magicgui](https://napari.org/magicgui/) and the `viewer.window.add_functio_widget` method (#1856). We leverage the newly release `0.2` series of magicgui which seperates out an abstract function and widget API from its Qt backend. 

In this release we also seperate out more of the Qt functionality from napari making it easier to run headless (#2039, #2055). We also add a `napari.run` method to then launch the napari application (#2056).

We've also made good progress on our experimental support for an octree system for rendering large 2D multiscale images. You can try this functionality setting `NAPARI_ASYNC=1` as an environment variable.


## New Features
- Add support for function widgets (#1856)


## Improvements
- Use evented dataclass for dims (#1917)
- Replace layer dims with private named tuple (#1919)
- Add information about screen resolution (#1957)
- Add asdict and update to evented dataclass (#1966)
- async-30: ChunkLoader Stats (#1972)
- async-31: Fix Monitor Feature Toggle (#1975)
- octree: Fix float64 and check downscale ratio (#1976)
- async-32: Faster Octree Rendering (#1977)
- async-33: Cleanup QtPoll and Tiled Visuals (#1980)
- async-34: Octree Performance (#1989)
- async-35: OctreeDisplayOptions (#1991)
- async-36: New OctreeChunkLoader (#1992)
- async-37: Multilevel Rendering (#1995)
- async-38: Better Multilevel Rendering (#1997)
- async-39: Better Config and Remove Qt Code (#1999)
- Improve label paint lag (#2000)
- async-40: Shared Memory Resources (#2004)
- async-41: Better Octree Performance (#2010)
- Start magicgui apps too (#2016)
- async-42: New LoaderPool and remove ChunkKey (#2025)
- Two small path improvements (#2030)
- Cleanup dockwidget removal (#2036)
- Prepare for evented viewer (#2038)
- Add a MousemapProvider (#2040)
- Cleanup warnings in tests (#2041)
- async-43: New LoaderPoolGroup (#2049)
- Complete qt isolation (#2055)
- Create QApplication on demand in viewer.Window, add napari.run function (#2056)
- Remove global app logic from Window init (#2065)
- async-45: Docs and Cleanup (#2067)

## Bug Fixes
- Fix append and remove from layerlist (#1955)
- Change label tooltip checkbox (#1978)
- Fix cursor size (#1983)
- Fix layer removing by disconnecting each event individually (#1984)
- Compatibility with magicgui v0.2.0 (#1994)
- Skip rounding of numbers when comparing data slice (#2017)
- Fix InitVar problem (#2023)
- Fix channel_axis for affine transforms in add_image (#2026)
- Fix typos in theme event and vispy camera (#2034)
- Fix magicgui test funcs (#2044)
- Fix magicwidget.native detection of "empty" widgets (#2046)
- async-44: Fix Pixel Shift Bug (#2052)
- Fix missing console widget (#2063)


## API Changes
- Drop layer.shape (#1990)
- Drop image layer.shape (#2002)
- Move keymap handling off viewer (#2003)
- Drop scale, axes background color, connect directly to viewer (#2037)


## Deprecations
- Move interactive from viewer to camera (#2008)
- Move palette off viewer (#2031)


## Build Tools
- Rename artifact for nightly build releases (#1971)
- Update latest tag alone with nightly build (#2001)
- Only raise leaked widgets errors in tests if no other exception was raised (#2043)
- Bump minimum numpy requirement to 1.16.5  (#2050)
- Tox tests on github actions (#2051)
- Move all requirements to extras (#2054)
- Drop additional perfmon test pass (#2058)
- Fix name in gha test (#2059)
- Fix big sur on GHA and fix failed test template (#2061)
- Update bundle.py (#2064)
- Update pre-commit  and add pyupgrade (#2068)


## 8 authors added to this release (alphabetical)

- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Heath Patterson](https://github.com/napari/napari/commits?author=NHPatterson) - @NHPatterson
- [kir0ul](https://github.com/napari/napari/commits?author=kir0ul) - @kir0ul
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi


## 7 reviewers added to this release (alphabetical)

- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi

