# napari 0.4.3

We're happy to announce the release of napari 0.4.3!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights
In this release we've added two new analysis and GUI focused [hook specifications](https://napari.org/docs/dev/plugins/hook_specifications.html) for our plugin developers (#2080).

The first one `napari_experimental_provide_function_widget` allows you to provide a function or list of functions that we
will turn into a GUI element using using [magicgui](https://napari.org/magicgui/). This hook spec leverages the newly added and the `viewer.window.add_function_widget` method (#1856) and the newly recently released `0.2` series of magicgui which seperates out an abstract function and widget API from its Qt backend. These functions can take in and return napari layer, allowing you to
provide analysis functionality to napari without having to write GUI code.

The second one `napari_experimental_provide_dock_widget` allows you to provide a QWidget or list of QWidgets that we will instantiate with access to the napari viewer and add to the GUI. This hook spec leverages our `viewer.window.add_dock_widget` method, and allows you to provide highly customized GUI elements that could include additional plots or interactivity.

Both of these hook specs are marked as `experimental` as we're likely to evolve the API here in response to user needs, and we're excited to get early feedback from plugin developers on them.

In this release we also seperate out more of the Qt functionality from napari making it easier to run headless (#2039, #2055). We also added a `napari.run` method as an alternative to using the `napari.gui_qt` context manager (#2056).

We've also made good progress on our `experimental` support for an octree system for rendering large 2D multiscale images. You can try this functionality setting `NAPARI_OCTREE=1` as an environment variable. See our [asynchronous rendering guide](https://napari.org/guides/stable/rendering.html) for more details on how to use the octree and its current limitations.

Finally we've added our [0.4 series roadmap](https://napari.org/roadmaps/0_4.html) and a [retrospective on our 0.3 roadmap](https://napari.org/roadmaps/0_3_retrospective.html)!


## New Features
- Add support for function widgets (#1856)
- Gui hookspecs (#2080)


## Improvements
- Use evented dataclass for dims (#1917)
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
- Add a MousemapProvider (#2040)
- Cleanup warnings in tests (#2041)
- async-43: New LoaderPoolGroup (#2049)
- Complete qt isolation (#2055)
- Create QApplication on demand in viewer.Window, add napari.run function (#2056)
- Remove global app logic from Window init (#2065)
- async-45: Docs and Cleanup (#2067)
- Better bound magicgui viewer (#2100)
- reduce call of _extent_data in layer (#2106)


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
- Save app reference in Window init (#2076)
- Add deprecated parameters for updating theme (#2074)
- Coerce name before layer is added to layerlist (#2087)
- Fix stale data in magicgui `*Data` parameters (#2088)
- Make dock widgets non-tabbed (#2096)
- Fix overly strict magic kwargs (#2099)
- Undo calling pyrcc with python sys.executable (#2102)


## API Changes
- The ``axis`` parameter is no longer present on the ``current_step``, ``range``, or ``axis_labels`` events. Instead a single event is emitted whenever the tuple changes (#1917)
- The deprecated public layer dims has been removed in 0.4.2 and the private ``layer._dims`` is now a NamedTuple (#1919)
- The deprecated ``layer.shape`` arrtibute has been removed. Instead you should use the ``layer.extent.data`` and ``layer.extent.world attributes`` to get the extent of the data in data or world coordinates (#1990, #2002)
- Keymap handling has been moved off the ``Viewer`` and ``Viewer.keymap_providers`` has been removed. The ``Viewer`` itself
can still provide keymappings, but no longer handles keymappings from other objects like the layers. (#2003)
- Drop scale background color and axes background color. These colors are now determined by defaults or the canvas background color. (#2037)
- ``event.text`` was renamed ``event.value`` for the events emitted when changing ``Viewer.status``, ``Viewer.title``,
``Viewer.help``, and ``event.item`` was renamed ``event.value`` for the event emitted when changing ``Viewer.active_layer`` (#2038)


## Deprecations
- The ``Viewer.interactive`` parameter has been deprecated, instead you should use ``Viewer.camera.interactive`` (#2008)
- The ``Viewer.palette`` attribute has been deprecated. To access the palette you can get it using ``napari.utils.theme.register_theme`` dictionary using the ``viewer.theme`` as the key (#2031)
- Annotating a magicgui function with a return type of ``napari.layers.Layer`` is deprecated. To indicate that your function returns a layer data tuple, please use a return annotation of ``napari.types.LayerDataTuple`` or ``List[napari.types.LayerDataTuple]``(#2079)


## Build Tools and Support
- 0.4 roadmap (#1906)
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
- Skip perfmon test on python 3.9 on CI (#2073)
- async-46: Rendering Guide and Code Comments (#2078)
- Update get-tag action (#2083)
- Update magicgui examples (#2084)
- Add examples tests (#2085)
- Skip testing examples on CI (#2094)
- Fix roadmap headings in docs (#2097)
- Add PR 2106 to 0.4.3 release notes (#2107)
- Fix `pytest --pyargs napari` test on pip install. Add CI test (#2109)


## 9 authors added to this release (alphabetical)

- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Heath Patterson](https://github.com/napari/napari/commits?author=NHPatterson) - @NHPatterson
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [kir0ul](https://github.com/napari/napari/commits?author=kir0ul) - @kir0ul
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
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

