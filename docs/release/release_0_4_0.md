# napari 0.4.0

We're happy to announce the release of napari 0.4.0! This might be our biggest
release yet — see below for highlights. Note also that, following the [NEP-29
deprecation policy](https://numpy.org/neps/nep-0029-deprecation_policy.html),
we have dropped support for Python 3.6: napari 0.4 requires Python 3.7 or
higher to run.

napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).

For more information, examples, and documentation, please visit our website at
https://napari.org

## Highlights
napari 0.4.0 is the culmination of months of improvements to our data models.
It finally brings the data from all layers into a consistent, global coordinate
system. This means our display is more accurate (we aim for pixel-perfect
precision), and it will be easier to build applications for accurate
measurement on top of napari.

Thanks to the global coordinate system we are now able to display a scale bar,
axis directions, and provide a cursor model that reports the current cursor
coordinates within the current world view. (Currently, physical units are not
supported, but they are coming soon!) We can also natively display data
transformed with an arbitrary affine transform, making it easy to view things
like light sheet data straight out of the microscope!

We have also added a new layer type, our first in over a year, to display
tracking data. Many thanks to Alan Lowe from UCL/The Turing Institute for this
contribution!

Finally, our experimental asynchronous rendering mode continues to be improved.
Use the `NAPARI_ASYNC=1` environment variable to try it, and please report
issues at https://github.com/napari/napari/issues.

We thank the many contributors who have made this release possible!

## New Features
- Add camera model (#854)
- Tracks layer (#1361)
- Affine transforms (#1616)
- Add axes visual (#1719)
- Add scale bar (#1720)
- Add a welcome visual (#1721)
- Add BOP (blue/orange/purple) colormaps (#1743)
- Add cursor model (#1763)


## Improvements
- Dataclass decorator with events and properties (#1475)
- Docker Integration by adding dockerfile and docker.md (#1496)
- Don't show colormap for rgb images (#1586)
- Fix NameError: name 'Optional' is not defined on Python 3.9 (#1650)
- Async-2.6: Move async into experimental  (#1659)
- Skip axes with size 1 in roll operation (#1665)
- Reduce the default dask cache size (#1666)
- Async-2.7: Vendor Cachetools (#1671)
- Pull out layer visual transform code (#1673)
- Async-3: Add ChunkCache (#1675)
- Async-4: Config File (#1676)
- Async-5: Timing Infrastructure (#1678)
- Async-6: DelayQueue (#1679)
- Async-7: Auto-async (#1683)
- Async-8: config cleanup (#1684)
- Async-9: async unit testing infra (#1685)
- Allow dask development version to work with napari (#1692)
- Autogenerate add_* methods (#1694)
- Async-10: add experimental vendor package humanize (#1698)
- Async-11: IPython commands (#1699)
- Async-12: Processes (#1704)
- Allow Tracks layer to accept data as a list or pandas DataFrame (#1705)
- Async-14: Prototyping Octree Visuals (#1707)
- Async-13: Fix "render to black" problem (#1715)
- Compute all min/max values in one pass with Dask (#1731)
- Async-15: Octree Rendering (#1751)
- Async-16: New utils.config.py (#1769)
- Add warning on failed save (#1770)

## Bug Fixes
- Status bar value bug for automatically down-sampled images (#1577)
- Magic name handle attribute exception (#1635)
- Speed up points selection and selection display (#1648)
- Fix pan zoom scaling (#1657)
- Fix world origin (#1660)
- Remove stray print (#1668)
- Fix is_running, fix tests (#1680)
- Fix 404 links in documentation and remove warnings in docs build (#1682)
- Fix image extent (#1696)
- Fix for tracks layer properties as dataframe (#1711)
- Fixes magic_name for all layers, fixes #1709 due to #1694 autogen (#1714)
- Add tests for #1709 for magic_name with add_* layers (#1716)
- Fix delete all shapes (#1718)
- Fix nD mutliscale (#1723)
- Fix initialization of tracks color_by (#1725)
- Fix screen change glitch on pixel scale (#1729)
- Handle non-existant properties keys for color_by in Tracks layer (#1736)
- Fix scale bar length in 3D (#1738)
- Fix world coordinates origin in 3D (#1740)
- Fix for #1745, introduces `on_matrix_change` in VispyTrackLayer (#1746)
- Fix blending and opacity in tracks shader (#1749)
- Fix grid size calculation (#1752)
- Add affine doc strings (#1753)
- Fix and test scale and translate broadcast (#1754)
- Fix shear setter (#1755)
- Fix flipped transform (#1757)
- Fix warning that filenames might not be set (#1765)
- Fix scaled painting (#1772)
- Fix setup config duplication of dask requirement (#1778)
- Adjust welcome colors of text (#1780)
- Fix gray to grays for vispy (#1784)
- Fix welcome text contrast (#1788)
- Fix zoom of scale bar and axes visuals on launch (#1791)

## API Changes and Deprecations
- Make layer dims private (#1581)
- Drop Python 3.6 (#1652)
- Make signals public attribute of WorkerBase (#1681)
- Add extent named tuple (#1771)

## Build Tools and Docs
- New perfmon doc (#1634)
- Build docs action (#1638)
- Increase retry action timeout from 10 to 30 mins (#1651)
- Add CI tests agains pre-release dependency versions (#1708)
- Pin dask !=2.28.0 due to performance problems, see napari issue #1656 (#1712)
- Fix top-level links referenced in README.md and published on napari.org (#1750)
- Update installation readme (#1758)


## 20 authors added to this release (alphabetical)

- [Abhishek Patil](https://github.com/napari/napari/commits?author=zeroth) - @zeroth
- [Aditya Mandke](https://github.com/napari/napari/commits?author=ekdnam) - @ekdnam
- [Alan R Lowe](https://github.com/napari/napari/commits?author=quantumjot) - @quantumjot
- [Christoph Gohlke](https://github.com/napari/napari/commits?author=cgohlke) - @cgohlke
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Jean-Christophe Fillion-Robin](https://github.com/napari/napari/commits?author=jcfr) - @jcfr
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Mark Kittisopikul](https://github.com/napari/napari/commits?author=mkitti) - @mkitti
- [Matthew Rocklin](https://github.com/napari/napari/commits?author=mrocklin) - @mrocklin
- [Matthias Wagner](https://github.com/napari/napari/commits?author=matthias-us) - @matthias-us
- [Max Hess](https://github.com/napari/napari/commits?author=MaksHess) - @MaksHess
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Nicolas CARPi](https://github.com/napari/napari/commits?author=NicolasCARPi) - @NicolasCARPi
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Volker Hilsenstein](https://github.com/napari/napari/commits?author=VolkerH) - @VolkerH
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi


## 18 reviewers added to this release (alphabetical)

- [Alan R Lowe](https://github.com/napari/napari/commits?author=quantumjot) - @quantumjot
- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Ben Cooper](https://github.com/napari/napari/commits?author=bkcooper) - @bkcooper
- [David Hoese](https://github.com/napari/napari/commits?author=djhoese) - @djhoese
- [Davis Bennett](https://github.com/napari/napari/commits?author=d-v-b) - @d-v-b
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Justine Larsen](https://github.com/napari/napari/commits?author=justinelarsen) - @justinelarsen
- [Kevin Yamauchi](https://github.com/napari/napari/commits?author=kevinyamauchi) - @kevinyamauchi
- [Kira Evans](https://github.com/napari/napari/commits?author=kne42) - @kne42
- [Lia Prins](https://github.com/napari/napari/commits?author=liaprins-czi) - @liaprins-czi
- [Lucy Obus](https://github.com/napari/napari/commits?author=LCObus) - @LCObus
- [Mark Kittisopikul](https://github.com/napari/napari/commits?author=mkitti) - @mkitti
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Volker Hilsenstein](https://github.com/napari/napari/commits?author=VolkerH) - @VolkerH
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi

