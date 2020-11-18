# napari 0.4.1

We're happy to announce the release of napari 0.4.1!
napari is a fast, interactive, multi-dimensional image viewer for Python.
It's designed for browsing, annotating, and analyzing large multi-dimensional
images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based
rendering), and the scientific Python stack (numpy, scipy).


For more information, examples, and documentation, please visit our website:
https://github.com/napari/napari

## Highlights
The napari 0.4.1 release is mainly a bug fix and improvements release following on from
recent 0.4.0 release. We've added a couple of nice new rendering modes for volumetric
data like minimum and average intesity projections, text labels to our axes visual,
and full colorbars in our colormap selection dropdown.


## New Features
- Live tiff loader example (#1610)
- Add NAPARI_EXIT_ON_ERROR (#1812)
- Add text labels to axes (#1860)
- Add minimum intensity projection shader to 3d rendering mode (#1861)
- Average intensity projection shader for 3d rendering (#1871)


## Improvements
- Integrated plugin dialog with install/uninstall & remote discovery (#1357)
- Use TypedEventedList and TypedList for LayerList and TransformChain (#1504)
- Add checkbox to labels layer controls for displaying selected color mode (#1762)
- Async-17: MiniMap (#1774)
- Async-18: Basic Quadtree Rendering (#1793)
- Mouse callback examples tweaks (#1796)
- Simplify viewer and make headless mode easier (#1808)
- Move viewer keybindings (#1810)
- Async-19: Minimap and Test Images (#1813)
- Pre-sort tracks data by ID and time (#1814)
- Enhance affine example to include scipy.ndimage (#1815)
- Sort tracks and track properties by ID then t (#1818)
- Add a simple grid object (#1821)
- Add example: visualize 2d+timelapse as 3D space-time images, a.k.a. "kymographs" (#1831)
- Async-20: Better QtTestImage (#1834)
- Async-21: TiledImageVisual with Texture Atlas (#1837)
- Return estimate for 3D texture size instead of hard-coded value (#1857)
- Drop old typed list (#1859)
- Colormap presention in dropdown list (#1862)
- Async-22: Support Edge/Corner Tiles (#1867)
- Async-23: QtRender Fixes (#1870)
- Async-24: Multiscale Octree w/ Async Loading (Golden Spike) (#1876)
- Add reverse gray colormap (#1879)
- Async-25: Remove VispyCompoundImageLayer (#1890)
- Async-26: Artificial Delay and fixes (#1891)


## Bug Fixes
- Fix zoom of scale bar and axes visuals on launch (#1791)
- Fix ValueError in Layer.coordinates during layer load (#1798)
- Fix TransformChain.__init__() (#1809)
- Fix #1811: Move visual creation to end of __init__ (#1835)
- Fix keybinding inheritance (#1838)
- Fix lower triangular shear values (#1839)
- Fix bug for painting when scale above 1  (#1840)
- Enhance accessibility by ensuring welcome screen contrast meets required standards (#1863)
- Apply requested opacity in points layer (#1864)
- Fix active layer update (#1882)
- Fix affine composition order (#1884)
- fix affine warning (#1886)


## API Changes
- Changes to grid mode API (#1821)
- Change grid size to shape (#1847)


## Build Tools and Docs
- Update setup.cfg to add aliases for pyqt5 and pyside in extras_require (#1795)
- Unpin qt, exclude 5.15.0 (#1804)
- Allow ability to run all tests on local Windows installs (#1807)
- Try different test for popup skipif mark (#1817)
- Fix image.sc badge link to forum (#1828)
- Fix bundle build by adding sudo for disk detach (#1844)
- Patch mac bundle differently (#1848)
- Bump qtconsole version requirement to fix #1854 (#1855)


## 16 authors added to this release (alphabetical)

- [Abhishek Patil](https://github.com/napari/napari/commits?author=zeroth) - @zeroth
- [Abigail McGovern](https://github.com/napari/napari/commits?author=AbigailMcGovern) - @AbigailMcGovern
- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Daniyah Aldawsari](https://github.com/napari/napari/commits?author=ddawsari) - @ddawsari
- [Draga Doncila Pop](https://github.com/napari/napari/commits?author=DragaDoncila) - @DragaDoncila
- [Frauke Albrecht](https://github.com/napari/napari/commits?author=froukje) - @froukje
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [kir0ul](https://github.com/napari/napari/commits?author=kir0ul) - @kir0ul
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Umberto Perco](https://github.com/napari/napari/commits?author=UmbWill) - @UmbWill
- [Volker Hilsenstein](https://github.com/napari/napari/commits?author=VolkerH) - @VolkerH
- [Will Moore](https://github.com/napari/napari/commits?author=will-moore) - @will-moore
- [Ziyang Liu](https://github.com/napari/napari/commits?author=ziyangczi) - @ziyangczi


## 10 reviewers added to this release (alphabetical)

- [Abhishek Patil](https://github.com/napari/napari/commits?author=zeroth) - @zeroth
- [Abigail McGovern](https://github.com/napari/napari/commits?author=AbigailMcGovern) - @AbigailMcGovern
- [Alister Burt](https://github.com/napari/napari/commits?author=alisterburt) - @alisterburt
- [Genevieve Buckley](https://github.com/napari/napari/commits?author=GenevieveBuckley) - @GenevieveBuckley
- [Grzegorz Bokota](https://github.com/napari/napari/commits?author=Czaki) - @Czaki
- [Juan Nunez-Iglesias](https://github.com/napari/napari/commits?author=jni) - @jni
- [Nicholas Sofroniew](https://github.com/napari/napari/commits?author=sofroniewn) - @sofroniewn
- [Philip Winston](https://github.com/napari/napari/commits?author=pwinston) - @pwinston
- [Talley Lambert](https://github.com/napari/napari/commits?author=tlambert03) - @tlambert03
- [Volker Hilsenstein](https://github.com/napari/napari/commits?author=VolkerH) - @VolkerH

