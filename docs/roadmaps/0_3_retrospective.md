# Roadmap 0.3 Retrospective

Periodically, the napari team produces roadmaps to clarify priorities in the coming months of development. When a new roadmap is introduced, we will retrospectively examine how well actual development tracked with the appropriate roadmap.

Below, we examine our 0.3.x roadmap (reproduced in its entirety) and compare it to what we achieved in the corresponding 6 months (in inline quote blocks).

## For 0.3.* series of releases - April 2020-Nov 2020

The napari roadmap captures current development priorities within the project and should serve as a guide for napari core developers, to encourage and inspire contributors, and to provide insights to external developers who are interested in building for the napari ecosystem. For more details on what this document is and is not, see the [about this document section](#about-this-document).

The [mission](our-mission) of napari is to be a foundational multi-dimensional image viewer for Python and provide graphical user interface (GUI) access to image analysis tools from the Scientific Python ecosystem for scientists across domains and levels of coding experience. To work towards this mission, we have set the following three high-level priorities to guide us over the upcoming months:

- Make the **data viewing and annotation** capabilities **bug-free and fast**.

- Make a **downloadable application** with reader / writer plugin management.

- Make accessible documentation, tutorials, and demos.

Once the above goals are met, we will develop napari's capabilities for image processing and analysis. We are prioritizing the robustness and polish of the core viewer before adding advanced features or support for functional or interactive plugins to ensure that plugin development will happen against a solid foundation, and because we have repeatedly encountered biologists and other scientists who have trouble even looking at their data, and annotating it. We are prioritizing the downloadable application with reader / writer plugin management so that we can start getting feedback from non-coding users to better understand their needs.

> Note added Novemember 2020. We have now finished our 0.3 series of releases. Overall we made good progress towards these priorites. We now provide cross-platform support for downloadable app that you can install reader and writer plugins into. We added support for world coordinates, we reduced the number of bugs by ~20% (>80 to <60), and we've begun begun to support asynchronous rendering and improve performance. We also had some highlights that weren't on our roadmap including contributions from the community like the Tracks layer. We've added more documentation and tutorials, but a redesign of our website is still in progress. For a detailed lookback at the work that was done during the `0.3` series of releases you can look at comments under the individual bullet items below. For a look at where things are heading next, please see our [0.4 roadmap](https://napari.org/roadmaps/0_4.html).

## Make the data viewing and annotation capabilities bug-free, fast and easy to use

- **Better support for viewing big datasets**. Currently, napari is fast when viewing on-disk datasets that can be naturally sliced along one axis (e.g. a time series) *and where loading one slice is fast*. However, when the loading is slow, the napari UI itself becomes slow, sometimes to the point of being unusable. We aim to improve this by making views and interactions non-blocking ([#845](https://github.com/napari/napari/issues/845)), and improving caching ([#718](https://github.com/napari/napari/issues/718)). We will also ensure that napari can be used `headless` without the GUI.

    > We've made good progress towards `asynchronous` loading of data in our [async](https://github.com/napari/napari/labels/async) work, but we still have much more to do. The async functionality can be enabled right now in an opt in fashion but is still considered experimental as we develop it further. We're also working on an octree which will enable better multiscale rendering, see this [rendering explanation note](rendering-explanation), which has become available during our initial 0.4 series of releases.


- **Improving the performance of operations on in-memory data**. Even when data is loaded in memory, some operations, such as label and shape painting, slicing along large numbers of points, or adjusting contrast and gamma, can be slow. We will continue developing our [benchmark suite](https://github.com/napari/napari/blob/master/docs/developers/BENCHMARKS.md) and work to integrate it into our development process. See the [`performance` label](https://github.com/napari/napari/labels/performance) for a current list of issues related to performance.

    > We've improved some perfomance issues, such as adjusting gamma and contrast limits, which now happen on the shader, but still have more to go. Painting in labels remains slow for large datasets and still needs to be improved. We added an [explantory note on perfomance](napari-performance) to detail how we're thinking about perfomance in napari and ensuring it stays high. We've also added some [perfomance monitoring tooling](perfmon) that will help us during this process. We successful used this functionality to improve performance of the Shapes Layer and will continue this work during the 0.4 series of releases.


- Add a unified **world coordinate system**. Scientists need to measure data that comes from a real space with physical dimensions. Currently, napari has no concept of the space in which data lives: everything is unitless. Further, it is unclear at various parts in the UI whether a coordinate has been transformed. And finally, some data are acquired with distortions, such as skew in data collected on stage-scanning lightsheet microscopes, and napari should be able to account for those distortions by chaining together transforms - including affine and ultimately deformable transforms. We are tracking progress in this area in the [World Coordinates project board](https://github.com/napari/napari/projects/10).

    > We have completed our world coordinate system, including support for affine transforms, a scale bar, and an axes rotation visual. We have more work to do to add support for physical units which will come in the 0.4 series of releases, non-orthogonal slicing, and non-rigid transforms, which will be saved for future roadmaps.

- Ensure **easy installation with a success rate close to 100%**. We can still struggle with installation of Qt in some cases, and have had problems with dependency conflicts with other Python packages. We have recently added a [conda-forge package](https://github.com/conda-forge/napari-feedstock) and are working towards distributing bundled applications in [#496](https://github.com/napari/napari/pull/496). See the [`installation` label](https://github.com/napari/napari/labels/installation) for a current list of issues related to installation.

    > We have made good progress on improving the success rate of our installation process, and now encounter less installation related new issues. In particular, installation in fresh environments is often successful, especially using conda-forge, and our bundled app now exists and works for a large proportion of users. Many platform specific bugs still remain though. Windows is often a problem, partly due to lack of representation in the core team, and macOS Big Sur has raised a whole new set of issues, though again, this works with conda-forge and python 3.9.0 as of the 0.4 series of releases.

- **Eliminate all known bugs**. See the [`bug` label](https://github.com/napari/napari/labels/bug) for a current list of known bugs. We also want to make it easier for users to report bugs ([#1090](https://github.com/napari/napari/issues/1090)). Additionally, when bugs are encountered, we want to examine whether an improved software architecture could have prevented them. For example, we are undertaking a refactor to centralize event handling within napari and avoid circular, out of order, or repeated function calls ([#1040](https://github.com/napari/napari/issues/1040)). In the process of eliminating bugs we should also ensure that our continuous integration, i.e. automatic testing of each change on cloud infrastructure, is robust, and that we have increased coverage of our tests, including more GUI tests with screenshots. See the [`tests` label](https://github.com/napari/napari/labels/tests) for more information about our tests.

    > We've reduced the number of open [`bug`](https://github.com/napari/napari/labels/bug) labeled issues by more than 20% in the past two months from > 80 to < 60 and added some more screenshot and GUI tests, but test coverage of the GUI can still be improved.

- **Improve support for annotation** of points [#858](https://github.com/napari/napari/issues/858), shapes [#177](https://github.com/napari/napari/issues/177), labels, and images [#209](https://github.com/napari/napari/issues/209) including text rendering [#600](https://github.com/napari/napari/pull/600).

    > We have added programmatic support for text and other properties in points, shapes, and labels, allowing users to effectively visualize many properties in their data. However, support for editing these properties from the UI remains limited or non-existent, so we will aim to improve this in the next roadmap.

- Add **linked multi-canvas support** ([#760](https://github.com/napari/napari/issues/760)) to allow orthogonal views, or linked views of two datasets, for example to select keypoints for image alignment, or simultaneous 2D slices with 3D rendering.

    > We have not done this work yet, and this work will be moved to our 0.4 series of releases.

- Add **layer groups** [#970](https://github.com/napari/napari/issues/970), which allow operating on many layers simultaneously making the viewer easier to use for multispectral or multimodal data, or, in the context of multiple canvases, where one wants to assign different groups to different canvases.

    > We have not done this work yet, and this work will be moved to our 0.4 series of releases.

- Improve the **user interface and design** of the viewer to make it easier to use. A [napari design audit](https://github.com/napari/napari/issues/469) last year dramatically improved the usability of the viewer. We must continue to run through these periodically to ensure the UI is friendly to new users, particularly non-programmers. See the [`design` label](https://github.com/napari/napari/labels/design) for more information.

    > We have conducted a product heuristics analysis to [identify design and usability issues](https://github.com/napari/product-heuristics-2020), and will now be working during the 0.4 series of releases to implement them.

## Make a downloadable application with basic plugin management and persistent settings

- Distribute **a bundled application for each major OS** [#496](https://github.com/napari/napari/pull/496) to allow scientists to use napari without requiring a Python development environment, which can be difficult to install and maintain. The bundle should contain the necessary machinery to add plugins [#1074](https://github.com/napari/napari/issues/1074).

    > We have succeeded in providing a bundled app that you can install plugins into across all platforms. We have more work to do around plugin management and app updates that will continue in our 0.4 series of releases, but the bundled app is now usable to install napari by people who aren't familiar with the command line.

- Support for reader plugins [#937](https://github.com/napari/napari/pull/937) and writer plugins [#1068](https://github.com/napari/napari/issues/1068) to allow **viewing of domain-specific data and saving of annotations**. For more details see the [`plugins` label](https://github.com/napari/napari/labels/plugins) on our repository.

    > We have a stable specification for adding both [reader and writer plugins](hook-specifications-reference) that is being used in the community.

- Support for persistent settings [#834](https://github.com/napari/napari/pull/834) to allow **saving of preferences** between launches of the viewer.
    
    > We have not done this work yet, and this work will be moved to our 0.4 series of releases.


## Provide accessible documentation, tutorials, and demos

- Improve our website [napari.org](https://napari.org) to provide easy access to all napari related materials, including the [**four types of documentation**](https://www.divio.com/blog/documentation/): learning-oriented tutorials, goal-oriented how-to-guides or galleries, understanding-oriented explanations (including developer resources), and a comprehensive API reference. See [#764](https://github.com/napari/napari/issues/764) and the [`documentation` label](https://github.com/napari/napari/labels/documentation) on the napari repository for more details.

    > We have begun the design work for this but have not begun the implementation yet. This work will be moved to our 0.4 series of releases.

- Add a **napari human interface guide** for plugin developers, akin to [Apple's Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/). We want such a guide to promote best practices and ensure that plugins provide a consistent user experience.

    > We have not done this work yet, and this work will be moved to our 0.4 series of releases.

## Work prioritized for future roadmaps

We’re also planning or working on the following more advanced features, which will likely be prioritized in future roadmaps:

- General support for undo / redo functionality [#474](https://github.com/napari/napari/issues/299), a history feature, and macro generation.

- Complete serialization of the viewer [#851](https://github.com/napari/napari/pull/851) to enable sharing the entire viewer state. This feature will be supported after `writer plugins` have been added.

- Support for generating animations [#780](https://github.com/napari/napari/pull/780). This feature will be supported after we have the ability to serialize the viewer state [#851](https://github.com/napari/napari/pull/851).

    > We have moved development of the animation functionality to a standalone plugin. Currently under active development at https://github.com/sofroniewn/napari-animation.

- Draggable and resizable layers [#299](https://github.com/napari/napari/issues/299) and [#989](https://github.com/napari/napari/pull/989). This feature will be supported after we have added support for world coordinates [project board 10](https://github.com/napari/napari/projects/10) and rotations to our transform model.

- Linked 1D plots such as histograms, timeseries, or z-profiles [#823](https://github.com/napari/napari/pull/823) and [#675](https://github.com/napari/napari/pull/675).

- Support for using napari with remote computation (i.e. a remote jupyter notebook [#495](https://github.com/napari/napari/issues/495)).

- Functional or interactive plugins that allow for analysis of data or add elements to the GUI.


## About this document

This document is meant to be a snapshot of tasks and features that the `napari` team is investing resources in during our 0.3 series of releases starting April 2020. This document should be used to guide napari core developers, encourage and inspire contributors, and provide insights to external developers who are interested in building for the napari ecosystem. It is not meant to limit what is being worked on within napari, and in accordance with our [values](our-values) we remain **community-driven**, responding to feature requests and proposals on our [issue tracker](https://github.com/napari/napari/issues) and making decisions that are driven by our users’ requirements, not by the whims of the core team.

This roadmap is also designed to be in accordance with our stated [mission](our-mission) to be the **multi-dimensional image viewer for Python** and to **provide graphical user interface (GUI) access to a plugin ecosystem of image analysis tools for scientists** to use in their daily work.

For more details on the high level goals and decision making processes within napari you are encouraged to read our [mission and values statement](mission-and-values) and look at our [governance model](napari-governance). If you are interested in contributing to napari, we'd love to have your contributions, and you should read our [contributing guidelines](napari-contributing) for information on how to get started.

Another good place to look for information around bigger-picture discussions within napari are our issues tagged with the [`long-term feature` label](https://github.com/napari/napari/labels/long-term%20feature).
