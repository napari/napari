# Roadmap for 0.2.x series of releases

This document is meant to be a live snapshot of tasks and features that the `napari` team is investing resources in during our 0.2.x series of releases in Spring 2020. It should be used to guide napari core developers, encourage and inspire contributors, and provide insights to external developers who are interested in building for the napari ecosystem. It is not meant to limit what is being worked on within napari, and in accordance with our [values](MISSION_AND_VALUES.md#our-values) we remain **community-driven**, responding to feature requests and proposals on our [issue tracker](https://github.com/napari/napari/issues) and making decisions that are driven by our users’ requirements, not by the whims of the core team.

This roadmap is also designed to be in accordance with our stated [mission](MISSION_AND_VALUES.md#our-mission) to be the **multi-dimensional image viewer for Python** and to **provide graphical user interface (GUI) access to a plugin ecosystem of image analysis tools for scientists** to use in their daily work.

For more details on the high level goals and decision making processes within napari you are encouraged to read our [mission and values statement](MISSION_AND_VALUES.md) and look at our [governance model](GOVERNANCE.md). If you are interested in contributing to napari, we'd love to have your contributions, and you should read our [contributing guidelines](CONTRIBUTING.md) for information on how to get started.

Another good place to look for information around bigger-picture discussions within napari are our issues tagged with the [`long-term feature` label](https://github.com/napari/napari/labels/long-term%20feature).

## Priority areas and strategy

As of Spring 2020 napari is still in a relative early phase. We've seem some initial excitement from the Python image analysis community, but still have much work to do to achieve our mission of being a foundational multi-dimensional image viewer for Python and providing value to scientists doing image analysis across domains and levels of coding experience.

Our growth and adoption strategy is to provide immediate value around multi-dimensional image visualization and annotation to people already doing image analysis in Python. We are working to make it easy for these developers to then distribute their image analysis code in the form of [plugins for napari](MISSION_AND_VALUES.md#our-vision-for-plugins), thereby allowing scientists with little coding experience to re-use these plugins in their own research via the napari GUI.

To work towards this vision, we have set the following four high-level priorities to guide us over the upcoming months:

- Improving documentation, tutorials, and demos.

- Ensuring a robust foundational software architecture.

- Enhancing the viewer through development of fundamental features.

- Adding basic plugin management infrastructure.

We will prioritize the robustness and polish of the core viewer before adding support for functional or interactive plugins to ensure that plugin development can happen against a stable foundation.

### Improving documentation, tutorials, and demos

Education and documentation are [core values](MISSION_AND_VALUES.md#our-values) of the napari team and one of our [mission objectives](MISSION_AND_VALUES.md#our-mission) is to be **well-documented** with **comprehensive tutorials and examples**.

We've established a website [napari.org](https://napari.org) to provide easy access to all napari related materials, and are currently in the process of improving both the organization and content for the website, taking inspiration from [well-known documentation guides](https://www.divio.com/blog/documentation/). Some of goals are listed below:

- Provide **tutorials** that are *learning-oriented* and teach the fundamental aspects of napari, including the `viewer`, `layers`, and `dimensions` to newcomers. These tutorials must be kept up-to-date with any API changes.

- Provide **explanations** that are *understanding-oriented* and give background and context for design decisions, including choices relating to our software architecture, plugin management, and data models.

- Provide **how-to-guides** that are *goal-oriented* and show how to use napari to solve specific problems, prioritizing major scientific use-cases and challenges. These guides should contain step-by-step instructions, and serve to motivate newcomers to try using napari to solve their problems.

- Provide a comprehensive **API reference** that is *information-oriented* and describes napari in detail. Every function and attribute should have complete, accurate, and easy to understand docs strings that are easily accessible through this reference.

For more details on the website improvements see [#764](https://github.com/napari/napari/issues/764) and the [`documentation` label](https://github.com/napari/napari/labels/documentation) on the napari repository. The repository for the website, API reference, and tutorials are at [napari/napari.github.io](https://github.com/napari/napari.github.io), [napari/tutorials](https://github.com/napari/tutorials), and [napari/docs](napari/docs) respectively.

### Ensuring a robust foundational software architecture

One of the [values](MISSION_AND_VALUES.md#our-values) of napari is to be **a tool others can build on** and one of our [mission objectives](MISSION_AND_VALUES.md#our-mission) is to be **easy to use and install**. Having a robust foundational software architecture is essential to achieving those goals, and we're in the process of taking numerous steps to ensure that we're building for the long-term including:

- A model-view-controller inspired refactor to centralize event handling within napari and avoid circular or out of order calls, see [#1020](https://github.com/napari/napari/issues/764) for details.

- Lazy loading and better management of our console to improve resource handling, see [#787](https://github.com/napari/napari/issues/787) and associated issues for details.

- Improved unit testing, focusing on making our CI more robust, increasing coverage of our tests, and adding GUI tests. See the [`tests` label](https://github.com/napari/napari/labels/tests) for more information. 

- Eliminating all known bugs. See the [`bug` label](https://github.com/napari/napari/labels/bug) for a current list of known bugs. 

- Ensuring our installation success rate is close to 100%. We can still struggle with installation of Qt in complex cases, and have had problems with dependency conflicts with other Python packages. We have recently added support for a [conda-forge package](https://github.com/conda-forge/napari-feedstock) and are working towards distributing bundled applications in [#496](https://github.com/napari/napari/pull/496). See the [`installation` label](https://github.com/napari/napari/labels/installation) for a current list issues related to installation.

- Monitoring and improving performance using [benchmarks](BENCHMARKS.md) to ensure we remain highly performant as we develop new features. See the [`performance` label](https://github.com/napari/napari/labels/performance) for a current list issues related to performance.

### Enhancing the viewer through development of fundamental features

The heart of napari is its viewer, a highly performant, fully-featured graphical user interface, alone with fundamental layer types, including images, labels, points, shapes, surfaces, and vectors. A top development priority of this roadmap is to ensure that all the basic features of the viewer and layers that are needed for image analysis and visualization are in place and working well. A list of currently planned features can be found in [#420](https://github.com/napari/napari/issues/420) and is summarized into main areas here.

- Support for a world based coordinate system, allowing for physical coordinates, and chaining together of transforms - including affine and deformable transforms. This coordinate system should include support for multiscale representations of data such as pyramids, and a model for our camera system. See the [World Coordinates project board](https://github.com/napari/napari/projects/10) for more details.

- Layer groupings [#970](https://github.com/napari/napari/issues/970) and linked multicanvas support [#760](https://github.com/napari/napari/issues/760) that will allow groups of layers to be assigned to different canvases that can render different views of the data, for example simultaneous 2D orthoviews and 3D rendering, or linked 2D views of the same physical location in different images.

- Support for annotation labels including text rendering [#600](https://github.com/napari/napari/pull/600).

- Linked 1D plots such as histograms, timeseries, or z-profiles [#823](https://github.com/napari/napari/pull/823) and [#675](https://github.com/napari/napari/pull/675).

- Support for generating animations [#780](https://github.com/napari/napari/pull/780).

- Draggable and resizable layers [#299](https://github.com/napari/napari/issues/299) and [#989](https://github.com/napari/napari/pull/989).

- Improved support for big and remote data, see [#881](https://github.com/napari/napari/issues/881), including making changing views non-blocking [#845](https://github.com/napari/napari/issues/845), better caching [#718](https://github.com/napari/napari/issues/718), and more done asynchronously.

Not prioritized during this roadmap, but features that will likely be develop in future roadmaps are general support for undo / redo functionality [#474](https://github.com/napari/napari/issues/299), a history feature, and macro generation.

### Adding basic plugin management infrastructure

A core aspect of the vision for napari is a [thriving plugin ecosystem](MISSION_AND_VALUES.md#our-vision-for-plugins), and we are taking steps now to lay some of the foundations of that ecosystem. We have recently adopted [pluggy](https://pluggy.readthedocs.io/en/latest/) plugin management framework within napari for this purpose. See [#936](https://github.com/napari/napari/issues/936) and linked issues for details. Pluggy is already widely used within the Python community, in particular by the [pytest](https://docs.pytest.org/en/latest/index.html) testing framework, allowing us to benefit from years of development and community support.

As noted above, we are prioritizing the robustness and polish of the core viewer before adding support for functional or interactive plugins to ensure that plugin development can happen against a stable foundation.

However, we are now though adding some basic support for plugin management, including establishing how plugins will be  discovered and adding support for reader plugins [#937](https://github.com/napari/napari/pull/937) that will make it easier for people to get data in napari. Adding basic support for a very limited subset of plugins now allows us to ensure that any work we're doing to enhance the robustness of napari or add new features to the viewer will be compatible with our plans for a plugin ecosystem in the future.

Work that is not part of this roadmap, but that will be done as part of future roadmaps, includes establishing an interface for functional and interactive plugins, and increased tooling to run plugins from the viewer.

Our progress with all things plugin related can be followed with the [`plugins` label](https://github.com/napari/napari/labels/plugins) on our repository.
