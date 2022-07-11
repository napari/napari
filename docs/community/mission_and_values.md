(mission-and-values)=
# Mission and Values

This document is meant to help guide decisions about the future of `napari`, be it in terms of
whether to accept new functionality, changes to the styling of the code or graphical user interface (GUI),
or whether to take on new dependencies, among other things. It serves as a point of reference for [core developers](core-dev-guide) actively working on the project, and an introduction for
newcomers who want to learn a little more about where the project is going and what the team's
values are. You can also learn more about how the project is managed by looking at our [governance model](napari-governance).

## Our founding principles

The founding napari team came together around a shared vision for a Python based image viewer that is

- n-dimensional first

- native (no browser required)

- blazing fast (GPU-powered)

- native 2D and 3D viewing

- with minimal new windows and pop-ups, choosing layering instead

As the project has grown we've turned these original principles into the mission statement and set of values that we described below.

(our-mission)=
## Our mission

napari aims to be the **multi-dimensional image viewer for Python** and to **provide GUI access to a plugin ecosystem of image analysis tools for scientists** to use in their daily work. We hope to accomplish this by:

- being **easy to use and install**. We are careful in taking on new dependencies, sometimes making them optional, and will support a fully packaged installation that works cross-platform.

- being **well-documented** with **comprehensive tutorials and examples**. All functions in our API have thorough docstrings clarifying expected inputs and outputs, and we maintain [tutorials and examples on our website](http://napari.org) to explain different use cases and working modes.

- providing **GUI access** to all critical functionality so napari can be used by people with no coding experience.

- being **interactive** and **highly performant** in order to support very large data sets.

- providing a **consistent and stable API** to enable plugin developers to build on top of napari without their code constantly breaking and to enable advanced users to build out sophisticated Python workflows. Right now we are still making breaking changes with minor version numbers `0.x` and do not have a deprecation policy, but we will work to add one soon.  

- **ensuring correctness**. We strive for complete test coverage of both the code and GUI, with all code reviewed by a core developer before being included in the repository.

(our-values)=
## Our values

- We are **inclusive**. We welcome and mentor newcomers who are making their first contribution and strive to grow our most dedicated contributors into [core developers](core-dev-guide). We have a [Code of Conduct](napari-coc) to make napari
a welcoming place for all.

- We are **community-driven**. We respond to feature requests and proposals on our [issue tracker](https://github.com/napari/napari/issues), making decisions that are driven by our users’ requirements, not by the whims of the core team.

- We provide **a tool others can build on** and making it easy for people to develop and share plugins that extend napari's functionality and to develop fully custom applications that consume napari.

- We serve **scientific applications** primarily, over “consumer” image editing in the vein of Photoshop or GIMP. This often means prioritizing n-dimensional data support, and rejecting implementations of “flashy” features that have little scientific value.

- We are **domain agnostic** within the sciences. Functionality that is highly specific to particular scientific domains belongs in plugins, whereas functionality that cuts across many domains and is likely to be widely used belongs inside napari.

- We **focus on visualization**, leaving image analysis functionality and highly custom file IO to plugins.

- We value **simple, readable implementations**. Readable code that is easy to understand, for newcomers and maintainers alike, makes it easier to contribute new code as well as prevent bugs.

- We value **education and documentation**. All functions should have docstrings, preferably with examples, and major functionality should be explained in our [tutorials](https://napari.org/tutorials). Core developers can take an active role in finishing documentation examples.

- We **minimize [magic](https://en.wikipedia.org/wiki/Magic_(programming))** and always provide a way for users to opt out of magical behaviour and guessing by providing explicit ways to control functionality. We support NumPy array like objects and we prefer to educate users rather than make decisions on their behalf. This does not preclude the use of sensible defaults.

## Our vision for plugins

As noted above, napari aims to support a plugin ecosystem for scientific image analysis. We have recently taken some of the first steps towards adopting the [pluggy](https://pluggy.readthedocs.io/en/latest/) plugin management framework within napari for this purpose. See [#936](https://github.com/napari/napari/issues/936) and linked issues for details. Pluggy is already widely used within the Python community, in particular by the [pytest](https://docs.pytest.org/en/latest/index.html) testing framework, allowing us to benefit from years of development and community support.

We will be adding more dedicated documentation around plugins shortly, and this section will likely continue to be significantly updated over the coming months as more work is done, but we will lay out here some of our motivation and vision for the plugin ecosystem.

Image analysis is heterogeneous and often highly specialized within different domains of science. napari alone will not try to meet all the image analysis needs of the scientific community, but instead try to be a foundational visualization tool that provides access to domain specific analysis through community developed plugins.

We want to make it as easy as possible for developers to build plugins for napari, with as little to no specific napari code in your plugin as possible. Philosophically a napari plugin without napari should just be Python, and they should always be importable and runnable as such. We are looking into using minimal typing and function annotations that don't change the runtime of your functions, but lets napari know how to interpret the inputs and outputs of your functions and integrate them with our GUI code. We are also planning to leverage git for versioning and installation of plugins.

As noted above, we'll be beginning to actively work on the plugin infrastructure over coming months and would love to get the input of the community. You can follow and contribute to our discussions and progress using the [`plugin` label on our repository](https://github.com/napari/napari/labels/plugins).

## Acknowledgements

We share a lot of our mission and values with the `scikit-image` project with whom we share founding members, and acknowledge the influence of their [mission and values statement](https://scikit-image.org/docs/dev/values.html) on this document.
