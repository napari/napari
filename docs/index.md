---
theme:
  metaDescription: napari is a fast, interactive, multi-dimensional image viewer for Python. It's designed for browsing, annotating, and analyzing large multi-dimensional images. It's built on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the scientific Python stack (numpy, scipy).
  quickLinks:
    - title: Community
      content: Meet the team, our mission, and our values
      url: /community/index.html

    - title: Tutorials
      content: Step by step guides for common napari workflows
      url: /tutorials/index.html

    - title: Plugins
      content: Learn how to create a plugin that works with the napari ecosystem
      url: /plugins/index.html

    - title: Release notes
      content: See what’s been updated in the latest releases
      url: /release/index.html

    - title: API reference
      content: Information on specific functions, classes, and methods
      url: /api/index.html

    - title: Roadmaps
      content: Find out what we plan to build next and into the near future
      url: /roadmaps/index.html

    - title: Developer guides
      content: Explanations about how napari works behind the screen
      url: /guides/index.html

    - title: Developer resources
      content: All you need to know to contribute to the napari codebase
      url: /developers/index.html

    - title: Source code
      content: Jump out to GitHub to take a look at the code that runs napari
      url: https://github.com/napari/napari

    - title: napari hub
      content: Discover, install, and share napari plugins
      url: https://www.napari-hub.org
---

# napari

Napari is a fast, multi-dimensional image viewer for the Python programming
language. It also aims to provide GUI-based access to the scientific Python
image analysis ecosystem.

```python
import napari
viewer = napari.view_image(my_array)
```

<video width="90%" autoplay loop muted playsinline>
  <source src="images/alisterburt-viewer-cryoet.webm" type="video/webm" />
  <img src="images/alisterburt-viewer-cryoet.png"
      title="Your browser does not support the video tag." />
</video>

Napari can help you:

- quickly *view* 2D, 3D, and higher-dimensional image data
- *overlay* segmentations (labels), points, vectors, polygons, surface meshes,
  vectors, and tracking data
- *annotate* data, seamlessly weaving human curation with scripts and
  interactive sessions using Python
- *analyze* data with a [rich ecosystem of plugins](https://napari-hub.org)
- and [more!](https://napari.org/community)

## Installation

Ready to get started? Napari can be installed with pip
(`pip install napari`) and conda (`conda install -c conda-forge napari`).
You can read our [installation guide](tutorials/fundamentals/installation) for
more details, or continue with our
[getting started tutorial](tutorials/fundamentals/getting_started).

# Community

## Code of conduct

Napari has a [Code of Conduct](./community/code_of_conduct) that should be
honored by everyone who participates in the `napari` community.

## Ask questions

Still have questions? We use the [image.sc
forum](https://forum.image.sc/tag/napari) to answer usage questions about
napari.

## Report bugs

If you encounter any issues while using napari, please check out our
[Github Issues](https://github.com/napari/napari/issues) and submit a
new one if it doesn't already exist.

## Chat with us

We chat on a platform called [Zulip](https://napari.zulipchat.com). It's a good
place to chat informally about the project, or ask someone from the community
for directions. If you're not sure about the best way to engage with the
project or whether napari can be useful for you, drop by!

## Meet with us

Napari runs weekly or fortnightly meetings that anyone is welcome to attend,
where we discuss project progress and development. Additionally, we have a
number of working groups which meet with varying frequency about specific
topics of napari development. To participate, see the [community
calendar](community/meeting_schedule). (It might be a good idea to drop us a
line on [Zulip](https://napari.zulipchat.com) so we know to expect you!)

[maybe embed calendar here]

## Follow us

We have a Twitter account! Follow us for the latest napari news, or for
inspiration about usage. (We often retweet cool usage demos from our
community!)

[embed twitter feed]

## Citing napari

If you find `napari` useful please cite this repository using its DOI as
follows:

> napari contributors (2019). napari: a multi-dimensional image viewer for
> python. [doi:10.5281/zenodo.3555620](https://zenodo.org/record/3555620)

Note this DOI will resolve to all versions of napari. To cite a specific version
please find the DOI of that version on our [zenodo
page](https://zenodo.org/record/3555620). The DOI of the latest version is in
the badge at the top of this page.
