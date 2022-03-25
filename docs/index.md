---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
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

## Visualize images, NumPy arrays, or other arrays

Napari is a fast multi-dimensional image viewer in Python
and for Python. (But it *might* be useful for you outside of
Python![*](no-python))

```{code-cell} ipython3
:tags: [hide-input]
# just setting up data
from skimage import data
cells = data.cells3d()
```

```{code-cell} ipython3
import napari
viewer = napari.view_image(
    cells,  # a 4D NumPy array
    channel_axis=1,
    ndisplay=3,
)
```

```{code-cell} ipython3
:tags: [remove-input]

from napari.utils import nbscreenshot
viewer.camera.angles = (-30, 30, -135)
viewer.camera.zoom = 6.5
nbscreenshot(viewer)
```

## Annotate images with segmentations, points, polygons, and more

Napari can display complex annotations on top of your images, whether generated
by a Python computation, manually edited, or a combination of the two.

```{code-cell} ipython3
from skimage import features, filters, segmentation, util

nuclei = cells[:, 1]
smoothed = filters.gaussian(nuclei, sigma=20)
thresholded = filters.threshold_otsu(smoothed) < smoothed
maxima = features.peak_local_max(thresholded)

labels = util.label_points(coords, nuclei.shape)
edges = filters.farid(nuclei)

segments = segmentation.watershed(
    edges, labels, mask=thresholded, compactness=0.01
)

maxima_layer = viewer.add_points(maxima)
```

```{code-cell} ipython3
:tags: [remove-input]
screenshots = []
screenshots.append(viewer.screeshot())
```

```{code-cell} ipython3
edges_layer = viewer.add_image(edges, colormap='bop orange')
```

```{code-cell} ipython3
:tags: [remove-input]
screenshots.append(viewer.screeshot())
```

```{code-cell} ipython3
segments_layer = viewer.add_labels(segments)
```

```{code-cell} ipython3
:tags: [remove-input]
screenshots.append(viewer.screeshot())
```

```{code-cell} ipython3
:tags: [remove-input]

for coord in []:
    points.add(coord)
    screenshots.append(viewer.screenshot())

```

```{code-cell} ipython3
new_labels = util.label_points(maxima_layer.data, nuclei.shape)
new_segments = segmentation.watershed(
    edges, new_labels, mask=thresholded, compactness=0.01
)
segments_layer.data = new_segments
```

```{code-cell} ipython3
:tags: [remove-input]

screenshots.append(viewer.screenshot())
# HELP! generate and display gif or preferably mp4 of screenshots here.
```

## Use plugins to add capabilities to napari, including data analysis

[carousel of plugins. Highlights for me include napari-clusters-plotter,
napari-skimage-regionprops, napari-feature-classifier,
napari-time-series-plotter, napari-properties-plotter, napari-arboretum.]

## Installation

Ready to get started? Please read our [installation
guide](tutorials/fundamentals/installation).

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
