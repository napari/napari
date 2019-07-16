# napari tutorials

Welcome to the **napari** tutorials! We've provided a couple tutorials to
explore the main usage modes and methods of **napari**, but before we dive
in, let's talk a bit about what **napari** actually is and why we're developing it.

## what is napari?

**napari** is a fast, interactive, multi-dimensional image viewer for Python. It's designed for browsing, annotating, and analyzing large multi-dimensional images. It's built on top of `Qt` (for the GUI), `vispy` (for performant GPU-based rendering), and the scientific Python stack (e.g. `numpy`, `scipy`). It includes critical viewer features out-of-the-box, such as support for large multi-dimensional data, and layering and annotation. By integrating closely with the Python ecosystem, **napari** can be easily coupled to leading machine learning and image analysis tools (e.g. `scikit-image`, `scikit-learn`, `TensorFlow`, `PyTorch`), enabling more user-friendly automated analysis.

We're developing **napari** in the open! But the project is in an **alpha** stage, and there will still occasionally be breaking changes from patch to patch. You can follow progress on this repository, test out new versions as we release them, and contribute ideas and code.

To get a sense of our current plans checkout and contribute to the discussion on some of our [long-term feature issues](https://github.com/napari/napari/issues?q=is%3Aissue+is%3Aopen+label%3A%22long-term+feature%22) on gitub.

## getting started

These tutorials target people who want to use
**napari** as a user. If you are also interested in contributing to **napari** then
please check out [Contributing Guidelines](../CONTRIBUTING.md).

If you've already got **napari** installed then begin with our [Getting Started](getting_started.md) tutorial. For help installing **napari** checkout our [Installing napari](installation.md) tutorial.

## all tutorials

- [installing napari](installation.md)
- [getting started tutorial](getting_started.md)
- [napari viewer tutorial](viewer.md)
- [image layer tutorial](image.md)
- [labels layer tutorial](labels.md)
- [points layer tutorial](points.md)
- [shapes layer tutorial](shapes.md)
- [pyramid layer tutorial](pyramid.md)
- [vectors layer tutorial](vectors.md)
