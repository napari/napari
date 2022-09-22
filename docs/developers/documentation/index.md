(docs_contributing_guide)=
# Contributing Documentation

This guide will teach you how to submit new documents to napari's usage documentation.

## Prerequisites

Prerequisites depend on the type of contribution you wish to make. In general
you will require:

- Some familiarity with [`git`](https://git-scm.com)
- A [GitHub](https://github.com) account

If you wish to add/amend documentation that does not contain code, you
will also require:

- A clean conda environment with napari docs dependencies installed:

    - You can install these with `pip install "napari[docs]"`
    - These dependencies will allow you to preview your document locally, as it would appear on `napari.org`
    - They will also install `jupytext`, which you will need to contribute documents containing code or viewer interactions

If you wish to add/amend documentation that does contain code, you will also require:

- A clean conda environment with a development installation of napari, see
  the [contributor guide](napari-contributing#setting-up-a-development-installation)
  for details
- [Jupyter notebook](https://jupyter.org/) installed
- Familiarity with Jupyter notebooks (code cells and markdown cells)
- Familiarity with using napari through a Jupyter notebook

## 0. Before you start

If you'd like to contribute a brand new document to our usage section, it might be worth [opening an issue](https://github.com/napari/napari/issues/new?assignees=&labels=documentation&template=documentation.md&title=)
 on our repository first to discuss the content you'd like to see and get some early feedback from the community.
The napari team can also suggest what type of document would be best suited, and whether there are already
existing documents that could be expanded to include the content you think is lacking.

Examples of documents you might want to contribute are:

- [**Explanations**](../../guides/index) (in [`napari/docs/guides`](https://github.com/napari/napari/tree/main/docs/guides)):
  in depth content about napari architecture, development choices and some complex features
- [**Tutorials**](../../tutorials/index) (in [`napari/docs/tutorials`](https://github.com/napari/napari/tree/main/docs/tutorials)):
  detailed, reproducible step by step guides, usually combining multiple napari features to complete a potentially complex task
- [**How-tos**](../../howtos/index) (in [`napari/docs/howtos/`](https://github.com/napari/napari/tree/main/docs/howtos)):
  simple step by step guides demonstrating the use of common features
- [**Examples**](../../../examples/README.rst) (in [`napari/examples/`](https://github.com/napari/napari/tree/main/examples)):
  code examples of how to use napari
- [**Getting started**](../../tutorials/start_index) (in [`napari/docs/tutorials/fundamentals`](https://github.com/napari/napari/tree/main/docs/tutorials/fundamentals):
  these documents are a mix of tutorials and how-tos covering the fundamentals of installing and working with napari for beginners

```{admonition} Got materials for a workshop?
:class: tip

If you already have teaching materials e.g. recordings, slide decks or jupyter notebooks
hosted somewhere, you can add links to these on our [napari workshops](../../further-resources/napari-workshops.md) page.
```

## 1. Write your documentation

Fork and clone [our repository](https://github.com/napari/napari). If you are
a0mending existing documentation, you can do so in your preferred text editor.
If you wish to add a new tutorial or a how-to, we recommend you use our
[template](./docs_template).

Our goal is that all tutorials and how-tos are easily downloadable and executable by our users.
This helps ensure that they are reproducible and makes them easier to maintain.
We therefore provide a notebook template for our documents. Inside the template
you'll find handy tips for taking screenshots of the viewer, hiding code cells,
using style guides and what to include in the required prerequisites section.

[Jupyter notebooks](https://jupyter.org/) are a great option for our documents, because they allow you to easily combine code and well formatted text in markdown.
However, their [raw JSON format](https://numpy.org/numpy-tutorials/content/pairing.html#background) is not great for version control, so we use [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/) documents in our repository and on napari.org.

Make a copy of `napari/docs/developers/documentation/docs_template.md`.
You can edit the template directly in Jupyter notebook, or in your preferred text editor.

```{admonition} Already have a notebook?
:class: tip

If you have an existing `.ipynb` Jupyter notebook that you'd like to contribute, you can convert it to MyST markdown
and then edit the `.md` file to prepare it for contributing.

Run `jupytext your-notebook.ipynb --to myst` to create a new MyST version of your file,
`your-notebook.md`. Edit this file to include the relevant sections from the docs template.
```

### Next steps

Depending on the type of contribution you are making, you may be able to skip
some steps:

* If you are amending an existing document you can skip straight
  to [Step #3 - Preview your document](#3-preview-your-document)
* If you are adding new documentation and would prefer a simplier workflow,
  you can also skip straight to
  [Step #3 - Preview your document](#3-preview-your-document) and ask the
  napari team to update the TOC for you when you open your pull request
* For all other documentation changes, follow the steps below

## 2. Update TOC

If you are adding a new documentation file, you will need to add your document
to the correct folder based on its content (see the [list above](#0-before-you-start)
for common locations), and update `napari/docs/_toc.yml`.

If you're adding a document
to an existing group, simply add a new `- file:` entry in the appropriate spot. For example, if I wanted to add
a `progress_bars.md` how to guide, I would place it in `napari/docs/howtos` and update `_toc.yml` as below:

```yaml
- file: howtos/index
subtrees:
- titlesonly: True
entries:
- file: howtos/layers/index
subtrees:
- titlesonly: True
    entries:
    - file: howtos/layers/image
    - file: howtos/layers/labels
    - file: howtos/layers/points
    - file: howtos/layers/shapes
    - file: howtos/layers/surface
    - file: howtos/layers/tracks
    - file: howtos/layers/vectors
- file: howtos/connecting_events
- file: howtos/napari_imageJ
- file: howtos/docker
- file: howtos/perfmon
- file: howtos/progress_bars # added
```

To create a new subheading, you need a `subtrees` entry. For example, if I wanted to add `geo_tutorial1.md` and `geo_tutorial2.md`
to a new `geosciences` subheading in tutorials, I would place my documents in a new folder `napari/docs/tutorials/geosciences`,
together with an `index.md` that describes what these tutorials would be about, and then update `_toc.yml` as below:

```yaml
- file: tutorials/index
subtrees:
- entries:
    - file: tutorials/annotation/index
    subtrees:
    - entries:
        - file: tutorials/annotation/annotate_points
    - file: tutorials/processing/index
    subtrees:
    - entries:
        - file: tutorials/processing/dask
    - file: tutorials/segmentation/index
    subtrees:
    - entries:
        - file: tutorials/segmentation/annotate_segmentation
    - file: tutorials/tracking/index
    subtrees:
    - entries:
        - file: tutorials/tracking/cell_tracking
    - file: tutorials/geosciences/index                 # added
    subtrees:                                           # added
    - entries:                                          # added
        - file: tutorials/geosciences/geo_tutorial1     # added
        - file: tutorials/geosciences/geo_tutorial2     # added
```

## 3. Preview your document

If your documentation change includes code, it is important that you ensure
the code is working and executable. This is why you will need to have a
development installation of napari installed. [Examples](..../examples/README)
are automatically executed when the documentation is built and code problems can
also be caught when previewing the built documentation. If your documentation
change does not include code, you only need the napari docs dependencies
installed.

There are two ways you can preview the documentation website, by building
locally or downloading the GitHub Actions built documentation when you submit
your pull request.

To build the documentation locally, run `make docs` from the root of
the `napari` repository (assuming you've installed the [docs prerequisites](#prerequisites)).

```bash
make docs
```

The rendered HTML will be placed in `napari/docs/_build`. Find `index.html` in this folder and drag it
into a browser to preview the website with your new document.

Alternatively, When you submit your pull request, napari continuous integration
includes a GitHub action that builds the documentation and saves the artifact
for you to download. This is another way to check that your built documentation
looks as you expect. To download the built documentation, go to your PR, scroll
down to the continuous integration tests, then:

1. click on 'details' next to 'Build Docs / Build & Upload Artifact (pull_request)'

![doc-continuous-integration-1](images/doc-ci-1.png)

2. click on 'summary' on the top left corner

![doc-continuous-integration-1](images/doc-ci-2.png)

3. scroll down to 'artifacts' and click on 'docs' to download the built documentation

![doc-continuous-integration-1](images/doc-ci-3.png)

## 4. Submit your pull request

Once you have written and previewed your document, it's time to open a pull request to [napari's main repository](https://github.com/napari/napari) and contribute it to our codebase.

If you are simply contributing one file (e.g., a tutorial or how-to page) you
can use the [GitHub web interface to open your pull request](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository). Ensure you
document is added to the correct folder based on its content (see the
[list above](#0-before-you-start) for common locations).

To open a pull request via git and the command line, follow [this guide](https://www.digitalocean.com/community/tutorials/how-to-create-a-pull-request-on-github).
You can also reach out to us on [zulip](https://napari.zulipchat.com/#narrow/stream/212875-general) for assistance!

Not sure where to place your document or update `_toc.yml`? Make a best guess and open the pull request - the napari team will
help you edit your document and find the right spot!
