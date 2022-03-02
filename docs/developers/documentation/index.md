# Contributing Documentation

This guide will teach you how to submit new documents to napari's usage documentation.

## Prerequisites

- Familiarity with `git`
- A [GitHub](https://github.com) account

## 0. Before you start

If you'd like to contribute a brand new document to our usage section, it might be worth [opening an issue](https://github.com/napari/napari/issues/new?assignees=&labels=documentation&template=documentation.md&title=)
 on our repository first to discuss the content you'd like to see and get some early feedback from the community.
The napari team can also suggest what type of document would be best suited, and whether there are already
existing documents that could be expanded to include the content you think is lacking. 

Examples of documents you might want to contribute are:

- **Explanations** (in `napari/docs/guides`): in depth content about napari architecture, development choices and some complex features 
- **Tutorials** (in `napari/docs/tutorials`): detailed, reproducible step by step guides, usually combining multiple napari features to complete a potentially complex task
- **How-tos** (in `napari/docs/howtos/`): simple step by step guides demonstrating the use of common features
- **Getting started** (in `napari/docs/tutorials/fundamentals`): these documents are a mix of tutorials and how-tos covering the fundamentals of installing and working with napari for beginners

```{admonition} Got materials for a workshop?
:class: tip

If you already have teaching materials e.g. recordings, slide decks or jupyter notebooks
hosted somewhere, you can add links to these on our [napari workshops](../../further-resources/napari-workshops.md) page.
```

If you are writing a document whose content is mostly text,
you can write a plain markdown document and skip straight to [Step #4 - Update TOC](#4-update-toc).
If you are writing a how-to guide or tutorial that requires executing code or working with the napari viewer, follow
the steps below to prepare your document.

### Prerequisites for contributing documentation with code

- [Jupyter notebook](https://jupyter.org/) installed
- Familiarity with Jupyter notebooks (code cells and markdown cells)
- Familiarity with using napari through a Jupyter notebook

## 1. Download our template

Our goal is that all tutorials and how-tos are easily downloadable and executable by our users. 
This helps ensure that they are reproducible and makes them easier to maintain. 
We therefore provide a notebook template for our documents.

Fork and clone [our repository](https://github.com/napari/napari), and make a copy of `napari/docs/developers/documentation/docs_template.md`.

You can now edit the template directly in your prefered text editor, or you can open it in Jupyter notebook
if you have Jupytext installed.

```{admonition}
:class: tip

You can install jupytext alone with 

```bash
pip install jupytext
```

or you can install our docs dependenices, which inlcude jupytext with

```bash
pip install "napari[docs]"
```

```

## 2. Write your document

Follow the template to write your document. Inside the template you'll also find handy tips for taking screenshots of the viewer,
hiding code cells, using style guides and what to include in the required prerequisites section.

## 3. Pair your notebook with MyST Markdown

[Jupyter notebooks](https://jupyter.org/) are a great option for our documents, because they allow you to easily combine code and well formatted text in markdown. 
However, their [raw JSON format](https://numpy.org/numpy-tutorials/content/pairing.html#background) is not great for version control, so we use [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/) documents in our repository and on napari.org.  

If you are editing the template directly, your notebook is already in MyST markdown! You can go straight to [Step #4 - Update TOC](#4-update-toc).
Alternatively, if you are working on your own `.ipynb` file, follow these steps to pair your notebook with MyST markdown format before submitting your pull request.

### 3.1 Install napari docs requirements
You can install `jupytext`, as well as other napari docs requirements from the command line:

```
pip install napari[docs]
```

### 3.2 Pair your notebook
Once requirements are installed, you can start Jupyter notebook as you usually would. Pairing your notebook with MyST Markdown
 will now be an option in the notebook's `File -> Jupytext` menu. 
Selecting this option will generate a new markdown file for you in the same working directory as your notebook.
If you pair through Jupyter notebook, your markdown file will be updated every time you save the notebook,
so you don't need to worry about keeping them synced.

You can pair your notebook from the command line as well using the following command:

```
jupytext --set-formats ipynb,myst your_notebook.ipynb
```

Then, after making any changes to your notebook, run:

```
jupytext --sync your_notebook.ipynb
```

You can also just convert a notebook from the command line, though this will not *sync* your notebook with your markdown document - any changes to the notebook would require another conversion. To convert your notebook from the command line run:

```
jupytext your_notebook.ipynb --to myst
```

That's it! `your_notebook.md` is now ready to contribute to napari!

## 4. Update TOC

Add your document to the correct folder based on its content (see the [list above](#0-before-you-start) for common locations), and update `napari/docs/_toc.yml`. 

If you're adding a document
to an existing group, simply add a new `- file:` entry in the appropriate spot. For example, if I wanted to add 
a `progress_bars.md` how to guide, I would place it in `napari/docs/howtos` and update `_toc.yml` as below:

```yml
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

```yml
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

## 5. Preview your document

Once you've added your document to the `docs` folder and updated the `_toc.yml`, you can preview the website
 locally by installing napari and the required dependencies, then running `make docs` from the root of
the `napari` repository.

```{warning}
It's best to run these commands in a fresh conda environment!
```

```bash
pip install ".[all, dev, docs]"
make docs
```

The rendered HTML will be placed in `napari/docs/_build`. Find `index.html` in this folder and drag it
into a browser to preview the website with your new document.

## 6. Submit your pull request

Once you have written and previewed your document, it's time to open a pull request to [napari's main repository](https://github.com/napari/napari) and contribute it to our codebase. 
If you've never opened a Pull Request, you may find [this guide](https://www.digitalocean.com/community/tutorials/how-to-create-a-pull-request-on-github) useful.
You can also reach out to us on [zulip](https://napari.zulipchat.com/#narrow/stream/212875-general) for assistance!

Not sure where to place your document or update `_toc.yml`? Make a best guess and open the pull request - the napari team will
help you edit your document and find the right spot!
