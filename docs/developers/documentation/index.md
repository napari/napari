---
jupytext:
  formats: ipynb,md:myst
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
  intro: This guide will walk you through submitting a new document to napari.org.
---

# Contributing Documentation

## Prerequisites
- [Jupyter notebook](https://jupyter.org/) installed
- Familiarity with Jupyter notebooks (code cells and markdown cells)
- Familiarity with using napari through a Jupyter notebook
- A [GitHub account](https://github.com)

## 0. Before you start

If you'd like to contribute a brand new tutorial or how-to, it might be worth [opening an issue](https://github.com/napari/napari/issues/new?assignees=&labels=documentation&template=documentation.md&title=)
 on our repository first to discuss the content you'd like to see and get some early feedback from the community.

Our main types of documents are:

- **Explanations** (in `napari/docs/guides`): in depth content about napari architecture, development choices and some complex features 
- **Tutorials** (in `napari/docs/tutorials`): detailed, reproducible step by step guides, usually combining multiple napari features to complete a potentially complex task
- **How-tos** (in `napari/docs/howtos/`): simple step by step guides demonstrating the use of common features

The napari team can also suggest the type of document that would be best suited, and whether there are already
existing documents that could be expanded to include the content you think is lacking. 

## 1. Download our template

Our goal is that all tutorials and how-tos are easily downloadable and executable by our users. 
This helps ensure that they are reproducible and makes them easier to maintain. 
We therefore provide a notebook template for our documents.

To download, [click on this link](https://raw.githubusercontent.com/napari/napari.github.io/9aac1c2d69ef0b0bc196ec1ccacba1c1f806eea4/developers/documentation/docs_template.md) then right-click anywhere on the page and click `Save as`. 

```{admonition} Choose .md as the file extension
:class: important

This will open a Save dialog and prepopulate the filename and extension. Often, the default file extension chosen will be `.txt`.
You need to change this extension to `.md` to open this file later in Jupyter notebook.

On Mac, changing file extensions in the save dialog doesn't always work. You may have to navigate to the file in your directory
and rename it without the `.txt` extension.
```

## 2. Write your document

Follow the template to write your document. Inside the template you'll also find handy tips for taking screenshots of the viewer,
hiding code cells, using style guides and what to include in the required prerequisites section.

## 3. Pair your notebook with MyST Markdown

[Jupyter notebooks](https://jupyter.org/) are a great option for our documents, because they allow you to easily combine code and well formatted text in markdown. 
However, their [raw JSON format](https://numpy.org/numpy-tutorials/content/pairing.html#background) is not great for version control, so we use [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/) documents in our repository and on napari.org.  

Follow these steps to pair your notebook with MyST markdown format before submitting your pull request.

### 3.1 Install Jupytext
You can install `jupytext` from the command line:

```
pip install jupytext
```

or

```
conda install jupytext -c conda-forge
```

### 3.2 Pair your notebook
Once installed, you can start Jupyter notebook as you usually would. Pairing your notebook with MyST Markdown
 will now be an option in the notebook's `File -> Jupytext` menu, as in the screenshot below. Selecting this option will generate a new markdown file
 for you in the same working directory as your notebook.
If you pair through Jupyter notebook, your markdown file will be updated every time you save the notebook,
so you don't need to worry about keeping them synced.

![Screenshot of Jupyter Notebook with File -> Jupytext menu open and Pair Notebook with MyST Markdown selected.](images/jupyter_jupytext.png)

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
## 4. Submit your pull request

Once you have written and prepared your document, it's time to open a pull request to [napari's main repository](https://github.com/napari/napari) and contribute it to our codebase. 
If you're not familiar with Git or pull requests, follow [this how-to](./open_pr_online.md) to open your pull request online through GitHub. 
If you already know how to submit pull requests but aren't sure where to put your document, here are the folder paths:

- **Explanations** are in `napari/docs/guides/`
- **Tutorials** are in `napari/docs/tutorials/`
- **How-tos** are in `napari/docs/howtos/`

Not sure where to place your document? Make a best guess and open the pull request - the napari team will
help you edit your document and find the right spot!
