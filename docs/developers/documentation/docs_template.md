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
---

# Docs template

This template will guide you to write a well formatted document and prepare it for contribution to napari.org.

## Prerequisites

Fill out this section with a list of things the reader will need to prepare to be able to follow this document.
Include things like:
- links to any existing napari.org tutorials or how-to guides that can help the user fulfill these prerequisites
- the level of python/napari knowledge required to follow this document
    - try to be specific about what skills are needed e.g. 
    'connecting callbacks to layer events' or 'using matplotlib to produce plots'
- plugins that should be installed
- python packages that should be installed (don't list napari or its dependencies)
- links to sample data that can be used to follow your document
  - **don't add this data to the repository**

## Write your document
Fill out the main content of the document - your explanation, how-to steps or tutorial. 

## Include pictures

You should include pictures/videos, particularly when describing interactions with the viewer. If an action can be performed both through the viewer and through code, you should include both options. Keep accessibility in mind when adding images. Each image should be
accompanied by complete and descriptive alt-text. If you're using arrows/circles to highlight portions of the image, make sure
they contrast well against the background of the image.

### Use GitHub's media hosting

GitHub now hosts videos and images added to comments and documents via their drag and drop API.
This is not only very simple to achieve, but it keeps our repositories small.
To use GitHub image hosting, upload your document to your pull request with no images. 
Then, navigate to the document in your pull request and select "Edit file" from the three-dot menu on the top right of your file.
Once you're in file editing mode, place your cursor on the line where you want the image to appear, and drag and drop the image.
Don't forget to edit your alt text!

### Take screenshots of the viewer

It's common for napari documentation to include code that has some effect in the napari Viewer e.g. adding layers,
changing layer properties or using a plugin. These changes in the Viewer should be shown to the user, and this can
be easily achieved in your notebook with napari's `nbscreenshot` utility. Follow these steps to include screenshots of the napari viewer and hide code cells.

#### 1. Use `nbscreenshot` for screenshots of the viewer

This utility allows you to pass an active Viewer as a parameter and produces a screenshot of the Viewer at that 
point in time. This screenshot will be displayed to the user in the notebook.

```{code-cell} ipython3
import napari
from napari.utils import nbscreenshot

viewer = napari.Viewer()
# opens sample data and adds layer to the Viewer
viewer.open_sample('scikit-image', 'cells3d')

# takes a screenshot and produces it as output for this cell
nbscreenshot(viewer)
```

#### 2. Hide input cells

As you can see, it's simple to produce screenshots of the napari Viewer in your notebooks. However, if you look through napari's
existing documentation, none of the code cells include calls to `nbscreenshot`, yet the screenshots are still produced. In fact,
it would be distracting if all the code cells included `nbscreenshot`, and might be frustrating for users who
want to execute these notebooks in their own workflows.

To avoid this frustration, we place calls to `nbscreenshot` in a hidden cell in your notebooks.
You can completely remove input (i.e. the code that's running) in a notebook cell by adding a `remove-input` tag to the cell metadata.

How you add cell tags depends on how you're editing your notebook. 

1. If you're working in Jupyter notebook,
you can open up the Tags toolbar for your cell using `View -> Cell Toolbar -> Tags`. You can then add any tags you want
(e.g. `remove-input`) by typing into the text entry box of the toolbar and clicking `Add Tag`. 
Here's what the Tags toolbar looks like.

![Jupyter notebook cell with Tags toolbar highlighted by a black square in the top right of the cell. Tags toolbar involves a button
with ellipses for seeing existing tags, a text entry box for adding new tags, and an Add Tag button](images/jupyter_cell_tags.png)

2. If you're editing a MyST Markdown file directly, you can add tags to your code blocks like so:

    ```{code-cell}
    :tags: [remove-input]

    print("Your code here")
    ```

#### What to put in hidden cells

Alongside your call to `nbscreenshot`, you can also place other potentially distracting code in these tagged cells, 
such as resizing the Viewer window or opening a menu. In general, if you're running code the reader isn't meant to run,
this should be in a hidden cell.
The screenshot below is produced by the following code, which has been hidden from you using the `remove-input` tag.

```python
from napari.utils import nbscreenshot

viewer.window._qt_window.resize(750, 550)
viewer.dims.current_step = (25, 0, 1)
nbscreenshot(viewer)
```

Note how we've included the `nbscreenshot` import in this hidden cell. Even though in the
example above we imported `nbscreenshot` to show its functionality, you should place the
import in a hidden cell when you write your documentation.

```{code-cell} ipython3
:tags: [remove-input]

from napari.utils import nbscreenshot

viewer.window._qt_window.resize(750, 550)
viewer.dims.current_step = (25, 0, 1)
nbscreenshot(viewer)
```

Here are some examples of settings you might want to use in a hidden input cell to make your screenshot look pretty:
- window resizing (as above)
- [toggling visible layers](https://napari.org/api/stable/napari.layers.Layer.html#napari.layers.Layer.visible)
- [setting the slider position to a particular slice](https://napari.org/api/stable/napari.components.Dims.html#napari.components.Dims.current_step)
- [adjusting contrast limits](https://napari.org/api/stable/napari.layers.Image.html#napari.layers.Image.contrast_limits)
- [switching between 2D and 3D](https://napari.org/api/stable/napari.components.Dims.html#napari.components.Dims.ndisplay)
- switching to grid mode - use `viewer.grid.enabled = True` to enable grid mode
- [setting the camera zoom](https://napari.org/api/stable/napari.components.Camera.html#napari.components.Camera.zoom)

## Check that your notebook runs

If your guide contains any code the user is expected to run, make sure that the notebook can be executed from start to finish without requiring any data or packages that you haven't listed in the prerequisites.

## Use the Google developer's style guide

This [style guide](https://developers.google.com/style/) should answer all your questions about when to italicise and when to bold, which
words to capitalize in your headings (spoiler - we use sentence case for our headings) and other style conventions.
