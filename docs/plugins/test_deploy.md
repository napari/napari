(plugin-test-deploy)=
# Test and Deploy

(plugin-testing-tips)=
## Tips for testing napari plugins

Testing is a big topic!  If you are completely new to writing tests in python,
consider reading this post on [Getting Started With Testing in
Python](https://realpython.com/python-testing/)

We recommend using
[pytest](https://docs.pytest.org/en/6.2.x/getting-started.html) for testing your
plugin. Aim for [100% test coverage](./best_practices.md#how-to-check-test-coverage)!

### The `make_napari_viewer` fixture

Testing a napari `Viewer` requires some setup and teardown each time.  We have
created a [pytest fixture](https://docs.pytest.org/en/6.2.x/fixture.html) called
`make_napari_viewer` that you can use (this requires that you have napari
installed in your environment).

To use a fixture in pytest, you simply include the name of the fixture in the
test parameters (oddly enough, you don't need to import it!).  For example, to
create a napari viewer for testing:

```
def test_something_with_a_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    ...  # carry on with your test
```

### Prefer smaller unit tests when possible

The most common issue people run into when designing tests for napari plugins is
that they try to test everything as a full "integration test", starting from the
napari event or action that would trigger their plugin to do something.  For
example, let's say you have a dock widget that connects a mouse callback to the
viewer:

```py
class MyWidget:
    def __init__(self, viewer: 'napari.Viewer'):
        self._viewer = viewer

        @viewer.mouse_move_callbacks.append
        def _on_mouse_move(viewer, event):
            if 'Shift' in event.modifiers:
                ...

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return MyWidget
```

You might think that you need to somehow simulate a mouse movement in napari in
order to test this, but you don't! Just *trust* that napari will call this
function with a `Viewer` and an `Event` when a mouse move has been made, and
otherwise leave `napari` out of it.

Instead, focus on "unit testing" your code: just call the function directly with
objects that emulate, or "mock" the objects that your function expects to
receive from napari. You may also need to slightly reorganize your code.  Let's
modify the above widget to make it easier to test:

```py
class MyWidget:
    def __init__(self, viewer: 'napari.Viewer'):
        self._viewer = viewer
        # connecting to a method rather than a local function
        # makes it easier to test
        viewer.mouse_move_callbacks.append(self._on_mouse_move)

    def _on_mouse_move(self, viewer, event):
        if 'Shift' in event.modifiers:
            ...
```

To test this, we can often just instantiate the widget with our own viewer, and
then call the methods directly. As for the `event` object, notice that all we
care about in this plugin is that it has a `modifiers` attribute that may or may
not contain the string `"Shift"`.  So let's just fake it!

```py
class FakeEvent:
    modifiers = {'Shift'}

def test_mouse_callback(make_napari_viewer):
    viewer = make_napari_viewer()
    wdg = MyWidget(viewer)
    wdg._on_mouse_move(viewer, FakeEvent())
    # assert that what you expect to happen actually happened!
```

## Preparing for release

To help users find your plugin, make sure to use the `Framework :: napari`
[classifier] in your package's core metadata. (If you used the cookiecutter,
this has already been done for you.)

Once your package is listed on [PyPI] (and includes the `Framework :: napari`
[classifier]), it will also be visible on the [napari
hub](https://napari-hub.org/). To ensure you are providing the relevant metadata and
description for your plugin, see the following documentation in the [napari hub
GitHub](https://github.com/chanzuckerberg/napari-hub/tree/main/docs)’s docs
folder:

- [Customizing your plugin’s
  listing](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/customizing-plugin-listing.md)
- [Writing the perfect description for your
  plugin](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/writing-the-perfect-description.md)

```{admonition} The hub
For more about the napari hub, see the [napari hub About
page](https://www.napari-hub.org/about). To learn more about the hub’s
development process, see the [napari hub GitHub’s
Wiki](https://github.com/chanzuckerberg/napari-hub/wiki).

If you want your plugin to be available on PyPI, but not visible on the napari
hub, you can add a `.napari/config.yml` file to the root of your repository with
a visibility key. For details, see the [customization
guide][hub-guide-custom-viz].
```

Finally, once you have curated your package metadata and description, you can
preview your metadata, and check any missing fields using the
napari hub preview page service. Check out [this guide](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/setting-up-preview.md) for instructions on how to set it up. 

## Deployment

When you are ready to share your plugin, [upload the Python package to
PyPI][pypi-upload] after which it will be installable using `pip install
<yourpackage>`, or (assuming you added the `Framework :: napari` classifier)
in the builtin plugin installer dialog.

If you used the {ref}`plugin-cookiecutter-template`, you can also
[setup automated deployments][autodeploy] on github for every tagged commit.

````{admonition} What about conda?
While you are free to distribute your plugin on anaconda cloud in addition to
or instead of PyPI, the built-in napari plugin installer doesn't currently install
from conda. In this case, you may guide your users to install your package on the
command line using conda in your readme or documentation.

A future version of napari and the napari stand-alone application may support
directly installing from conda.
````

When you are ready for users, announce your plugin on the [Image.sc
forum](https://forum.image.sc/tag/napari).


[classifier]: https://pypi.org/classifiers/
[pypi]: https://pypi.org/
[pypi-upload]: https://packaging.python.org/tutorials/packaging-projects/#uploading-the-distribution-archives
[hubguide]: https://github.com/chanzuckerberg/napari-hub/blob/main/docs/customizing-plugin-listing.md
[hub-guide-custom-viz]: https://github.com/chanzuckerberg/napari-hub/blob/main/docs/customizing-plugin-listing.md#visibility
[hub-guide-preview]: https://github.com/chanzuckerberg/napari-hub/blob/main/docs/setting-up-preview.md
[autodeploy]: https://github.com/napari/cookiecutter-napari-plugin#set-up-automatic-deployments

