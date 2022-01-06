(plugin-test-deploy)=
# Test and deploy

## 4. Preparing for release

Use the `Framework :: napari` [classifier] in your package's core metadata to
make your plugin more discoverable. If you used the cookiecutter, this has
already been done for you.

Once your package, with its `Framework :: napari` classifier, is listed on
[PyPI], it will also be visible on the [napari hub][hub], alongside all other
napari plugins. More about python packing can be found in the [packaging guide].

You can customize your plugin’s listing for the hub by following this
[guide][hubguide]. If you'd like to know what your _napari hub_ plugin page
will look like, you can use the _napari hub plugin preview_ service - see this
[guide][hub-guide-preview] to get started.

The napari hub reads the metadata of your package and displays it in a number
of places so that users can easily find your plugin and decide if it provides
functionality they need. Most of this metadata lives inside your package
configuration files.

You can also include a napari hub specific description file at
`/.napari/DESCRIPTION.md`. The hub preferentially displays this file over your
repository’s `README.md` when it’s available. This file allows you to maintain a
more developer/repository oriented `README.md` while still making sure potential
users get all the information they need to get started with your plugin.

Finally, once you have curated your package metadata and description, you can
preview your metadata, and check any missing fields using the `napari-hub-cli`
tool. Install this tool using

```bash
pip install napari-hub-cli
```

and preview your metadata with

```bash
napari-hub-cli preview-metadata ./my-npy-reader
```

For more information on the tool see the `napari-hub-cli`
[README](https://github.com/chanzuckerberg/napari-hub-cli).

If you want your plugin to be available on PyPI, but not visible on the napari
hub, you can add a `.napari/config.yml` file to the root of your repository
with a visibility key. For details, see the [customization
guide][hub-guide-custom-viz].

## 5. Share your plugin with the world

Once you are ready to share your plugin, [upload the Python package to
PyPI][pypi-upload] and it can then be installed with a simple
`pip install mypackage`. If you used the {ref}`plugin-cookiecutter-template`,
you can also [setup automated deployments][autodeploy].

If you are using Github, add the ["napari-plugin"
topic](https://github.com/topics/napari-plugin) to your repo so other
developers can see your work.

The [napari hub](https://www.napari-hub.org/) automatically displays
information about PyPI packages annotated with the `Framework :: napari`
[Trove classifier](https://pypi.org/classifiers/), to help end users discover
plugins that fit their needs. To ensure you are providing the relevant
metadata and description for your plugin, see the following documentation in
the [napari hub
GitHub](https://github.com/chanzuckerberg/napari-hub/tree/main/docs)’s docs
folder:

- [Customizing your plugin’s listing](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/customizing-plugin-listing.md)
- [Writing the perfect description for your plugin](https://github.com/chanzuckerberg/napari-hub/blob/main/docs/writing-the-perfect-description.md)

For more about the napari hub, see the [napari hub About page](https://www.napari-hub.org/about).
To learn more about the hub’s development process, see the [napari hub GitHub’s Wiki](https://github.com/chanzuckerberg/napari-hub/wiki).

When you are ready for users, announce your plugin on the [Image.sc Forum](https://forum.image.sc/tag/napari).
