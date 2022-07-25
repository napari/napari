(find-and-install-plugins)=
# Finding and installing a napari plugin

napari plugins are Python packages distributed on the Python Package Index
(PyPI), and annotated with the tag [`Framework ::
napari`](https://pypi.org/search/?q=&o=&c=Framework+%3A%3A+napari).  The
[napari hub](https://napari-hub.org) uses this data, together with additional
metadata, to produce a more user friendly way to find napari plugins.

Similarly, plugins annotated on PyPI with `Framework :: napari` are listed in
the `Plugins > Install/Uninstall Plugins` menu within napari.

## Finding plugins on the napari hub

The [napari hub](https://napari-hub.org) hosts information about all plugins.
You can browse, search, and filter to find plugins that fit your needs.
You can also share links to specific search results and plugins.

## Installing plugins with napari

All PyPI packages annotated with the `Framework :: napari` tag can be installed
directly from within napari:

1. From the “Plugins” menu, select “Install/Uninstall Plugins...”.

   ![napari viewer's Plugins menu with Install/Uninstall Plugins as thr first item.](/images/plugin-menu.png)

2. In the resulting window that opens, where it says “Install by name/URL”,
    enter the name of the plugin you want to install (or *any* valid pip
    [requirement
    specifier](https://pip.pypa.io/en/stable/reference/requirement-specifiers/)
    or [VCS scheme](https://pip.pypa.io/en/stable/topics/vcs-support))


   ![napari viewer's Plugin dialog. At the bottom of the dialog, there is a place to install by name, URL, or dropping in a file.](/images/plugin-install-dialog.png)

   ```{admonition} Example
   If you want to install `napari-svg` directly from the development branch on the [github repository](https://github.com/napari/napari-svg), enter `git+https://github.com/napari/napari-svg.git` in the text field.
   ```

3. Click the “Install” button next to the input bar.
