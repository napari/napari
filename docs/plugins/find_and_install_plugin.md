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

2. In the resulting window that opens, where it says “Install by name/URL”, type the name of the plugin you want to install.

   ![napari viewer's Plugin dialog. At the bottom of the dialog, there is a place to install by name, URL, or dropping in a file.](/images/plugin-install-dialog.png)

3. Click the “Install” button next to the input bar.

```{note}
When wanting to install a plugin from a URL you need to respect a specific syntax required by pypi!
You need to add the vcs (version control system) information before the URL for pip to be able to handle the URL.
You will find more information about this on the [pip documentation](https://pip.pypa.io/en/stable/topics/vcs-support/)


Example:
If you want to install the plugin `napari-svg` directly from the [source code url](https://github.com/napari/napari-svg), you need to write `git+https://github.com/napari/napari-svg.git` in the text field.
```
