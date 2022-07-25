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
If you want to install a plugin from a URL, you need to use the required syntax by either using any valid pip [VCS scheme](https://pip.pypa.io/en/stable/topics/vcs-support) (ex. `git`, `svn`) or [requirement specifier](https://pip.pypa.io/en/stable/reference/requirement-specifiers/).

Example:
If you want to install the plugin `napari-svg` directly from the [source code url](https://github.com/napari/napari-svg), you need to write `git+https://github.com/napari/napari-svg.git` in the text field.
```
