(plugins-index)=

# Plugins

```{admonition} Introducing npe2
:class: tip
We introduced a new plugin engine in December 2021. The new library
[`npe2`][npe2] is a re-imagining of how napari interacts with plugins. Rather
than importing a package to discover plugin functionality, a static manifest
file is used to declaratively describe a plugin's capabilities. Details can be
found in the [](npe2-manifest-spec).

Plugins targeting `napari-plugin-engine` will continue to work, but we
recommend migrating to `npe2` eventually. `npe2` includes tooling to
help automate the process of migrating plugins. See the [migration
guide](npe2-migration-guide) for details.

For getting started writing new npe2 plugins see the [](npe2-getting-started).
```

napari loves plugins. Plugins allow people to customize and extend to napari.

This document describes:

- How to [build, run and publish a plugin](how-to-build-a-plugin).
- Where to find guides and code samples to help get you started.
- [Guidelines](best-practices) for writing plugins.

If you are looking for plugins, head to the [napari
hub](https://napari-hub.org). Plugins can be installed directly from within
napari or with package installers (pip/conda) via the command line -- to learn
more see {ref}`find-and-install-plugins`.

## What can plugins do?

- Change the look of napari with a color theme
- Add custom widgets and user interface elements
- Add file format support - readers and writers
- Provide sample data

(how-to-build-a-plugin)=

## How to build plugins?

New plugins should target `npe2`. See the {ref}`npe2-getting-started`.

For a guide on how to create a plugin using `napari-plugin-engine`, see
{ref}`plugins-for-plugin-developers`. For a complete list of _hook
specifications_ that developers can implement, see the
{ref}`hook-specifications-reference`.

For special considerations when building a napari plugin, see
{ref}`best-practices`.

## Looking for help

If you have questions, try asking on the [zulip
chat](https://napari.zulipchat.com/). Submit issues to the [napari github
repository](https://github.com/napari/napari).

[npe1]: https://github.com/napari/napari-plugin-engine
[npe2]: https://github.com/tlambert03/npe2
