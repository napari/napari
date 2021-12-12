(plugins-index)=

# Plugins

napari loves plugins. Plugins allow people to add new readers and writers for
accessing new kinds of data, customize napari's appearance, or add new widgets
for interacting with data.

This document describes:

- How to build, run and publish a plugin.
- Where to find guides and code samples to help get you started.
- Guidelines for writing plugins.

If you are looking for plugins, head to the
[napari hub](https://napari-hub.org). Plugins can be installed directly from
within napari -- to learn more see {ref}`find-and-install-plugins`.

```{admonition} Introducing npe2
We've introduced a new plugin engine. The new library [`npe2`][npe2] is a
re-imagining of how napari interacts with plugins. Rather than importing a
package to discover plugin functionality, a static manifest file is used to
declaratively describe a plugin's capabilities. Details can be found in the
{ref}`npe2-manifest-spec`.

Plugins targeting `napari-plugin-engine` will continue to work, but we
recommend migrating to `npe2` as soon as possible. `npe2` includes tooling to
help automate the process of migrating plugins. See the [migration
guide](npe2-migration-guide) for details.

For getting started writing new npe2 plugins see the
{ref}`npe2-getting-started`.
```

## What can plugins do?

- Change the look of napari with a color theme
- Add custom dock widgets
- Add file format support - readers and writers
- Provide sample data

## How to build plugins?

For a guide on how to create a plugin, see
{ref}`plugins-for-plugin-developers`. For a complete list of _hook
specifications_ that developers can implement, see the
{ref}`hook-specifications-reference`.

For special considerations when building a napari plugin, see
{ref}`best-practices`.

## Looking for help

If you have questions, try asking on the [zulip
chat](https://napari.zulipchat.com/). Submit issues to the [napari github
repository](https://github.com/napari/napari). Feature requests or issues
regarding the plugin API can be submitted to the [npe2 github
repository](https://github.com/napari/npe2).

For napari contributors looking to
understand how the napari plugin architecture is implemented, see
{ref}`plugins-for-napari-developers`.

[npe1]: https://github.com/napari/napari-plugin-engine
[npe2]: https://github.com/tlambert03/npe2
