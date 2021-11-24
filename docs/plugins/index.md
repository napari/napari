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
within napari; to learn more see {ref}`find-and-install-plugins`.

## What can plugins do?

- Change the look of napari with a color theme - [theming][]
- Add custom dock widgets - [dock widgets][]
- Add file format support - [readers][] and [writers][]
- Provide sample data - [data providers][]

If you'd like a more comprehensive overview of the plugin system, check out
the [`npe2` specification](npe2-manifest-spec). The [plugins guide overview][]
includes code samples and guides that illustrate plugin usage.

## How to build plugins?

See the [getting started guide](npe2-getting-started) for starting a new plugin.

If you have an existing `napari-plugin-engine` plugin and want to migrate to
the new plugin engine, [`npe2`][npe2], see the [migration
guide](npe2-migration-guide).

We recommend referring to the [best practices guide](plugin-best-practices) for
tips on developing a quality plugin.

## Introducing `npe2`

napari interacts with plugins via a _plugin engine_ that

- exposes plugin functionality and metadata for use by napari
- discovers plugins that are installed in the python environment
- manages the life-cycle of plugins within napari

We've introduced a new plugin engine to replace the original
[`napari-plugin-engine`][npe1]. The new library [`npe2`][npe2] is a
re-imagining of how napari interacts with plugins. Rather than importing a
package to discover plugin functionality, a static manifest file is used to
declaratively describe a plugin's capabilities. With `npe2`:

- Plugin discovery is faster and safer because napari can get a description of
  the plugin without needing to importing and run code.
- The capabilities of a plugin are known before any plugin code is run. Napari
  uses that information to specifically invoke plugin functionality only when
  needed.

```{admonition} Backwards compatibility
Plugins targeting `napari-plugin-engine` will continue to work, but we
recommend migrating to `npe2` as soon as possible. `npe2` includes tooling to
help automate the process of migrating plugins. See the [migration
guide](npe2-migration-guide) for details.
```

## Looking for help

If you have questions, try asking on the [zulip
chat](https://napari.zulipchat.com/). Submit issues to the [napari github
repository](https://github.com/napari/napari). Feature requests or issues
regarding the plugin API can be submitted to the [npe2 github
repository](https://github.com/napari/npe2).

## OLD STUFF

For a guide on how to create a plugin, see
{ref}`plugins-for-plugin-developers`. For napari contributors looking to
understand how the napari plugin architecture is implemented, see
{ref}`plugins-for-napari-developers`. For a complete list of _hook
specifications_ that developers can implement, see the
{ref}`hook-specifications-reference`.

napari users can install plugins directly from their instance of napari to enhance their workflows; to learn more see {ref}`find-and-install-plugins`.

## npe2

Manifest
: Hey! It's the [Manifest spec](npe2-manifest-spec)

Migration guide
: Hey! It's a [Migration guide](npe2-migration-guide).

[npe1]: https://github.com/napari/napari-plugin-engine
[npe2]: https://github.com/tlambert03/npe2
