(plugins-index)=
# Plugins


```{note}
These pages describe the process of **building** a plugin.

If you are looking to **use** published plugins, see the
[guide on installing plugins](find-and-install-plugins), or head to the [napari hub][napari_hub] to
search for plugins.
```

Plugins allow developers to customize and extend napari.  This includes

- Adding file format support with [readers] and [writers]
- Adding custom [widgets] and user interface elements
- Providing [sample data][sample_data]
- Changing the look of napari with a color [theme]

Here you can find:

- How to [build, test and publish a plugin](how-to-build-a-plugin).
- Where to find [guides](./guides) to help get you started.
- [Best practices](./best_practices) when developing plugins.

```{admonition} Introducing npe2
:class: important

We introduced a new plugin engine [`npe2`][npe2] in December 2021.

Unless otherwise stated, most of the documentation herein pertains
to the new npe2 format (which uses a static `napari.yaml` manifest)

Plugins targeting the first generation `napari-plugin-engine` 
(using `@napari_hook_implementation` decorators) will
continue to work for at least the first half of 2022, but we
recommend migrating to `npe2`. See the
[migration guide](npe2-migration-guide) for details.
```

(how-to-build-a-plugin)=
## How to build plugins

If you're just getting started with napari plugins, try our
[Your First Plugin](./first_plugin) tutorial.

For a list of all available contribution points and specifications,
see the [Contributions reference](./contributions)

If you're ready to publish your plugin, see [Test and deploy](./test_deploy)

For special considerations when building a napari plugin, see
{ref}`best-practices`.

## Looking for help?

If you have questions, try asking on the [zulip chat][napari_zulip].
Submit issues to the [napari github repository][napari_issues].

[npe1]: https://github.com/napari/napari-plugin-engine
[npe2]: https://github.com/napari/npe2
[napari_issues]: https://github.com/napari/napari/issues/new/choose
[napari_zulip]: https://napari.zulipchat.com/
[napari_hub]: https://napari-hub.org
[readers]: contributions-readers
[writers]: contributions-writers
[widgets]: contributions-widgets
[sample_data]: contributions-sample-data
[theme]: contributions-themes
