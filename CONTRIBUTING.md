# contributing

Thank you for your interest in contributing to napari!
Contributions to napari are very welcome and encouraged!
If at any point you have questions or would like to discuss your contribution, the core team and napari community are active on the [napari Zulip chat](https://napari.zulipchat.com/) and available on regular [online community meetings](https://napari.org/dev/community/meeting_schedule.html).
You may also reach out on our [GitHub Issues](https://github.com/napari/napari/issues).
We are here and excited to help!

Read our [contributing guide](https://napari.org/dev/developers/contributing/index.html) for a detailed overview of how to get started with contributing to napari, including how to set up a development environment, how to run tests, and how to submit a pull request.

The [AI use policy and guidelines](https://napari.org/dev/developers/contributing/ai.html) should be read and followed when using AI tools to assist with development.

If you want to contribute to or edit our documentation, go to [napari/docs](https://github.com/napari/docs) and also read our [documentation contributing guide](https://napari.org/dev/developers/contributing/documentation/index.html).

## Hot reload

From a development checkout with a Qt backend installed, run:

```bash
uv pip install -e '.[reload]'
napari --dev
```

Use the **Hot Reload** dock to see watched files, pause reloading, or inspect
widgets. Napari watches `napari` and `napari_builtins` by default. Repeat
`--dev-module` for each extra module, for example
`napari --dev --dev-module my_plugin --dev-module another_plugin`.
For a Python script that creates a viewer, set `NAPARI_DEV=1` before launch
and optionally set `NAPARI_DEV_MODULES` to a comma-separated list.

To try live rebuilding, open **Help > Tips and Tricks**, then edit a button
label in the `TipsWidget` constructor and save. The dock contents rebuild in
place. For other widgets you may need to reopen them or restart Napari to see
constructor changes.
