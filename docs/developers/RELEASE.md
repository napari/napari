# Release Guide

This guide documents `napari`'s release process.
Currently, it only handles distribution, but as the project matures,
it will include generating release notes, documentation, etc.

This is mainly meant for the core developers who will actually be performing the release.
They will need to have a [PyPI](https://pypi.org) account with upload permissions to the `napari` package.

You will also need the additional `release` dependencies (`pip install -e .[release]`) to complete the release process.

> [`MANIFEST.in`](../MANIFEST.in) determines which non-Python files are included.
> Make sure to check that all necessary ones are listed before beginning the release process.

The `napari/napari` repository must have a PyPI API token as a GitHub secret.
This likely has been done already, but if it has not, follow
[this guide](https://pypi.org/help/#apitoken) to gain a token and
[this guide](https://help.github.com/en/actions/automating-your-workflow-with-github-actions/creating-and-using-encrypted-secrets)
to add it as a secret.

## determining the version

The version of `napari` is automatically determined at install time by
[`setuptools_scm`](https://github.com/pypa/setuptools_scm) from the latest
[`git` tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) beginning with
`v`. Thus, you'll need to tag the
[reference](https://git-scm.com/book/en/v2/Git-Internals-Git-References) with
the new version number. It is likely something like `X.Y.Z`. Before making a
release though we need to generate the release notes.

## generating release notes

1. Make a list of merges, contributors, and reviewers by running
   ``python docs/release/generate_release_notes.py -h`` and following that file's usage.
   For each release generate the list to include everything since the last release for which there
   are release notes (which should just be the last release). For example making the release notes
   for the `0.2.1` release can be done as follows:

   ```bash
   python docs/release/generate_release_notes.py v0.2.0 master --version 0.2.1 | tee docs/release/release_0_2_1.md
   ```

2. Scan the PR titles for highlights, deprecations, API changes,
   and bugfixes, and mention these in the relevant sections of the notes.
   Try to present the information in an expressive way by mentioning
   the affected functions, elaborating on the changes and their
   consequences. If possible, organize semantically close PRs in groups.

3. Make sure the file name is of the form ``doc/release/release_<major>_<minor>_<release>.md``.

4. Make and merge a PR with these release notes before moving onto the next steps.

## tagging the new release candidate

First we will generate a release candidate, which will contain the letters `rc`.
Using release candidates allows us to test releases on PyPI without using up the actual
release number.

You can tag the current source code as a release candidate with:

```bash
git tag vX.Y.Zrc1 master
```

If the tag is meant for a previous version of master, simply reference the specific commit:

```bash
git tag vX.Y.Zrc1 abcde42
```

Note here how we are using `rc` for release candidate to create a version of our release we can test
before making the real release.

You can read more on tagging [here](https://git-scm.com/book/en/v2/Git-Basics-Tagging).

## testing the release candidate

Our CI automatically makes a release, copying the release notes to the tag and uploading the distribution to PyPI.
You can trigger this by pushing the new tag to `napari/napari`:

```bash
git push upstream --tags
```

The release candidate can then be tested with

```bash
pip install --pre napari
```

It is recommended that the release candidate is tested in a virtual environment in order to isolate dependencies.

If the release candidate is not what you want, make your changes and repeat the process from the beginning but
incrementing the number after `rc` on tag (e.g. `vX.Y.Zrc2`).

Once you are satisfied with the release candidate it is time to generate the actual release.

## generating the actual release

To generate the actual release you will now repeat the processes above but now dropping the `rc`.
For example:

```bash
git tag vX.Y.Z master
git push upstream --tags
```
