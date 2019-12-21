# release guide

This guide documents `napari`'s release process.
Currently, it only handles distribution, but as the project matures,
it will include generating release notes, documentation, etc.

This is mainly meant for the core developers who will actually be performing the release.
They will need to have a [PyPI](https://pypi.org) account with upload permissions to the `napari` package.

You will also need the additional `release` dependencies in `requirements/release.txt` to complete the release process.

## determining the version

The version of `napari` is automatically determined by [`versioneer`](https://github.com/warner/python-versioneer)
from the latest [`git` tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) beginning with `v`.
Thus, you'll need to tag the [reference](https://git-scm.com/book/en/v2/Git-Internals-Git-References) with the new version number. It is likely something like `X.Y.Z`. Before making a release though we need to generate the release notes.

## generating release notes

1. Review and cleanup ``docs/release/release_dev.txt``. This may be empty if it has not been
   updated during work on the last release.

2. Make a list of merges, contributors, and reviewers by running
   ``python docs/release/generate_release_notes.py -h`` and following that file's usage. For minor or major releases generate the list to include everything since the last minor or major release.
   For other releases generate the list to include
   everything since the last release for which there
   are release notes (which should just be the last release). For example making the release notes
   for the `0.2.0` release can be done as follows:
   ```
   python docs/release/generate_release_notes.py v0.1.0 master --version 0.2.0 | tee docs/release/release_0_2_0.rst
   ```

3. Paste this list at the end of the ``release_dev.txt``.

4. Scan the PR titles for highlights, deprecations, API changes,
   and bugfixes, and mention these in the relevant sections of the notes.
   Try to present the information in an expressive way by mentioning
   the affected functions, elaborating on the changes and their
   consequences. If possible, organize semantically close PRs in groups.

5. Make sure the file name is of the form ``doc/release/release_<major>_<minor>_<release>.txt``.

6. Copy ``doc/release/release_template.txt`` to
   ``doc/release/release_dev.txt`` for the next release.

7. Copy relevant deprecations from ``release_<major>_<minor>_<release>.txt``
   to ``release_dev.txt``.

8. Make and merge a PR with these release notes before moving onto the next steps.


## Tagging the new release candidate

First we will generate a release candidate, which will contain the letters `rc`.
Using release candidates allows us to test releases on PyPi without using up the actual
release number.

You should include a basic message with the tag `"Version X.Y.Zrc1"`:
```bash
$ git tag -a vX.Y.Zrc1 -m "Version X.Y.Zrc1" master
```

If the tag is meant for a previous version of master, simply reference the specific commit:
```bash
$ git tag -a vX.Y.Zrc1 -m "Version X.Y.Zrc1" abcde42
```

Note here how we are using `rc` for release candidate to create a version of our release we can test
before making the real release.

You can read more on tagging [here](https://git-scm.com/book/en/v2/Git-Basics-Tagging).

## creating the distribution

Before creating a new distribution, make sure to delete any previous ones:
```bash
$ rm -rf dist build
```

Now you can build the distribution:
```bash
$ python setup.py sdist bdist_wheel
```

[`MANIFEST.in`](../MANIFEST.in) determines which non-Python files are included.
Make sure to check that all necessary ones are listed before beginning the release process.

## uploading the release candidate to PyPI

Upload the release candidate with:
```bash
$ python -m twine upload dist/napari-X.Y.Zrc1.tar.gz
```

The release candidate can then be tested with

```bash
$ pip install --pre napari
```
or

```bash
$ pip install -U --pre napari
```
if napari is already installed.

If the release candidate is not what you want, make your changes and repeat the process from the beginning but
incrementing the number after `rc` on tag (e.g. `vX.Y.Zrc2`).

Once you are satisfied with the release candidate it is time to generate the actual release.

## Generating the actual release
To generate the actual release you will now repeat the processes above but now dropping the `rc`.
For example:

```bash
$ git tag -a vX.Y.Z -m "Version X.Y.Z" master
$ rm -rf dist build
$ python setup.py sdist bdist_wheel
$ python -m twine upload dist/napari-X.Y.Z.tar.gz
```

At the very end you should push the new tags to the repo.
```bash
$ git push upstream --tags
```
