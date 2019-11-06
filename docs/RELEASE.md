# release guide

This guide documents `napari`'s release process.
Currently, it only handles distribution, but as the project matures,
it will include generating release notes, documentation, etc.

This is mainly meant for the core developers who will actually be performing the release.
They will need to have a [PyPI](https://pypi.org) account with upload permissions to the `napari` package.

## determining the version

The version of `napari` is automatically determined by [`versioneer`](https://github.com/warner/python-versioneer)
from the latest [`git` tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) beginning with `v`.
Thus, you'll need to tag the [reference](https://git-scm.com/book/en/v2/Git-Internals-Git-References) with the new version number.
You should include a message with the tag but because we don't generate release notes for now,
this will be a basic `"Version X.Y.Zrc1"`:
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

## uploading to PyPI

You'll need `twine` installed for this step:
```bash
$ pip install twine
```

To make sure that everything is working properly, first upload to `test.pypi.org`,
entering your credentials when prompted:
```bash
$ python -m twine upload --repository-url=https://test.pypi.org/legacy/ dist/*
```

Then create a new environment and download the release from the test servers:
```bash
$ python -m pip install --extra-index-url https://test.pypi.org/simple/ napari==X.Y.Z
```

Try running some examples and tests to verify that everything is working properly.
If these fail, delete the tag with:
```bash
$ git tag -d vX.Y.Z
```

Make your changes and repeat the process from the beginning but with a slightly different tag (e.g. `vX.Y.Z.0`).
Once the release passes, remember to reset the tag name to the original `vX.Y.Z`
to avoid uploading a mislabeled version.

Then, you may complete the release with:
```bash
$ python -m twine upload dist/*
```

Don't forget to push the new tag to the repo!
```bash
$ git push upstream --tags
```

## generating release notes

1. Review and cleanup ``docs/release/release_dev.txt``.

2. Make a list of merges, contributors, and reviewers by running
   ``python docs/release/generate_release_notes.py -h`` and following that file's usage. For minor or major releases generate the list to include everything since the last minor or major release.
   For other releases generate the list to include
   everything since the last release for which there
   are release notes (which should just be the last release). For example making the release notes
   for the `0.2.0` release can be done as follows:
   ```
   python docs/release/generate_release_notes.py v0.1.0 master --version 0.2.0 | tee docs/release/release_0_2.rst
   ```

3. Paste this list at the end of the ``release_dev.txt``.

4. Scan the PR titles for highlights, deprecations, API changes,
   and bugfixes, and mention these in the relevant sections of the notes.
   Try to present the information in an expressive way by mentioning
   the affected functions, elaborating on the changes and their
   consequences. If possible, organize semantically close PRs in groups.

5. Rename the file to ``doc/release/release_<major>_<minor>.txt`` for a minor release and ``doc/release/release_<major>_<minor>_<release>.txt`` otherwise

6. Copy ``doc/release/release_template.txt`` to
   ``doc/release/release_dev.txt`` for the next release.

7. Copy relevant deprecations from ``release_<major>_<minor>_<release>.txt``
   to ``release_dev.txt``.
