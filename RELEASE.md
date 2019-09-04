# release guide

This guide documents `napari`'s release process.
Currently, it only handles distribution, but as the project matures,
will include generating release notes, documentation, etc.

## determining the version

The version of `napari` is automatically determined by [`versioneer`](https://github.com/warner/python-versioneer)
from the latest [`git` tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) beginning with `v`.
Thus, you'll need to tag the [reference](https://git-scm.com/book/en/v2/Git-Internals-Git-References) with the new version number.
You should include a message with the tag but because we don't generate release notes for now,
this will be a basic `"Version X.Y.Z"`:
```bash
$ git tag -a vX.Y.Z -m "Version X.Y.Z" master
```

If the tag is meant for a previous version of master, simply reference the specific commit:
```bash
$ git tag -a vX.Y.Z -m "Version X.Y.Z" abcde42
```

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

[`MANIFEST.in`](MANIFEST.in) determines which non-Python files are included.
Make sure to check that all necessary ones are listed before beginning the release process.

## uploading to PyPI

You'll need `twine` installed for this step:
```bash
$ pip install twine
```

To make sure that everything is working properly, first upload to `test.pypi.org`:
```bash
$ python -m twine upload --repository-url=test.pypi.org/legacy/ dist/*
```

Then create a new environment and download the release from the test servers:
```bash
$ python -m pip install --extra-index-url test.pypi.org/simple/ napari=X.Y.Z
```

Try running some examples and tests to verify that everything is working properly.
If these fail, delete the tag with:
```bash
$ git tag -d vX.Y.Z
```

Make your changes and repeat the process from the beginning but with a slightly different tag (e.g. `vX.Y.Z.0`). Once the release passes, remember to reset the tag to the original `vX.Y.Z`.

Then, you may complete the release with:
```bash
$ python -m twine upload dist/*
```

Don't forget to push the new tag to the repo!
```bash
$ git push upstream --tags
```
