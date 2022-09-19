# Release guide

This guide documents `napari`'s release process.
Currently, it only handles distribution, but as the project matures,
it will include generating release notes, documentation, etc.

# Timeline
New versions of `napari` will be released every two months. The first release candidate will be available one week prior to release for testing purposes. Multiple release candidates may become available during the week prior to release. Upcoming releases can be found in our public calendar.

The latest release candidate can be installed with 

`pip install --pre napari`

# Release management
The release will be coordinated by a release manager whose responsibilities include...

## Two weeks before release (one week before release candidate)
- Look through currently open PRs and get a sense of what would be good to merge before the first release candidate 
- Create a zulip thread in the release channel letting people know the release candidate is coming and pointing out PRs that would be nice to merge before release

At this stage, bug fixes and features that are close to landing should be prioritized. The release manager will follow up with PR authors, reviewing and merging as needed.
  
## Nine days before release (two days before release candidate)
- Generate release notes with the script in the release folder
- Fill in the release highlights and make a PR with the release notes

At this point the release manager should ideally be the only person merging PRs on the repo for the next week.

## One week before release 
- Add any recently merged PRs to release notes
- Merge release notes
- Make the release candidate
- Announce to release stream on zulip that the first release candidate is available for testing

## The week before release
- Merge any PRs and update release notes accordingly
- Make new release candidates as necessary and announce them on zulip

At this stage PRs merged should focus mainly on regressions and bug fixes. New features should wait until after release.

## The day of release
- make sure final rc has been tested
- ensure all PRs have been added to release notes and then make release and announce on zulip

# Release process

Additional `release` dependencies (`pip install -e .[release]`) are required to complete the release process.

> [`MANIFEST.in`](https://github.com/napari/napari/blob/main/MANIFEST.in) determines which non-Python files are included.
> Make sure to check that all necessary ones are listed before beginning the release process.

The `napari/napari` repository must have a PyPI API token as a GitHub secret.
This likely has been done already, but if it has not, follow
[this guide](https://pypi.org/help/#apitoken) to gain a token and
[this guide](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
to add it as a secret.

## Determining the version

The version of `napari` is automatically determined at install time by
[`setuptools_scm`](https://github.com/pypa/setuptools_scm) from the latest
[`git` tag](https://git-scm.com/book/en/v2/Git-Basics-Tagging) beginning with
`v`. Thus, you'll need to tag the
[reference](https://git-scm.com/book/en/v2/Git-Internals-Git-References) with
the new version number. It is likely something like `X.Y.Z`. Before making a
release though we need to generate the release notes.

## Generating release notes

1. Make a list of merges, contributors, and reviewers by running
   ``python docs/release/generate_release_notes.py -h`` and following that file's usage.
   For each release generate the list to include everything since the last release for which there
   are release notes (which should just be the last release). For example making the release notes
   for the `0.2.1` release can be done as follows:

   ```bash
   python docs/release/generate_release_notes.py v0.2.0 main --version 0.2.1 | tee docs/release/release_0_2_1.md
   ```

2. Scan the PR titles for highlights, deprecations, API changes,
   and bugfixes, and mention these in the relevant sections of the notes.
   Try to present the information in an expressive way by mentioning
   the affected functions, elaborating on the changes and their
   consequences. If possible, organize semantically close PRs in groups.

3. Make sure the file name is of the form ``doc/release/release_<major>_<minor>_<release>.md``.

4. Make and merge a PR with these release notes before moving onto the next steps.

## Update translation strings

As new code is included in the codebase, some of the strings that need to be translated might
not yet be using the `trans` methods. To help keep the codebase up to date in terms
of translations we added a test script that
[runs daily on CI](https://github.com/napari/napari/actions/workflows/test_translations.yml)
and can be also run locally to ensure that a release includes the most up to date translatable
strings.

The test script is available on the `/tools/test_strings.py` file and it relies on an additional
file `/tools/strings_list.py` to include strings to skip safely from translation.

The test checks:

  1. **Untranslated strings**: not using the `trans` methods.
  2. **Outdated skip strings**: should no longer be included in the `/tools/strings_list.py` file.
  3. **Translation usage errors**: where translation strings may be missing interpolation variables.

You can execute tests locally from the repository root, and follow the instructions printed
on the `stdout` if any test fails.

  ```bash
  pytest tools/ --tb=short
  ```

## Tagging the new release candidate

First we will generate a release candidate, which will contain the letters `rc`.
Using release candidates allows us to test releases on PyPI without using up the actual
release number.

You can tag the current source code as a release candidate with:

```bash
git tag vX.Y.Zrc1 main
```

If the tag is meant for a previous version of main, simply reference the specific commit:

```bash
git tag vX.Y.Zrc1 abcde42
```

Note here how we are using `rc` for release candidate to create a version of our release we can test
before making the real release.

You can read more on tagging [here](https://git-scm.com/book/en/v2/Git-Basics-Tagging).

## Testing the release candidate

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

## Generating the actual release

To generate the actual release you will now repeat the processes above but now dropping the `rc`.
For example:

```bash
git tag vX.Y.Z main
git push upstream --tags
```
