---
name: "✅ Release Checklist"
about: Checklist for managing a napari release. [For project maintainers only]
title: 'vX.Y.Z napari release checklist'
assignees: ''
---

**Release**: [vX.Y.Z](https://github.com/napari/napari/milestones/?)
**Stable Due Date**: 20YY-MM-DD

## Priority PRs/Issues

*Link to priority PRs/issues that should be completed prior to release.*

## Early management

- [ ] Make a new Zulip thread in the [release channel](https://napari.zulipchat.com/#narrow/stream/215289-release)
- [ ] Create milestone in `napari/napari` and `napari/docs` for the new release.
  - [ ] Set date of milestone to the expected release date for rc0, at least 3 days before the actual release date. Given a typical release cycle, a week prior is recommended.
    - *Note the Milestone date is on [Line Island Time (LINT)](https://www.timeanddate.com/time/zone/@4030926) where the village of napari is located.*
  - [ ] Add the Zulip release thread to the milestone description

## Triage the Milestone

- [ ] Triage the milestone throughout the release cycle.
  - [ ] Add PRs of interest to highlights. These help the manager keep track of important PRs that often require extra team attention.
  - [ ] Move PRs and issues that won't make it in the current release to the next milestone, or remove the milestone altogether. Add a note to PRs (and issues) from community members describing the actions the release manager is taking and provide actionable comments.

## Create Pre-release

*You may wish to make an alpha release at any point in the release cycle. However, you must make at least an empty release file to attach to the tag and merge it to `napari/docs`, otherwise the release will fail.*

- [ ] Communicate in Zulip release thread that pre-release is approaching.
- [ ] Create a draft of release notes and make PR to `napari/docs`.
  - [ ] Merge release notes to `napari/docs`
- [ ] Tag pre-release with release notes file and push to `napari/napari`. [See docs for details.](https://napari.org/dev/developers/coredev/release.html#tagging-the-new-release-candidate)
- [ ] Announce release candidate on Zulip in Release (and General, for meso or larger releases) and [forum.image.sc](https://forum.image.sc/announcements)
- [ ] Check for proper deployment (~1 hour after tagging)
  - [ ] [PyPI](https://pypi.org/project/napari/#history)
  - [ ] [napari.org](https://napari.org/dev/)

## Stable Release Prep

- [ ] Make a PR requesting any new contributors add their information to [CITATION.cff](https://github.com/napari/napari/blob/main/CITATION.cff) ([PR example](https://github.com/napari/napari/pull/8138)).
- [ ] Triage remaining PRs and Issues. At this stage, bug fixes and remaining (testable) features are prioritized.
- [ ] Make new PR to `napari/docs` with changes to release notes.
- [ ] Ensure releases are cut from other napari repos if needed. (e.g. `napari-plugin-manager`, triangulation libraries)
- [ ] Ensure [`conda-recipe/recipe.yaml`](https://github.com/napari/packaging/blob/main/conda-recipe/recipe.yaml) in `napari/packaging` is up-to-date (e.g. `run` dependencies match `pyproject.toml` requirements).
- [ ] Ensure that [`constraints`](https://github.com/napari/napari/tree/main/resources/constraints) files are up to date. Usually initiated by `@napari-bot` within a day of changes to dependencies and otherwise regularly scheduled.

## Create Stable Release

- [ ] Do a final build, push, and merge of release notes to `napari/docs`.
- [ ] Ensure `napari/docs` successfully deploys.
- [ ] Checkout `napari/napari:main` and tag with release notes file. [See docs for details.](https://napari.org/dev/developers/coredev/release.html#tagging-the-new-release-candidate)
- [ ] Push created tag to `napari/napari`. This triggers the deployment actions.
- [ ] Check that docs deployment is successful on `napari/docs` and `napari/napari.github.io`
- [ ] Ensure [PyPI](https://pypi.org/project/napari/#history) release is out
- [ ] Conda-forge [`napari-feedstock`](https://github.com/conda-forge/napari-feedstock). **[Bot is automatic, but may be manually triggered](https://napari.org/dev/developers/coredev/release.html#new-releases)**
- [ ] Ensure [conda-forge release](https://anaconda.org/conda-forge/napari) is out (should be minutes after merging the feedstock PR created above).
- [ ] Update symlink in `napari/napari.github.io` by manually triggering the [action](https://github.com/napari/napari.github.io/actions/workflows/symlink-stable.yml) with format "`X.Y.Z`" (no `v` prefix)
- [ ] Update the `version_switcher.json` in `napari/docs`. (e.g. [this PR](https://github.com/napari/docs/pull/826))

### Announce new release

- [ ] Announce in General on Zulip
- [ ] Edit the pre-release [image.sc](https://forum.image.sc) thread by updating the title and original post description to be the stable instructions, add final highlights, and post a reply to bump the thread.
- [ ] Post on [bluesky](https://bsky.app/profile/napari.org), [fosstodon](https://fosstodon.org/@napari), and [LinkedIn](https://www.linkedin.com/company/napari)
