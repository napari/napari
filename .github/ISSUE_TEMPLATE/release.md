---
name: "✅ Release Checklist"
about: Checklist for managing a napari release. [For project maintainers only]
title: 'vX.Y.Z napari release checklist'
assignees: ''
---

**Release**: [vX.Y.Z](https://github.com/napari/napari/milestones/?)
**Stable Due Date**: 20YY-MM-DD

## Early management

- [ ] Make a new Zulip thread in the [release channel](https://napari.zulipchat.com/#narrow/stream/215289-release)
- [ ] Create milestone in `napari/napari` and `napari/docs` for the new release and uppdate the links above to point to it.
- [ ] Set date of milestone to the expected release date (~21th).
- *Note the Milestone date is on [Line Island Time (LINT)](https://www.timeanddate.com/time/zone/@4030926) where the village of napari is located.*
- [ ] Add the Zulip release thread to the milestone description

## Triage PRs and issues

- Triage PRs and issues throughout the release cycle, prioritizing ones that were tagged with the current milestone.
- Add the highlight tag to PRs that should be highlighted. When you do, make sure to add in a comment the text for the highlight and a screenshot where applicable. If someone else added the tag, ensure they take responsibility for the highlight info.
- Move PRs and issues that won't make it in the current release to the next milestone, or remove the milestone altogether. Add a note to PRs (and issues) from community members describing the actions the release manager is taking and provide actionable comments.
- Ensure releases are cut from other napari repos if needed. (e.g. `napari-plugin-manager`, `npe2`, triangulation libraries)

## Create alpha release

This is just to test that everything works. It should happen before the first rc, and it has no bearing on feature/bugfix merging state.

- [ ] Create a draft of the release notes in `napari/docs` by [following the guide](https://napari.org/stable/developers/coredev/release.html#step-2-generating-release-notes) and merge the PR. No highlights or custom text are needed at this point.
- [ ] Create a tag in `napari/napari` with `a0` suffix (e.g: `v0.10.0a0`) with the release notes file attached to the tag and push to `napari/napari`. [See docs for details.](https://napari.org/dev/developers/coredev/release.html#tagging-the-new-release-candidate)
- [ ] Check for proper deployment (~1 hour after tagging)
  - [ ] [PyPI](https://pypi.org/project/napari/#history)
  - [ ] [napari.org](https://napari.org/dev/)

## Create the first Release Candidate

At around the 15th of the month, it's time to lock in the features and focus on bug fixing and finishing touches. After the first rc, only trivial PRs, maintenance, and bugfixes should be merged.

Release the first release candidate by following the same steps as above, but first:

- [ ] Communicate in Zulip release thread that pre-release is approaching.
- [ ] Announce release candidate on Zulip in Release (and General, for meso or larger releases) and [forum.image.sc](https://forum.image.sc/announcements)
- [ ] Add highlight text and screenshots to the draft release notes.
- [ ] tag with the `rcX` suffix, where `X` starts from 0 and increases every time a new release candidate needs to be made to include bug fixes.

## Stable Release Prep

Aim to release the full release around the 21st. The steps are again like the above, but this time:

- [ ] Make new PR to `napari/docs` with full changes to release notes; ensure all `highlight`s are included in the release notes text.
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
