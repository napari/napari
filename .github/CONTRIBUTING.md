# Contributing to GitHub workflows and actions

*Created: 2024-11-11; Updated:*

See the napari website for more detailed contributor information:
- [deployment](https://napari.org/stable/developers/contributing/documentation/docs_deployment.html)
- [contributing guide](https://napari.org/stable/developers/contributing/index.html)
- [core developer guide](https://napari.org/stable/developers/coredev/core_dev_guide.html)

## Workflows and actions

There are over 20 GitHub workflows found in `.github/workflows`.
The team creates a workflow to automate manual actions and steps.
This results in improved accuracy and quality. Some key workflows:
- `actionlint.yml` does static testing of GitHub action workflows
- benchmarks
- `reusable_run_tox_test.yml` uses our constraint files to install the
  compatible dependencies for each test environment which may differ
  by OS and qt versions. It is called from `test_pull_request.yml` and `test_comprehensive.yml`, not directly. 
- `upgrade_test_constraints.yml` automates upgrading dependencies for
  our test environments. It also has extensive commenting on what the
  upgrade process entails.

If adding a workflow, please take a moment to explain its purpose at the
top of its file.

## Templates

Used to provide a consistent user experience when submitting an issue or PR.
napari uses the following:
- `PULL_REQUEST_TEMPLATE.md`
- `ISSUE_TEMPLATE` directory containing:
   - `config.yml` to add the menu selector when "New Issue" button is pressed
   - `design_related.md`
   - `documentation.md`
   - `feature_request.md`
   - `bug_report.yml` config file to provide text areas for users to complete for bug reports.
- `FUNDING.yml`: redirect GitHub to napari NumFOCUS account
- Testing and bots
   - `missing_translations.md`: used if an action detects a missing language translation
   - `dependabot.yml`: opens a PR to notify maintainers of updates to dependencies
   - `labeler.yml` is a labels config file for labeler action
   - `BOT_REPO_UPDATE_FAIL_TEMPLATE.md` is an bot failure notification template
   - `TEST_FAIL_TEMPLATE.md` is a test failure notification template

## CODEOWNERS

This `CODEOWNERS` file identifies which individuals are notified if a
particular file or directory is found in a PR. Core team members can
update if desired.
