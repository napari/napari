#!/usr/bin/env bash

LAST_TAG="v0.4.17"
NEW_RELEASE="0.4.18"
FUTURE_RELEASE="0.5.0"

echo "## :wrench: Issue to trace v${NEW_RELEASE} release process"
echo

python list_opened_pr.py $NEW_RELEASE

echo

python filter_pr_that_may_be_selected.py $LAST_TAG main --milestone $NEW_RELEASE --target-branch refs/heads/v0.4.18x

echo

python filter_pr_that_may_be_selected.py $LAST_TAG main

echo

python filter_pr_that_may_be_selected.py $LAST_TAG main --milestone $FUTURE_RELEASE --skip-triaged --label bugfix

echo

python filter_opened_bug_issues.py $LAST_TAG main --milestone $NEW_RELEASE

echo

python filter_opened_bug_issues.py $LAST_TAG main --skip-triaged  --milestone none

echo
echo "The content of this issue will be updated to reflect current state."