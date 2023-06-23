"""
Script to run in a daily cron in order to ping people who added the "ready for
merge" label to a PR and the PR has not been merged.

If the PR has been merged and the label is still here, remove the label.

If it has been more than 24 hours since the label was added, add a comment as
well as an `overdue-merge` label.

"""

import argparse
import logging
import sys
from datetime import datetime
from os import environ

import requests

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
# log basic conflig
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


token = environ.get("GITHUB_TOKEN")
assert token is not None
HEADERS = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def list_open_pull_with_label(owner, repo, label):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {"state": "open", "labels": label}

    response = requests.get(url, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    if response.status_code == 200:
        pulls = response.json()
        return [pull['number'] for pull in pulls]
    return None


def get_label_added_timestamp(owner, repo, pr_number, label):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/events"

    response = requests.get(url, headers=HEADERS, timeout=10)
    if response.status_code == 200:
        events = response.json()
        for event in events:
            if event['event'] == 'labeled' and event['label']['name'] == label:
                timestamp = event['created_at']
                user_added_label = event['actor']['login']
                return (timestamp, user_added_label)
    else:
        # this should not happen unless the label was removed while we run this script
        log.error("Error: %s - %s", response.status_code, response.text)
    return None, None


def treat_pr(number, label, repo, owner, dry_run):
    (ts, user) = get_label_added_timestamp(owner, repo, number, label)

    # print how long ago the label was added
    delta_t = datetime.utcnow() - datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
    log.info(
        "PR #%s was labeled %r %s s ago by %s", number, label, delta_t, user
    )
    # if more than 24 hours ago and less than 48h ago, add a comment
    # we do that to avoid adding a comment on the PR every day
    # check that pull request is still open
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    response = requests.get(url, headers=HEADERS, timeout=10)
    if response.status_code == 200:
        pull = response.json()
        if pull['state'] != 'open':
            log.info("PR #%s is not open anymore, skipping", number)
            # removing "ready for merge" label
            if not dry_run:
                url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/labels/{label}"
                response = requests.delete(url, headers=HEADERS, timeout=10)
            if response.status_code == 204:
                log.info(
                    "Successfully removed %s label from PR #%s", label, number
                )
            else:
                log.error(
                    "Error: %s - %s", response.status_code, response.text
                )
            return

        # get the last time the PR was pushed to
        last_pushed_at = pull['updated_at']
    else:
        log.error('Error: {response.status_code} - {response.text}')

    log.info(
        "Checking if PR #%s has been pushed/modified to since the label was added",
        number,
    )
    if last_pushed_at > ts:
        # this can be description being edited, or a new commit being pushed
        log.info(
            "PR #%s has been pushed to, or edited since the %r label was added, skipping",
            number,
            label,
        )
        return

    # get the combined status of the PR
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{pull['head']['sha']}/status"
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()
    status = response.json()['state']
    if status != 'success':
        log.info('PR ', number, ' has a non success status, aborting:', status)
        return
    log.info('PR #%s Checking time since label has been added', number)
    if delta_t.days > 0 and delta_t.days < 2:
        # extra checks we may wan to do later:
        # - check that the last update to the PR pre-date the "ready for merge" label

        # - check that CI is green.
        # - what should we do it there are comments after the label was added?
        # - check that there is not conflict and that it can actually be merged ?

        log.info("Commenting on PR #%s", number)
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments"
        if not dry_run:
            response = requests.post(
                url,
                headers=HEADERS,
                json={
                    "body": f"Hi @{user}, it's been more than 24 hours since you added the '{label}' label to this PR."
                    "\nIt can now be merged."
                },
                timeout=10,
            )
        log.info("adding an `overdue-merge` labelto PR #%s", number)
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/labels"
        if not dry_run:
            response = requests.post(
                url,
                headers=HEADERS,
                json=["overdue-merge"],
                timeout=10,
            )

            if response.status_code == 201:
                log.info("Successfully commented on PR #%s", number)
            else:
                log.error(
                    "Error: %s - %s",
                    response.status_code,
                    response.text,
                )
    else:
        log.info(
            "PR #%s was labeled %r %s s ago, skipping",
            number,
            label,
            delta_t,
        )


def main(argv):
    target_owner = "napari"
    target_repo = "napari"
    target_label = "ready for merge"
    pulls_number = list_open_pull_with_label(
        target_owner, target_repo, target_label
    )
    # add a --dry-run option via argparse argument parser.

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for number in pulls_number:
        treat_pr(
            number,
            label=target_label,
            repo=target_repo,
            owner=target_owner,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
