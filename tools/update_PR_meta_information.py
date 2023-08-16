"""
Edit pull request description to remove HTML comments,
Add label based on the presence of task lists in the description
Add milestone based on metadata.

We might want to remove section with markdown task lists that are completely empty
"""

import re
import sys
from functools import lru_cache
from os import environ

import requests

REPO = 'napari/napari'
# GitHub API base URL
BASE_URL = "https://api.github.com"


def remove_html_comments(text):
    # Regular expression to remove HTML comments
    # [^\S\r\n] is whitespace but not new line
    html_comment_pattern = r"[^\S\r\n]*<!--(.*?)-->[^\S\r\n]*\s*"
    return re.sub(html_comment_pattern, "\n", text, flags=re.DOTALL)


@lru_cache
def get_description_from_pr(repo, pull_request_number, access_token):
    # Prepare the headers with the access token
    headers = {"Authorization": f"token {access_token}"}

    # Get the current pull request description
    pr_url = f"{BASE_URL}/repos/{repo}/pulls/{pull_request_number}"
    response = requests.get(pr_url, headers=headers)
    response.raise_for_status()
    response_json = response.json()
    return response_json["body"]


def get_labels_added_to_pr(repo, pull_request_number, access_token):
    # Prepare the headers with the access token
    headers = {"Authorization": f"token {access_token}"}

    # Get the current pull request description
    pr_url = f"{BASE_URL}/repos/{repo}/issues/{pull_request_number}/labels"
    response = requests.get(pr_url, headers=headers)
    response.raise_for_status()
    response_json = response.json()
    return [x["name"] for x in response_json["body"]]


text_to_label = {
    "Bug-fix": "bugfix",
    "New feature": "feature",
    "Maintenance": "maintenance",
    "Documentation": "documentation",
    "Enhancement": "enhancement",
    "Breaking change": "breaking change",
}


def get_labels_based_on_description(description):
    res = []
    lines = description.split('\n')
    for i, line in enumerate(lines):
        if "Type of change" in line:
            lines = lines[i + 1 :]
            break
    else:
        return []

    for line in lines:
        if "[x]" in line:
            for name, label in text_to_label.items():
                if name in line:
                    res.append(label)
                    break
    return res


def add_labels_to_pr(repo, pull_request_number, access_token) -> bool:
    description = get_description_from_pr(
        repo, pull_request_number, access_token
    )
    labels = get_labels_based_on_description(description)
    if not labels:
        print("No labels information in description")
        return True
    for label in get_labels_added_to_pr(
        repo, pull_request_number, access_token
    ):
        if label in labels:
            labels.remove(label)

    if not labels:
        print("No labels to add")
        return True

    headers = {"Authorization": f"token {access_token}"}

    response = requests.patch(
        f"{BASE_URL}/repos/{repo}/issues/{pull_request_number}/labels",
        headers=headers,
        json={"labels": labels},
    )

    if response.status_code != 200:
        print(
            f"Failed to add labels to PR. Status code: {response.status_code}"
        )
        return False

    print("Labels added successfully")
    return True


def edit_pull_request_description(
    repo, pull_request_number, access_token
) -> bool:
    current_description = get_description_from_pr(
        repo, pull_request_number, access_token
    )
    # Remove HTML comments from the description
    edited_description = remove_html_comments(current_description)

    if edited_description == current_description:
        print("No HTML comments found in the pull request description")
        return True

    # Update the pull request description
    update_pr_url = f"{BASE_URL}/repos/{repo}/pulls/{pull_request_number}"
    payload = {"body": edited_description}
    headers = {"Authorization": f"token {access_token}"}

    response = requests.patch(update_pr_url, json=payload, headers=headers)
    response.raise_for_status()

    if response.status_code == 200:
        print(
            f"Pull request #{pull_request_number} description has been updated successfully!"
        )
        return True

    print(
        f"Failed to update pull request description. Status code: {response.status_code}"
    )
    return False


def main():
    print('Will inspect PR description to remove html comments.')

    # note that the env between pull_request and pull_request_target are different
    # and the GitHub documentation is incorrect (or at least misleading)
    # and likely varies between pull request intra-repository and inter-repository
    # thus we log many things to try to understand what is going on in case of failure.
    # among other:
    # - github.event.repository.name is not the full slug, but just the name
    # - github.event.repository.org is empty if the repo is a normal user.

    repository_url = environ.get("GH_REPO_URL")
    print(f'Current repository is {repository_url}')
    repository_parts = repository_url.split('/')[-2:]

    slug = '/'.join(repository_parts)
    print(f'Current slug is {slug}')
    if slug != REPO:
        print('Not on main repo, aborting with success')
        sys.exit(0)

    # get current PR number from GitHub actions
    number = environ.get("GH_PR_NUMBER")
    print(f'Current PR number is {number}')

    access_token_ = environ.get("GH_TOKEN")
    if access_token_ is None:
        print("No access token found in the environment variables")
        # we still don't want fail status
        sys.exit(0)

    if not edit_pull_request_description(slug, number, access_token_):
        sys.exit(1)
    if not add_labels_to_pr(slug, number, access_token_):
        sys.exit(1)


if __name__ == "__main__":
    main()
