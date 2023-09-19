"""
Edit pull request description to remove HTML comments

We might want to remove section with markdown task lists that are completely empty
"""

import re
import sys
from os import environ

import requests

REPO = 'napari/napari'


def remove_html_comments(text):
    # Regular expression to remove HTML comments
    # [^\S\r\n] is whitespace but not new line
    html_comment_pattern = r"[^\S\r\n]*<!--(.*?)-->[^\S\r\n]*\s*"
    return re.sub(html_comment_pattern, "\n", text, flags=re.DOTALL)


def edit_pull_request_description(repo, pull_request_number, access_token):
    # GitHub API base URL
    base_url = "https://api.github.com"

    # Prepare the headers with the access token
    headers = {"Authorization": f"token {access_token}"}

    # Get the current pull request description
    pr_url = f"{base_url}/repos/{repo}/pulls/{pull_request_number}"
    response = requests.get(pr_url, headers=headers)
    response.raise_for_status()
    response_json = response.json()
    current_description = response_json["body"]

    # Remove HTML comments from the description
    edited_description = remove_html_comments(current_description)

    if edited_description == current_description:
        print("No HTML comments found in the pull request description")
        return

    # Update the pull request description
    update_pr_url = f"{base_url}/repos/{repo}/pulls/{pull_request_number}"
    payload = {"body": edited_description}
    response = requests.patch(update_pr_url, json=payload, headers=headers)
    response.raise_for_status()

    if response.status_code == 200:
        print(
            f"Pull request #{pull_request_number} description has been updated successfully!"
        )
    else:
        print(
            f"Failed to update pull request description. Status code: {response.status_code}"
        )


if __name__ == "__main__":
    print('Will inspect PR description to remove html comments.')

    # note that the env between pull_request and pull_request_target are different
    # and the github documentation is incorrect (or at least misleading)
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

    # get current PR number from github actions
    number = environ.get("GH_PR_NUMBER")
    print(f'Current PR number is {number}')

    access_token = environ.get("GH_TOKEN")
    if access_token is None:
        print("No access token found in the environment variables")
        # we still don't want fail status
        sys.exit(0)
    edit_pull_request_description(slug, number, access_token)
