"""
Edit pull request description to remove HTML comments

We might want to remove section with markdown task lists that are completely empty
"""

import re
import sys
from os import environ

import requests


def remove_html_comments(text):
    # Regular expression to remove HTML comments
    # [^\S\r\n] is whitespace but not new line
    html_comment_pattern = r"[^\S\r\n]*<!--(.*?)-->[^\S\r\n]*\n?"
    return re.sub(html_comment_pattern, "", text, flags=re.DOTALL)


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
    # Replace with your repository and pull request number
    # get cuurrent repository name from github actions
    repository_name = environ.get("GITHUB_REPOSITORY")
    if repository_name == "napari/napari":
        sys.exit(0)

    # get current PR number from github actions
    github_ref = environ.get("GITHUB_REF")
    refs, pull, number, merge = github_ref.split('/')
    assert refs == 'refs'
    assert pull == 'pull'
    assert merge == 'merge'

    # Replace with your GitHub access token
    access_token = environ.get("GITHUB_TOKEN")

    edit_pull_request_description(repository_name, number, access_token)
