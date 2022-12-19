"""For testing the URLs in the Help menu"""

import pytest
import requests

from napari._app_model.actions._help_actions import HELP_URLS


@pytest.mark.parametrize('url', HELP_URLS.keys())
def test_help_urls(url):
    if url == 'release_notes':
        pytest.skip("No release notes for dev version")

    r = requests.head(HELP_URLS[url])
    r.raise_for_status()
