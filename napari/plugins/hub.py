"""
These convenience functions will be useful for searching napari hub for
retriving plugin information and related metadata.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Generator, Optional
from urllib import error, request

from .utils import normalized_name, ProjectInfo

NAPARI_HUB_PLUGINS = 'https://api.napari-hub.org/plugins'
ANACONDA_ORG = 'https://api.anaconda.org/package/{channel}/{package_name}'


@lru_cache(maxsize=128)
def hub_plugin_info(
    name: str, min_dev_status=3, conda_forge=False,
) -> Optional[ProjectInfo]:
    """Get package metadat from the napari hub.

    Parameters
    ----------
    name : str
        name of the package
    min_dev_statur : int, optional
        Development status. Default is 3.
    conda_forge: bool, optional
        Check if package is available in conda-forge. Default is False.

    Returns
    -------
    ProjectInfo or None
        Project metadata or None.
    """
    try:
        with request.urlopen(NAPARI_HUB_PLUGINS + "/" + name) as resp:
            info = json.loads(resp.read().decode())
    except error.HTTPError:
        return None

    version = info["version"]
    norm_name = normalized_name(info["name"])
    if conda_forge:
        anaconda_api = ANACONDA_ORG.format(channel="conda-forge", package_name=norm_name)
        try:
            with request.urlopen(anaconda_api) as resp_api:
                anaconda_info = json.loads(resp_api.read().decode())
                if version not in anaconda_info.get("versions", []):
                    return None

        except error.HTTPError:
            return None

    classifiers = info.get("development_status", [])
    for _ in range(1, min_dev_status):
        if any(f'Development Status :: {1}' in x for x in classifiers):
            return None

    authors = ", ".join([author["name"] for author in info["authors"]])
    return ProjectInfo(
        name=norm_name,
        version=version,
        url=info["project_site"],
        summary=info["summary"],
        author=authors,
        license=info["license"] or "UNKNOWN",
    )


def iter_napari_plugin_info(
    skip={}, conda_forge=False
) -> Generator[ProjectInfo, None, None]:
    """Return a generator that yields ProjectInfo of available napari plugins.
    """
    with request.urlopen(NAPARI_HUB_PLUGINS) as resp:
        plugins = json.loads(resp.read().decode())

    already_yielded = set()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(hub_plugin_info, name, conda_forge=conda_forge)
            for name in sorted(plugins) if name not in skip
        ]

        for future in as_completed(futures):
            info = future.result()
            if info and info not in already_yielded:
                already_yielded.add(info)
                yield info
