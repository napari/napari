"""
These convenience functions will be useful for searching napari hub for
retriving plugin information and related metadata.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Generator, Optional, Tuple
from urllib import error, request

from npe2 import PackageMetadata

from .utils import normalized_name

NAPARI_HUB_PLUGINS = 'https://api.napari-hub.org/plugins'
ANACONDA_ORG = 'https://api.anaconda.org/package/{channel}/{package_name}'


@lru_cache(maxsize=1024)
def hub_plugin_info(
    name: str,
    min_dev_status=3,
    conda_forge=True,
) -> Tuple[Optional[PackageMetadata], bool]:
    """Get package metadata from the napari hub.

    Parameters
    ----------
    name : str
        name of the package
    min_dev_statur : int, optional
        Development status. Default is 3.
    conda_forge: bool, optional
        Check if package is available in conda-forge. Default is True.

    Returns
    -------
    Tuple of optional PackageMetadata and bool
        Project PackageMetadata and availability on conda forge.
    """
    try:
        with request.urlopen(NAPARI_HUB_PLUGINS + "/" + name) as resp:
            info = json.loads(resp.read().decode())
    except error.HTTPError:
        return None, False

    version = info["version"]
    norm_name = normalized_name(info["name"])
    is_available_in_conda_forge = True
    if conda_forge:
        is_available_in_conda_forge = False
        anaconda_api = ANACONDA_ORG.format(
            channel="conda-forge", package_name=norm_name
        )
        try:
            with request.urlopen(anaconda_api) as resp_api:
                anaconda_info = json.loads(resp_api.read().decode())
                versions = anaconda_info.get("versions", [])
                if versions:
                    if version not in versions:
                        version = versions[-1]

                    is_available_in_conda_forge = True
        except error.HTTPError:
            pass

    classifiers = info.get("development_status", [])
    for _ in range(1, min_dev_status):
        if any(f'Development Status :: {1}' in x for x in classifiers):
            return None, False

    authors = ", ".join([author["name"] for author in info["authors"]])
    data = PackageMetadata(
        metadata_version="1.0",
        name=norm_name,
        version=version,
        summary=info["summary"],
        home_page=info["project_site"],
        author=authors,
        license=info["license"] or "UNKNOWN",
    )
    return data, is_available_in_conda_forge


def iter_hub_plugin_info(
    skip={}, conda_forge=True
) -> Generator[Tuple[Optional[PackageMetadata], bool], None, None]:
    """Return a generator that yields ProjectInfo of available napari plugins."""
    with request.urlopen(NAPARI_HUB_PLUGINS) as resp:
        plugins = json.loads(resp.read().decode())

    already_yielded = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(hub_plugin_info, name, conda_forge=conda_forge)
            for name in sorted(plugins)
            if name not in skip
        ]

        for future in as_completed(futures):
            info, is_available_in_conda_forge = future.result()
            if info and info not in already_yielded:
                already_yielded.append(info)
                yield info, is_available_in_conda_forge
