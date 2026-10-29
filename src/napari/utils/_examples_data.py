import re
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import gdown
from pooch import get_logger
from pooch.downloaders import (
    DOIDownloader,
    FTPDownloader,
    HTTPDownloader,
    SFTPDownloader,
    choose_downloader,
)

# To add a new example file to this dictionary, add it to our napari.org google drive,
# with public view access and copy its file ID from the share link.
# The file ID is the section after /d/ and before /view
DOI_TO_FILE_GOOGLE_ID = {
    # used for "3D_vectors_through_time.py" example
    "doi:10.5281/zenodo.17668709/ddic.zip": "1-AAkBUXykxXG3Ve20jPk43co-m3f9ZWB",
    "doi:10.5281/zenodo.17668709/grey.zip": "1UZEoOtHMMeJMXVoWr6IZraZnvQe8ExNJ",
    # used for "surface_multi_texture.py" example
    'doi:10.5281/zenodo.13380203/PocilloporaDamicornisSkin.obj': "1yuPHWlLzowlfWzVMUg-mvAEe_Tmvpzy4",
    'doi:10.5281/zenodo.13380203/PocilloporaDamicornisSkin_Texture_0.jpg': "17tG44rMPWjAIoO_AlH9BaQkPY7GxxEN9",
    'doi:10.5281/zenodo.13380203/PocilloporaDamicornisSkin_GeneratedMat2.png': "1l_hGxDg6JARAyFMgWXuoIZs49qKXBv01",
    # used for "cell_tracking.py" example
    "doi:10.5281/zenodo.15852284/masks_pred.npz": "1tuHI_io67gVdFZBA5E97fXrAWaxI2cqi",
    "doi:10.5281/zenodo.17643282/01(1).zip": "1qhAbIZiEKfo_a38YFtw7KJ3JjEhicDeJ"
}

GOOGLE_DRIVE_PREFIX="https://drive.google.com/open?export=download&id="

SKIMAGE_MIRROR_FILES = {
    'cells3d.tif': 'cells3d.tif',
    'lily-of-the-valley-fluorescence.tif': 'lily.tif',
    'pivchallenge/B/B001_1.tif': 'pivchallenge-B-B001_1.tif',
    'pivchallenge/B/B001_2.tif': 'pivchallenge-B-B001_2.tif',
}

def skimage_mirror_url(url: str) -> str | None:
    match = re.fullmatch(
        r'https://gitlab\.com/scikit-image/data/-/raw/[^/]+/(.+)',
        url,
    )
    if match and (filename := SKIMAGE_MIRROR_FILES.get(match[1])):
        return f'https://data.napari.dev/examples/{filename}'
    return None


class SkimageMirrorDownloader(HTTPDownloader):
    def __call__(
        self,
        url: str,
        output_file: Any,
        pooch: Any,
        check_only: bool = False,
    ) -> bool | None:
        mirror_url = skimage_mirror_url(url)
        if mirror_url is None:
            raise ValueError(f'No scikit-image mirror configured for {url}')

        get_logger().info('Redirecting %s to %s', url, mirror_url)
        res =  super().__call__(
            mirror_url, output_file, pooch, check_only=check_only
        )
        get_logger().info('Finished mirror download: %s', mirror_url)
        return res


class BackupDownloader(HTTPDownloader):
    def __call__(self, url: str, output_file: str, pooch:Any, check_only:bool=False) -> bool | None:
        resource_id = DOI_TO_FILE_GOOGLE_ID[url]
        if check_only:
            download_url = GOOGLE_DRIVE_PREFIX + resource_id
            return super().__call__(download_url, output_file, pooch, check_only=check_only)
        gdown.download(id=resource_id, output=output_file, quiet=False)
        return None



def napari_choose_downloader(url: str, progressbar: bool=False) -> HTTPDownloader | SFTPDownloader | FTPDownloader | DOIDownloader:
    if url in DOI_TO_FILE_GOOGLE_ID:
        return BackupDownloader(progressbar=progressbar)
    if skimage_mirror_url(url) is not None:
        return SkimageMirrorDownloader(progressbar=progressbar)
    return choose_downloader(url)


P = ParamSpec('P')
R = TypeVar('R')


def nilearn_download_factory(
    fetch_surf_nki_enhanced: Callable[P, R],
) -> Callable[P, R]:
    def nilearn_change_url(*args: P.args, **kwargs: P.kwargs) -> R:
        if kwargs.get('n_subjects') == 1:
            # we kee copy of only data fetched with n_subjects=1,
            # so only for ths case we can use our mirror
            kwargs.setdefault('url', 'https://data.napari.dev/examples/')

        return fetch_surf_nki_enhanced(*args, **kwargs)

    return nilearn_change_url
