from typing import Any

import gdown
from pooch.downloaders import (
    DOIDownloader,
    FTPDownloader,
    HTTPDownloader,
    SFTPDownloader,
    choose_downloader,
)

DOI_TO_FILE_GOOGLE_ID = {
    "doi:10.5281/zenodo.17668709/ddic.zip": "1-AAkBUXykxXG3Ve20jPk43co-m3f9ZWB",
    "doi:10.5281/zenodo.17668709/grey.zip": "1UZEoOtHMMeJMXVoWr6IZraZnvQe8ExNJ",
    'doi:10.5281/zenodo.13380203/PocilloporaDamicornisSkin.obj': "1yuPHWlLzowlfWzVMUg-mvAEe_Tmvpzy4",
    'doi:10.5281/zenodo.13380203/PocilloporaDamicornisSkin_Texture_0.jpg': "17tG44rMPWjAIoO_AlH9BaQkPY7GxxEN9",
    'doi:10.5281/zenodo.13380203/PocilloporaDamicornisSkin_GeneratedMat2.png': "1l_hGxDg6JARAyFMgWXuoIZs49qKXBv01",

}

GODGLE_PREFIX="https://drive.google.com/open?export=download&id="


class BackupDownloader(HTTPDownloader):
    def __call__(self, url: str, output_file: str, pooch:Any, check_only:bool=False) -> bool | None:
        resource_id = DOI_TO_FILE_GOOGLE_ID[url]
        if check_only:
            download_url = GODGLE_PREFIX + resource_id
            return super().__call__(download_url, output_file, pooch, check_only=check_only)
        gdown.download(id=resource_id, output=output_file, quiet=False)
        return None



def napari_choose_downloader(url: str, progressbar: bool=False) -> HTTPDownloader | SFTPDownloader | FTPDownloader | DOIDownloader:
    if url in DOI_TO_FILE_GOOGLE_ID:
        return BackupDownloader(progressbar=progressbar)
    return choose_downloader(url)
