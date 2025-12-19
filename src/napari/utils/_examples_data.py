from pooch.downloaders import HTTPDownloader, choose_downloader

DOI_TO_FILE_GOOGLE_ID = {
    "doi:10.5281/zenodo.17668709/ddic.zip": "1-AAkBUXykxXG3Ve20jPk43co-m3f9ZWB",
    "doi:10.5281/zenodo.17668709/grey.zip": "1G0A8gRSJpeDfbeyrOBkFriXjVSOm22H9",
    'doi:10.5281/zenodo.13380203/PocilloporaDamicornisSkin.obj': "1yuPHWlLzowlfWzVMUg-mvAEe_Tmvpzy4",
    'doi:10.5281/zenodo.13380203/PocilloporaDamicornisSkin_Texture_0.jpg': "17tG44rMPWjAIoO_AlH9BaQkPY7GxxEN9",
    'doi:10.5281/zenodo.13380203/PocilloporaDamicornisSkin_GeneratedMat2.png': "1l_hGxDg6JARAyFMgWXuoIZs49qKXBv01",

}

GODGLE_PREFIX="https://drive.google.com/uc?export=download&id="


class BackupDownloader(HTTPDownloader):
    def __call__(self, url, *args, **kwargs):
        resource_id = DOI_TO_FILE_GOOGLE_ID[url]
        super().__call__(GODGLE_PREFIX + resource_id, *args, **kwargs)


def napari_choose_downloader(url, progressbar=False):
    if url in DOI_TO_FILE_GOOGLE_ID:
        return BackupDownloader(progressbar=progressbar)
    return choose_downloader(url)
