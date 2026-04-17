# /// script
# dependencies = [
#   "imageio",
#   "tifffile",
# ]
# ///

import re
import sys
from pathlib import Path
from pprint import pformat

import imageio
import tifffile

BUILTINS_YAML_PATH = (
    Path(__file__).parent.parent / 'src' / 'napari_builtins' / 'builtins.yaml'
)

re_exts = re.compile(
    r'(filename_patterns:\s+\[\n)([^]]*)(\n\s+\])', flags=re.DOTALL
)


def gen_extensions():
    imageio_exts = {
        f'*{ext}' for fmt in imageio.formats for ext in fmt.extensions
    }
    imageio_video_exts = {
        f'*{fmt.extension}'
        for fmt in imageio.config.video_extensions
        if any(name == 'FFMPEG' for name in fmt.priority)
    }
    tifffile_exts = {f'*.{ext}' for ext in tifffile.TIFF.FILE_EXTENSIONS}
    napari_exts = {f'*.{ext}' for ext in ('npy', 'zarr', 'csv')}

    exts = sorted(
        imageio_exts | imageio_video_exts | tifffile_exts | napari_exts
    )

    json = pformat(exts, compact=True, indent=10)[:-1].replace('[', ' ')
    return json


def get_extensions():
    with open(BUILTINS_YAML_PATH) as f:
        contents = f.read()

    match = re.search(re_exts, contents)
    return match.group(2)


def replace_extensions(new):
    with open(BUILTINS_YAML_PATH) as f:
        contents = f.read()

    with open(BUILTINS_YAML_PATH, 'w') as f:
        f.write(re.sub(re_exts, rf'\1{new}\3', contents))


if __name__ == '__main__':
    print('Checking reader extensions...')

    CI = '--ci' in sys.argv
    old = get_extensions()
    new = gen_extensions()
    if old != new:
        print('Extension patterns are not up to date!')
        if CI:
            print('Updating extensions...')
            replace_extensions(new)
            print('Extensions updated.')
            sys.exit(0)
        else:
            print('Here is the new extension block:\n')
            print(new)
            sys.exit(1)

    print('Extension patterns are up to date.')
