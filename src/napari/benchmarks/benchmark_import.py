import subprocess
import sys


class ImportTimeSuite:
    def time_import(self):
        cmd = [sys.executable, '-c', 'import napari']
        subprocess.run(cmd, stderr=subprocess.PIPE)


class FirstLayerTimeSuite:
    def time_create_unitless_image(self):
        cmd = [
            sys.executable,
            '-c',
            (
                'import numpy as np; from napari.layers import Image; '
                'Image(np.zeros((512, 512)))'
            ),
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)

    def time_create_unitless_labels(self):
        cmd = [
            sys.executable,
            '-c',
            (
                'import numpy as np; from napari.layers import Labels; '
                'Labels(np.zeros((512, 512), dtype=np.uint8))'
            ),
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)

    def time_create_image_with_units(self):
        cmd = [
            sys.executable,
            '-c',
            (
                'import numpy as np; from napari.layers import Image; '
                "Image(np.zeros((512, 512)), units=('um', 'nm'))"
            ),
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)


if __name__ == '__main__':
    from utils import run_benchmark

    run_benchmark()
