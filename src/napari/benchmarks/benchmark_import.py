import subprocess
import sys


class ImportTimeSuite:
    def time_import(self):
        cmd = [sys.executable, '-c', 'import napari']
        subprocess.run(cmd, stderr=subprocess.PIPE)


if __name__ == '__main__':
    from utils import run_benchmark

    run_benchmark()
