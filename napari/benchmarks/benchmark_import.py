import subprocess
import sys


class ImportTimeSuite:
    def time_import(self):
        cmd = [sys.executable, '-c', 'import napari']
        subprocess.run(cmd, stderr=subprocess.PIPE)
