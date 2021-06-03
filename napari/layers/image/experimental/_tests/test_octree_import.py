import subprocess
import sys


def test_octree_import():
    cmd = [sys.executable, '-c', 'import napari; v = napari.Viewer()']
    subprocess.run(cmd, check=True, env={'NAPARI_OCTREE': '1'})
