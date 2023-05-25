import os
import subprocess
import sys

from napari._tests.utils import skip_on_win_ci

CREATE_VIEWER_SCRIPT = """
import numpy as np
import napari
v = napari.view_image(np.random.rand(512, 512))
"""


@skip_on_win_ci
def test_octree_import():
    """Test we can create a viewer with NAPARI_OCTREE."""

    cmd = [sys.executable, '-c', CREATE_VIEWER_SCRIPT]

    env = os.environ.copy()
    env['NAPARI_OCTREE'] = '1'
    env['NAPARI_CONFIG'] = ''  # don't try to save config
    subprocess.run(cmd, check=True, env=env)
