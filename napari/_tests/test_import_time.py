import subprocess
import sys


def test_import_time(tmp_path):
    cmd = [sys.executable, '-X', 'importtime', '-c', 'import napari']
    proc = subprocess.run(cmd, capture_output=True, check=True)
    log = proc.stderr.decode()
    last_line = log.splitlines()[-1]
    time, name = [i.strip() for i in last_line.split("|")[-2:]]

    # we may need to change this threshold
    assert name == 'napari'
    assert int(time) < 1_000_000, "napari import taking longer than 1 sec!"
    print(f"\nnapari took {int(time)/1e6:0.3f} seconds to import")

    # common culprit of slow imports
    assert 'pkg_resources' not in log
