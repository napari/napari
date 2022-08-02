import logging
import os
import pathlib
import subprocess
from argparse import ArgumentParser

logging.basicConfig(
    format='%(levelname)s : %(asctime)s : %(message)s',
    level=logging.INFO,
)

parser = ArgumentParser(
    description='Run napari with one of the perfmon configurations.'
)
parser.add_argument(
    'config',
    help='The name of the sub-directory that contains the perfmon configuration file (e.g. slicing).',
)
parser.add_argument(
    'napari_args', nargs="*", help='The arguments to pass to napari.'
)
args = parser.parse_args()

logging.info(
    f'''Running napari_with_perfmon.py with the following arguments.
{args}'''
)

perfmon_dir = pathlib.Path(__file__).parent.parent.resolve(strict=True)
config_path = str(perfmon_dir / args.config / 'config.json')

env = os.environ.copy()
env['NAPARI_PERFMON'] = config_path

subprocess.check_call(
    'napari ' + ' '.join(args.napari_args),
    env=env,
    shell=True,
)
