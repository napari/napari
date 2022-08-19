import json
import logging
import pathlib
from argparse import ArgumentParser

import matplotlib.pyplot as plt

logging.basicConfig(
    format='%(levelname)s : %(asctime)s : %(message)s',
    level=logging.INFO,
)

parser = ArgumentParser(
    description='Plot the durations of a callable measured by perfmon.',
)
parser.add_argument(
    'config',
    help='The name of the sub-directory that contains the perfmon traces (e.g. slicing)',
)
parser.add_argument(
    'callable',
    help='The name of the callable to plot excluding the module (e.g. QtDimSliderWidget._value_changed).',
)
parser.add_argument(
    '--output', default='latest', help='The name added to output traces file.'
)
args = parser.parse_args()

logging.info(
    f'''Running plot_callable.py with the following arguments.
{args}'''
)

perfmon_dir = pathlib.Path(__file__).parent.parent.resolve(strict=True)

traces_path = perfmon_dir / args.config / f'traces-{args.output}.json'

with open(traces_path) as traces_file:
    traces = json.load(traces_file)

durations_ms = [
    trace['dur'] / 1000 for trace in traces if trace['name'] == args.callable
]

plt.violinplot(durations_ms, vert=False, showmeans=True, showmedians=True)

plt.title(f'{args.config} ({args.output}): {args.callable}')
plt.xlabel('Duration (ms)')
plt.yticks([])

plt.show()
