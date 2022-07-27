import argparse
import json
import pathlib

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='Plot the durations of a callable measured by perfmon.'
)
parser.add_argument(
    'subdir',
    type=str,
    help='The name of the perfmon sub-directory that contains the traces (e.g. slicing)',
)
parser.add_argument(
    'callable',
    type=str,
    help='The name of the callable to plot excluding the module.',
)
args = parser.parse_args()

perfmon_dir = pathlib.Path(__file__).parent.parent.resolve(strict=True)

traces_file_path = perfmon_dir / args.subdir / 'traces.json'

with open(traces_file_path) as traces_file:
    traces = json.load(traces_file)

durations_ms = [
    trace['dur'] / 1000 for trace in traces if trace['name'] == args.callable
]

plt.violinplot(durations_ms, vert=False, showmeans=True, showmedians=True)

plt.title(f'{args.subdir}: {args.callable}')
plt.xlabel('Duration (ms)')
plt.yticks([])

plt.show()
