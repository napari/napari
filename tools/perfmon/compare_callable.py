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
    'baseline',
    default='baseline',
    help='The name added to output traces file for the baseline measurement.',
)
parser.add_argument(
    'test',
    default='test',
    help='The name added to output traces file for the test measurement.',
)
args = parser.parse_args()

logging.info(
    f'''Running compare_callable.py with the following arguments.
{args}'''
)

perfmon_dir = pathlib.Path(__file__).parent.parent.resolve(strict=True)
config_dir = perfmon_dir / args.config


def _get_durations_ms(output_name: str) -> list[float]:
    file_path = str(config_dir / f'traces-{output_name}.json')
    with open(file_path) as traces_file:
        traces = json.load(traces_file)
    return [
        trace['dur'] / 1000
        for trace in traces
        if trace['name'] == args.callable
    ]


baseline_durations_ms = _get_durations_ms(args.baseline)
test_durations_ms = _get_durations_ms(args.test)

plt.violinplot(
    [baseline_durations_ms, test_durations_ms],
    vert=False,
)

plt.title(f'{args.config}: {args.callable} ({args.baseline} vs. {args.test})')
plt.xlabel('Duration (ms)')
plt.yticks([1, 2], [args.baseline, args.test])

plt.show()
