"""
Script to replace minimum requirements with exact requirements in setup.cfg

This ensures that our test matrix includes a version with only the minimum
required versions of packages, and we don't accidentally use features only
available in newer versions.

This script does nothing if the 'MIN_REQ' environment variable is anything
other than '1'.
"""

import os
import sys
from configparser import ConfigParser


def pin_config_minimum_requirements(config_filename):
    # read it with configparser
    config = ConfigParser()
    config.read(config_filename)

    NUMMIN = 'numpy>=1.18.5'

    if NUMMIN not in config['options']['install_requires']:
        sys.exit('numpy min version updated, please update minreq.py')
    # swap out >= requirements for ==
    config['options']['install_requires'] = (
        config['options']['install_requires']
        # meshzoo 10+ only support nu,py 1.20+, all older version of meshzoo
        # have been deleted.
        .replace(NUMMIN, 'numpy==1.20.0').replace('>=', '==')
    )

    config['options.extras_require']['pyside2'] = config[
        'options.extras_require'
    ]['pyside2'].replace('>=', '==')
    config['options.extras_require']['pyqt5'] = config[
        'options.extras_require'
    ]['pyqt5'].replace('>=', '==')

    # rewrite setup.cfg with new config
    with open(config_filename, 'w') as fout:
        config.write(fout)


if __name__ == '__main__':
    if os.environ.get('MIN_REQ', '') == '1':
        # find setup.cfg
        config_filename = os.path.join(
            os.path.dirname(__file__), "..", "setup.cfg"
        )
        pin_config_minimum_requirements(config_filename)
