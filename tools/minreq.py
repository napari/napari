"""
Script to replace minimum requirements with exact requirements in setup.cfg

This ensures that our test matrix includes a version with only the minimum
required versions of packages, and we don't accidentally use features only
available in newer versions.

This script does nothing if the 'MIN_REQ' environment variable is anything
other than '1'.
"""

import os
from configparser import ConfigParser
from pathlib import Path


def pin_config_minimum_requirements(config_filename):
    # read it with configparser
    config = ConfigParser()
    config.read(config_filename)

    # swap out >= requirements for ==
    config['options']['install_requires'] = config['options'][
        'install_requires'
    ].replace('>=', '==')
    config['options.extras_require']['pyside2'] = config[
        'options.extras_require'
    ]['pyside2'].replace('>=', '==')
    config['options.extras_require']['pyqt5'] = config[
        'options.extras_require'
    ]['pyqt5'].replace('>=', '==')
    config['options.extras_require']['testing'] = config[
        'options.extras_require'
    ]['testing'].replace('>=', '==')

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

        tox_file = Path(__file__).parent.parent / 'tox.ini'

        tox_file.write_text(
            tox_file.read_text().replace(' --only-binary :all:', '')
        )
