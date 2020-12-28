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

    # rewrite setup.cfg with new config
    with open(config_filename, 'w') as fout:
        config.write(fout)


def pin_test_minimum_requirements(requirements_filename):
    # read the file
    with open(requirements_filename, 'r') as file:
        lines = file.readlines()

    # force pandas==1.1.5 for compatibility with minimum numpy (1.16.0)
    output_lines = []
    for line in lines:
        if line == 'pandas\n':
            line = 'pandas==1.1.5\n'
        output_lines.append(line)

    # rewrite requirements/test.txt with new requirements
    with open(requirements_filename, 'w') as fout:
        fout.writelines(output_lines)


if __name__ == '__main__':
    if os.environ.get('MIN_REQ', '') == '1':
        # find setup.cfg
        config_filename = os.path.join(
            os.path.dirname(__file__), "..", "setup.cfg"
        )
        pin_config_minimum_requirements(config_filename)

        # find requirements/test.txt
        test_requirements_filename = os.path.join(
            os.path.dirname(__file__), "..", "requirements", "test.txt"
        )
        pin_test_minimum_requirements(test_requirements_filename)
