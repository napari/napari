"""
Script to replace minimum requirements for testing with exact requirements
in requirements/test.txt

This ensures that our test matrix includes a version with only the minimum
required versions of packages, and we don't accidentally use features only
available in newer versions.

This script does nothing if the 'MIN_REQ' environment variable is anything
other than '1'.
"""

import os


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
        # find requirements/test.txt
        config_filename = os.path.join(
            os.path.dirname(__file__), "..", "requirements", "test.txt"
        )
        pin_test_minimum_requirements(config_filename)
