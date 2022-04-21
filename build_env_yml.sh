#!/bin/bash

set -euo pipefail

setup_cfg_block_to_yaml_list () {
    # 1. use awk in "paragraph mode" (RS= ) to get the block of contiguous lines
    #    that contains the pattern passed in the command line
    # 2. then get the lines that start with 4 spaces (section indentation);
    #    these are the dependencies we need to install with conda
    # 3. then replace the spaces with the expected list marker for the yaml file
    awk -v RS= "/$1/" setup.cfg |  # (1)
        grep -E "^\s+" |           # (2)
        sed "s/    / - /g"         # (3)
}

setup_cfg_block_to_yaml_list $1
