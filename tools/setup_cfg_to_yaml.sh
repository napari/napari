#!/bin/bash

set -euo pipefail

setup_cfg_block_to_yaml_list () {
    # 1. use awk in "paragraph mode" (RS= ) to get the block of contiguous lines
    #    that contains the pattern passed in the command line; we use exit to quit
    #    after printing the first result only
    # 2. then get the lines that start with 1+ spaces (section indentation);
    #    these are the dependencies we need to install with conda
    # 3. then replace the spaces with the expected list marker for the yaml file
    awk -v RS= "/$1/{print; exit;}" setup.cfg | # (1)
        grep -E "^ +\w"                       | # (2)
        sed -r "s/^ +/- /g"                     # (3)
}

setup_cfg_block_to_yaml_list $1
