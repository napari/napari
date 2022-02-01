#-----------------------------------------------------------------------------
#  Copyright (C) 2019 Alberto Sottile
#
#  Distributed under the terms of the 3-clause BSD License.
#-----------------------------------------------------------------------------

__version__ = '0.5.2'

import sys
import platform

if sys.platform == "darwin":
    from distutils.version import LooseVersion as V
    if V(platform.mac_ver()[0]) < V("10.14"):
        from ._dummy import *
    else:
        from ._mac_detect import *
    del V
elif sys.platform == "win32" and int(platform.release()) >= 10:
    # Checks if running Windows 10 version 10.0.14393 (Anniversary Update) OR HIGHER. The getwindowsversion method returns a tuple.
    # The third item is the build number that we can use to check if the user has a new enough version of Windows.
    winver = int(platform.version().split('.')[2])
    if winver >= 14393:
        from ._windows_detect import *
    else:
        from ._dummy import *
elif sys.platform == "linux":
    from ._linux_detect import *
else:
    from ._dummy import *

del sys, platform