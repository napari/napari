#-----------------------------------------------------------------------------
#  Copyright (C) 2019 Alberto Sottile
#
#  Distributed under the terms of the 3-clause BSD License.
#-----------------------------------------------------------------------------

import typing

def theme():
    return None
        
def isDark():
    return None
    
def isLight():
    return None

def listener(callback: typing.Callable[[str], None]) -> None:
    raise NotImplementedError()
