#-----------------------------------------------------------------------------
#  Copyright (C) 2019 Alberto Sottile
#
#  Distributed under the terms of the 3-clause BSD License.
#-----------------------------------------------------------------------------

import ctypes
import ctypes.util
import subprocess
import sys
import os
from pathlib import Path
from typing import Callable

try:
    from Foundation import NSObject, NSKeyValueObservingOptionNew, NSKeyValueChangeNewKey, NSUserDefaults
    from PyObjCTools import AppHelper
    _can_listen = True
except ModuleNotFoundError:
    _can_listen = False


try:
    # macOS Big Sur+ use "a built-in dynamic linker cache of all system-provided libraries"
    appkit = ctypes.cdll.LoadLibrary('AppKit.framework/AppKit')
    objc = ctypes.cdll.LoadLibrary('libobjc.dylib')
except OSError:
    # revert to full path for older OS versions and hardened programs
    appkit = ctypes.cdll.LoadLibrary(ctypes.util.find_library('AppKit'))
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('objc'))

void_p = ctypes.c_void_p
ull = ctypes.c_uint64

objc.objc_getClass.restype = void_p
objc.sel_registerName.restype = void_p

# See https://docs.python.org/3/library/ctypes.html#function-prototypes for arguments description
MSGPROTOTYPE = ctypes.CFUNCTYPE(void_p, void_p, void_p, void_p)
msg = MSGPROTOTYPE(('objc_msgSend', objc), ((1 ,'', None), (1, '', None), (1, '', None)))

def _utf8(s):
    if not isinstance(s, bytes):
        s = s.encode('utf8')
    return s

def n(name):
    return objc.sel_registerName(_utf8(name))

def C(classname):
    return objc.objc_getClass(_utf8(classname))

def theme():
    NSAutoreleasePool = objc.objc_getClass('NSAutoreleasePool')
    pool = msg(NSAutoreleasePool, n('alloc'))
    pool = msg(pool, n('init'))

    NSUserDefaults = C('NSUserDefaults')
    stdUserDef = msg(NSUserDefaults, n('standardUserDefaults'))

    NSString = C('NSString')

    key = msg(NSString, n("stringWithUTF8String:"), _utf8('AppleInterfaceStyle'))
    appearanceNS = msg(stdUserDef, n('stringForKey:'), void_p(key))
    appearanceC = msg(appearanceNS, n('UTF8String'))

    if appearanceC is not None:
        out = ctypes.string_at(appearanceC)
    else:
        out = None

    msg(pool, n('release'))

    if out is not None:
        return out.decode('utf-8')
    else:
        return 'Light'

def isDark():
    return theme() == 'Dark'

def isLight():
    return theme() == 'Light'


def _listen_child():
    """
    Run by a child process, install an observer and print theme on change
    """
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    OBSERVED_KEY = "AppleInterfaceStyle"

    class Observer(NSObject):
        def observeValueForKeyPath_ofObject_change_context_(
            self, path, object, changeDescription, context
        ):
            result = changeDescription[NSKeyValueChangeNewKey]
            try:
                print(f"{'Light' if result is None else result}", flush=True)
            except IOError:
                os._exit(1)

    observer = Observer.new()  # Keep a reference alive after installing
    defaults = NSUserDefaults.standardUserDefaults()
    defaults.addObserver_forKeyPath_options_context_(
        observer, OBSERVED_KEY, NSKeyValueObservingOptionNew, 0
    )

    AppHelper.runConsoleEventLoop()


def listener(callback: Callable[[str], None]) -> None:
    if not _can_listen:
        raise NotImplementedError()
    with subprocess.Popen(
        (sys.executable, "-c", "import _mac_detect as m; m._listen_child()"),
        stdout=subprocess.PIPE,
        universal_newlines=True,
        cwd=Path(__file__).parent,
    ) as p:
        for line in p.stdout:
            callback(line.strip())
