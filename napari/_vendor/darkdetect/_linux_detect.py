#-----------------------------------------------------------------------------
#  Copyright (C) 2019 Alberto Sottile, Eric Larson
#
#  Distributed under the terms of the 3-clause BSD License.
#-----------------------------------------------------------------------------

import subprocess
import typing

def theme():
    # Here we just triage to GTK settings for now
    try:
        out = subprocess.run(
            ['gsettings', 'get', 'org.gnome.desktop.interface', 'gtk-theme'],
            capture_output=True)
        stdout = out.stdout.decode()
    except Exception:
        return 'Light'
    # we have a string, now remove start and end quote
    theme = stdout.lower().strip()[1:-1]
    if '-dark' in theme.lower():
        return 'Dark'
    else:
        return 'Light'

def isDark():
    return theme() == 'Dark'

def isLight():
    return theme() == 'Light'

def listener(callback: typing.Callable[[str], None]) -> None:
    with subprocess.Popen(
        ('gsettings', 'monitor', 'org.gnome.desktop.interface', 'gtk-theme'),
        stdout=subprocess.PIPE,
        universal_newlines=True,
    ) as p:
        for line in p.stdout:
            callback('Dark' if '-dark' in line.strip().removeprefix("gtk-theme: '").removesuffix("'").lower() else 'Light')
