from winreg import HKEY_CURRENT_USER as hkey, QueryValueEx as getSubkeyValue, OpenKey as getKey

import ctypes
import ctypes.wintypes

advapi32 = ctypes.windll.advapi32

# LSTATUS RegOpenKeyExA(
#     HKEY hKey,
#     LPCSTR lpSubKey,
#     DWORD ulOptions,
#     REGSAM samDesired,
#     PHKEY phkResult
# );
advapi32.RegOpenKeyExA.argtypes = (
    ctypes.wintypes.HKEY,
    ctypes.wintypes.LPCSTR,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.DWORD,
    ctypes.POINTER(ctypes.wintypes.HKEY),
)
advapi32.RegOpenKeyExA.restype = ctypes.wintypes.LONG

# LSTATUS RegQueryValueExA(
#     HKEY hKey,
#     LPCSTR lpValueName,
#     LPDWORD lpReserved,
#     LPDWORD lpType,
#     LPBYTE lpData,
#     LPDWORD lpcbData
# );
advapi32.RegQueryValueExA.argtypes = (
    ctypes.wintypes.HKEY,
    ctypes.wintypes.LPCSTR,
    ctypes.wintypes.LPDWORD,
    ctypes.wintypes.LPDWORD,
    ctypes.wintypes.LPBYTE,
    ctypes.wintypes.LPDWORD,
)
advapi32.RegQueryValueExA.restype = ctypes.wintypes.LONG

# LSTATUS RegNotifyChangeKeyValue(
#     HKEY hKey,
#     WINBOOL bWatchSubtree,
#     DWORD dwNotifyFilter,
#     HANDLE hEvent,
#     WINBOOL fAsynchronous
# );
advapi32.RegNotifyChangeKeyValue.argtypes = (
    ctypes.wintypes.HKEY,
    ctypes.wintypes.BOOL,
    ctypes.wintypes.DWORD,
    ctypes.wintypes.HANDLE,
    ctypes.wintypes.BOOL,
)
advapi32.RegNotifyChangeKeyValue.restype = ctypes.wintypes.LONG

def theme():
    """ Uses the Windows Registry to detect if the user is using Dark Mode """
    # Registry will return 0 if Windows is in Dark Mode and 1 if Windows is in Light Mode. This dictionary converts that output into the text that the program is expecting.
    valueMeaning = {0: "Dark", 1: "Light"}
    # In HKEY_CURRENT_USER, get the Personalisation Key.
    try:
        key = getKey(hkey, "Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize")
        # In the Personalisation Key, get the AppsUseLightTheme subkey. This returns a tuple.
        # The first item in the tuple is the result we want (0 or 1 indicating Dark Mode or Light Mode); the other value is the type of subkey e.g. DWORD, QWORD, String, etc.
        subkey = getSubkeyValue(key, "AppsUseLightTheme")[0]
    except FileNotFoundError:
        # some headless Windows instances (e.g. GitHub Actions or Docker images) do not have this key
        return None
    return valueMeaning[subkey]

def isDark():
    if theme() is not None:
        return theme() == 'Dark'

def isLight():
    if theme() is not None:
        return theme() == 'Light'

#def listener(callback: typing.Callable[[str], None]) -> None:
def listener(callback):
    hKey = ctypes.wintypes.HKEY()
    advapi32.RegOpenKeyExA(
        ctypes.wintypes.HKEY(0x80000001), # HKEY_CURRENT_USER
        ctypes.wintypes.LPCSTR(b'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize'),
        ctypes.wintypes.DWORD(),
        ctypes.wintypes.DWORD(0x00020019), # KEY_READ
        ctypes.byref(hKey),
    )

    dwSize = ctypes.wintypes.DWORD(ctypes.sizeof(ctypes.wintypes.DWORD))
    queryValueLast = ctypes.wintypes.DWORD()
    queryValue = ctypes.wintypes.DWORD()
    advapi32.RegQueryValueExA(
        hKey,
        ctypes.wintypes.LPCSTR(b'AppsUseLightTheme'),
        ctypes.wintypes.LPDWORD(),
        ctypes.wintypes.LPDWORD(),
        ctypes.cast(ctypes.byref(queryValueLast), ctypes.wintypes.LPBYTE),
        ctypes.byref(dwSize),
    )

    while True:
        advapi32.RegNotifyChangeKeyValue(
            hKey,
            ctypes.wintypes.BOOL(True),
            ctypes.wintypes.DWORD(0x00000004), # REG_NOTIFY_CHANGE_LAST_SET
            ctypes.wintypes.HANDLE(None),
            ctypes.wintypes.BOOL(False),
        )
        advapi32.RegQueryValueExA(
            hKey,
            ctypes.wintypes.LPCSTR(b'AppsUseLightTheme'),
            ctypes.wintypes.LPDWORD(),
            ctypes.wintypes.LPDWORD(),
            ctypes.cast(ctypes.byref(queryValue), ctypes.wintypes.LPBYTE),
            ctypes.byref(dwSize),
        )
        if queryValueLast.value != queryValue.value:
            queryValueLast.value = queryValue.value
            callback('Light' if queryValue.value else 'Dark')
