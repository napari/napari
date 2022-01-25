def install(*package_names: str):
    """install plugin(s) in active environment"""


def uninstall(*package_names: str):
    """uninstall plugin(s) from active environment"""


def register():
    """Register plugin dynamically.  (npe2) API pending"""


# don't think we need both unregister and deactivate?
# def unregister():
#     """Unregister plugin dynamically.  (npe2) API pending"""


def enable():
    """Enable plugin."""


def disable():
    """Disable plugin. (remove from menus, etc...)"""


def list():
    """List plugins"""
