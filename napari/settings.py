import os
import sys
from collections import abc, namedtuple

from .utils.settings_formats.json import JSON_FORMAT
from .vendored.appdirs import site_config_dir, user_config_dir

# Read/Write funcs should accept a filename, Write funcs also take a dict

_fields = ['key', 'default', 'description', 'type', 'callback']
SettingTuple = namedtuple("Setting", _fields, defaults=(None,) * len(_fields))

RESTORE_GEOMETRY = SettingTuple(
    "prefs/restore_geometry",
    True,
    "Preserve window size/position across sessions",
)


class ReprMixin:
    """Mixin to make View subclasses behave like builtin views"""

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self)})'


class SettingsKeys(ReprMixin, abc.KeysView):
    pass


class SettingsItems(ReprMixin, abc.ItemsView):
    pass


class SettingsValues(ReprMixin, abc.ValuesView):
    pass


class QSettingsMixin:
    """A (optional) mixin of aliases to mimic QSettings API"""

    def setValue(self, key, value):
        self.__setitem__(key, value)

    def value(self, key, default=None):
        return self.get(key, default=default)

    def allKeys(self):
        return self.keys()

    def contains(self, key):
        return key in self

    def remove(self, key):
        del self[key]


def validate_format(format):
    if not isinstance(format, abc.Sequence):
        raise TypeError("`format` must be a sequence (list, tuple)")
    if not len(format) == 3:
        raise ValueError("`format` must have three items")
    ext, readfunc, writefunc = format
    if not isinstance(ext, str):
        raise TypeError("The first item of `format` must be a string")
    if not callable(readfunc):
        raise TypeError(
            "The second item of `format` must be a callable "
            "function that reads settings from disk"
        )
    if not callable(writefunc):
        raise TypeError(
            "The third item of `format` must be a callable "
            "function that writes settings to disk"
        )


class Settings(abc.MutableMapping, QSettingsMixin):
    def __init__(
        self,
        initial=None,
        orgname='napari',
        appname='napari',
        scope='user',
        format=JSON_FORMAT,
        autosync=True,
    ):
        """Settings object to hold general application preferences & options.

        Parameters
        ----------
        initial : dict, optional
            Initial key:value map to populate settings, by default None
        orgname : str, optional
            Name of the application organization, by default 'napari'
        appname : str, optional
            Name of the application name, by default 'napari'
        scope : {'user', 'system'}, optional
            Whether settings are user-specific ('user') or shared by all users
            of the same system ('system') by default 'user'
        format : tuple, optional
            Custom (extension, readfunc, writefunc) for reading and writing
            settings to and from disk. by default JSON format
        autosync : bool, optional
            Whether to automatically sync to disk after set/delete operations,
            by default True

        Raises
        ------
        ValueError
            If `initial` argument is not a dict
        ValueError
            If `scope` is not in ('user', 'system')
        TypeError
            If `format` is not a sequence
        ValueError
            If `format` does not have exactly three items
        TypeError
            If the items in `format` are not (str, callable, callable)
        """
        self.scope = scope
        self.orgname = orgname
        self.appname = appname
        self.autosync = False
        self.format = format

        self._current = {}
        self._registered = {}
        if initial:
            if not isinstance(initial, dict):
                raise ValueError('`initial` argument must be a dict')
            self.update(initial)

        self.autosync = autosync
        if autosync:
            self.sync()

    @property
    def format(self):
        return (self.extension, self.readfunc, self.writefunc)

    @format.setter
    def format(self, format):
        validate_format(format)
        extension, readfunc, writefunc = format
        self.extension = extension
        self.readfunc = readfunc
        self.writefunc = writefunc

    def register_format(self, format):
        """alias for format setter"""
        self.format = format

    @property
    def scope(self):
        """Whether settings are user-specific or shared by all users."""
        return self._scope

    @scope.setter
    def scope(self, value):
        if value not in ('user', 'system'):
            raise ValueError('`scope` must be either "user" or "system"')
        self._scope = value

    @property
    def path(self):
        """Full filepath where the settings will be written/read to/from disk

        Returns
        -------
        str
            Filepath of the settings
        """
        if self.scope == 'user':
            dir = user_config_dir(self.appname, self.orgname)
        else:
            dir = site_config_dir(self.appname, self.orgname)
        fname = f'prefs.{self.extension}'
        return os.path.join(dir, fname)

    def __getitem__(self, key):
        if isinstance(key, SettingTuple):
            key = key.key
        return self._current[key]

    def __setitem__(self, key, value):
        self._current[key] = value

        cb = self._registered.get(key, {}).get('callback')
        if callable(cb):
            cb(value)

        if self.autosync:
            self.sync()

    def __delitem__(self, key):
        del self._current[key]
        if self.autosync:
            self.sync()

    def __iter__(self):
        return iter(self._current)

    def __len__(self):
        return len(self._current)

    def __repr__(self):
        return '{self.__class__.__name__}({self._current})'.format(self=self)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        return super().__getattr__(key)

    def keys(self):
        return SettingsKeys(self)

    def items(self):
        return SettingsItems(self)

    def values(self):
        return SettingsValues(self)

    def read(self):
        """Populate settings from disk.

        Raises
        ------
        TypeError
            If self.readfunc does not return a dict/Mapping
        """
        try:
            incoming = self.readfunc(self.path)
        except FileNotFoundError:
            incoming = {}
        except Exception as e:
            import warnings

            warnings.warn(f'Failed to read settings at {self.path}: {e}')
            incoming = {}

        if not isinstance(incoming, abc.Mapping):
            raise TypeError(
                f'{self.readfunc.__name__} returned an invalid format. '
                'settings readfuncs must return a mapping'
            )
        self._current = incoming

    def write(self):
        """Write settings to disk."""
        if not os.path.isdir(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.writefunc(self.path, self._current)

    def sync(self):
        """Writes any unsaved changes to permanent storage, and reloads any
        settings that have been changed in the meantime by another
        application."""
        # TODO: this is useless until we add implement logic that allows
        # the settings object to know what keys have been added/removed/updated
        # since the last read.
        if len(self):
            self.write()
        else:
            try:
                self.read()
            except FileNotFoundError:
                pass

    def register_setting(
        self, key, default, description, dtype=None, callback=None
    ):
        """Register a key as a setting that can appear in the GUI preferences.

        Only registered settings will be exposed to the user in the GUI
        preferences window.

        The value of type(default) will be used to generate the appropriate
        QWidget in the GUI. To generate a QComboBox with a selection of
        choices, default must be an Enum.  Choices will be introspected from
        the enum.

        Parameters
        ----------
        key : string
            key used to store the value of this setting
        default
            default value of this setting
        description : str
            description for this setting.
        dtype : type, optional
            Enforce a specific data type for this setting.
            by default type(default)
        callback : callable, optional
            if provided, will be called when the key is changed (with the new
            value as the argument)
        """
        if (
            dtype is not None
            and default is not None
            and dtype != type(default)
        ):
            raise NotImplementedError(
                'cannot currently set dtype to a different type than default '
                'value. Please raise an issue with a use case.'
            )
        if callback is not None and not callable(callback):
            raise TypeError("`callback` must be callable")

        self.setdefault(key, default)
        self._registered[key] = {
            'default': default,
            'description': description,
            'type': type(default) if dtype is None else dtype,
            'callback': callback,
        }


SETTINGS = Settings()

this = sys.modules[__name__]
_internal = {
    k: v for k, v in this.__dict__.items() if isinstance(v, SettingTuple)
}

for item in _internal.values():
    SETTINGS.register_setting(*item)
