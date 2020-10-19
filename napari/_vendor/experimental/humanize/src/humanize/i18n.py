import gettext as gettext_module
import os.path
from threading import local

__all__ = ["activate", "deactivate", "gettext", "ngettext"]

_TRANSLATIONS = {None: gettext_module.NullTranslations()}
_CURRENT = local()


def _get_default_locale_path():
    try:
        if __file__ is None:
            return None
        return os.path.join(os.path.dirname(__file__), "locale")
    except NameError:
        return None


def get_translation():
    try:
        return _TRANSLATIONS[_CURRENT.locale]
    except (AttributeError, KeyError):
        return _TRANSLATIONS[None]


def activate(locale, path=None):
    """Activate internationalisation.

    Set `locale` as current locale. Search for locale in directory `path`.

    Args:
        locale (str): Language name, e.g. `en_GB`.
        path (str): Path to search for locales.

    Returns:
        dict: Translations.
    """
    if path is None:
        path = _get_default_locale_path()

    if path is None:
        raise Exception(
            "Humanize cannot determinate the default location of the 'locale' folder. "
            "You need to pass the path explicitly."
        )
    if locale not in _TRANSLATIONS:
        translation = gettext_module.translation("humanize", path, [locale])
        _TRANSLATIONS[locale] = translation
    _CURRENT.locale = locale
    return _TRANSLATIONS[locale]


def deactivate():
    """Deactivate internationalisation."""
    _CURRENT.locale = None


def gettext(message):
    """Get translation.

    Args:
        message (str): Text to translate.

    Returns:
        str: Translated text.
    """
    return get_translation().gettext(message)


def pgettext(msgctxt, message):
    """'Particular gettext' function.

    It works with `msgctxt` .po modifiers and allows duplicate keys with different
    translations.

    Args:
        msgctxt (str): Context of the translation.
        message (str): Text to translate.

    Returns:
        str: Translated text.
    """
    # This GNU gettext function was added in Python 3.8, so for older versions we
    # reimplement it. It works by joining `msgctx` and `message` by '4' byte.
    try:
        # Python 3.8+
        return get_translation().pgettext(msgctxt, message)
    except AttributeError:
        # Python 3.7 and older
        key = msgctxt + "\x04" + message
        translation = get_translation().gettext(key)
        return message if translation == key else translation


def ngettext(message, plural, num):
    """Plural version of gettext.

    Args:
        message (str): Singlular text to translate.
        plural (str): Plural text to translate.
        num (str): The number (e.g. item count) to determine translation for the
            respective grammatical number.

    Returns:
        str: Translated text.
    """
    return get_translation().ngettext(message, plural, num)


def gettext_noop(message):
    """Mark a string as a translation string without translating it.

    Example usage:
    ```python
    CONSTANTS = [gettext_noop('first'), gettext_noop('second')]
    def num_name(n):
        return gettext(CONSTANTS[n])
    ```

    Args:
        message (str): Text to translate in the future.

    Returns:
        str: Original text, unchanged.
    """
    return message
