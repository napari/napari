"""
Localization utilities to find available language packs and packages with
localization data.
"""

import gettext
import importlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pkg_resources

# Entry points
NAPARI_LANGUAGEPACK_ENTRY = "napari.languagepack"

# Constants
DEFAULT_LOCALE = "en"
LOCALE_DIR = "locale"


def _main():
    """
    Run functions in this file in a subprocess and prints to stdout the results.
    """
    data = {}
    message = ""
    if len(sys.argv) == 2:
        func_name = sys.argv[-1]
        func = globals().get(func_name, None)

        if func:
            try:
                data, message = func()
            except Exception:
                message = traceback.format_exc()
    else:
        message = "Invalid number of arguments!"

    sys.stdout.write(json.dumps({"data": data, "message": message}))


def _get_installed_language_pack_locales() -> dict:
    """
    Get available installed language pack locales.

    Returns
    -------
    dict
        A dict with all the languages found.

    Notes
    -----
    This functions are meant to be called via a subprocess to guarantee the
    results represent the most up-to-date entry point information, which
    is defined on interpreter startup.
    """
    data = {}
    messages = []
    for entry_point in pkg_resources.iter_entry_points(
        NAPARI_LANGUAGEPACK_ENTRY
    ):
        try:
            data[entry_point.name] = os.path.basename(
                os.path.dirname(entry_point.load().__file__)
            )
        except Exception:
            messages.append(traceback.format_exc())

    message = "\n".join(messages)
    return data, message


def _get_display_name(
    locale: str, display_locale: str = DEFAULT_LOCALE
) -> str:
    """
    Return the language name to use with a `display_locale` for a given language locale.
    Parameters
    ----------
    locale: str
        The language name to use.
    display_locale: str, optional
        The language to display the `locale`.
    Returns
    -------
    str
        Localized `locale` and capitalized language name using `display_locale` as language.
    """
    # This is a dependency of the language packs to keep out of core
    import babel

    locale = locale if _is_valid_locale(locale) else DEFAULT_LOCALE
    display_locale = (
        display_locale if _is_valid_locale(display_locale) else DEFAULT_LOCALE
    )
    loc = babel.Locale.parse(locale)
    return loc.get_display_name(display_locale).capitalize()


def _is_valid_locale(locale: str) -> bool:
    """
    Check if a `locale` value is valid.

    Parameters
    ----------
    locale: str
        Language locale code.

    Notes
    -----
    A valid locale is in the form language (See ISO-639 standard) and an
    optional territory (See ISO-3166 standard).

    Examples of valid locales:
    - English: DEFAULT_LOCALE
    - Australian English: "en_AU"
    - Portuguese: "pt"
    - Brazilian Portuguese: "pt_BR"

    Examples of invalid locales:
    - Australian Spanish: "es_AU"
    - Brazilian German: "de_BR"
    """
    # This is a dependency of the language packs to keep out of core
    import babel

    valid = False
    try:
        babel.Locale.parse(locale)
        valid = True
    except babel.core.UnknownLocaleError:
        pass
    except ValueError:
        pass

    return valid


def _run_process_and_parse(cmd: list):
    """
    Run a list of commands and return the result parsed form stdout.

    Parameters
    ----------
    cmd: list
        List of commands

    Returns
    -------
    tuple
        A tuple in the form `(result_dict, message)`.
    """
    result = {"data": {}, "message": ""}
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = p.communicate()
        result = json.loads(stdout.decode('utf-8'))
    except Exception:
        result["message"] = (
            traceback.format_exc() + "\n" + repr(stderr.decode('utf-8'))
        )

    return result["data"], result["message"]


def get_installed_packages_locales() -> dict:
    """
    Get all jupyterlab extensions installed that contain locale data.

    Returns
    -------
    dict
        Ordered list of available language packs.
        >>> {"package-name": locale_data, ...}

    Examples
    --------
    - `entry_points={"jupyterlab.locale": "package-name = package_module"}`
    - `entry_points={"jupyterlab.locale": "jupyterlab-git = jupyterlab_git"}`
    """
    cmd = [sys.executable, __file__, "_get_installed_language_pack_locales"]
    found_package_locales, message = _run_process_and_parse(cmd)
    return found_package_locales, message


# --- Translators
# ----------------------------------------------------------------------------
class TranslationBundle:
    """
    Translation bundle providing gettext translation functionality.
    """

    def __init__(self, domain: str, locale: str):
        self._domain = domain
        self._locale = locale

        self._update_locale(locale)

    def _update_locale(self, locale: str):
        """
        Update the locale environment variables.

        Parameters
        ----------
        locale: str
            The language name to use.
        """
        self._locale = locale
        localedir = None
        if locale.split("_")[0] != DEFAULT_LOCALE:
            data, _ = get_installed_packages_locales()
            language_pack_module = data.get(locale)
            try:
                if language_pack_module is not None:
                    mod = importlib.import_module(language_pack_module)
                    localedir = Path(mod.__file__).parent / LOCALE_DIR
            except ImportError:
                pass

        gettext.bindtextdomain(self._domain, localedir=localedir)

    def gettext(self, msgid: str) -> str:
        """
        Translate a singular string.

        Parameters
        ----------
        msgid: str
            The singular string to translate.

        Returns
        -------
        str
            The translated string.
        """
        return gettext.dgettext(self._domain, msgid)

    def ngettext(self, msgid: str, msgid_plural: str, n: int) -> str:
        """
        Translate a singular string with pluralization.

        Parameters
        ----------
        msgid: str
            The singular string to translate.
        msgid_plural: str
            The plural string to translate.
        n: int
            The number for pluralization.

        Returns
        -------
        str
            The translated string.
        """
        return gettext.dngettext(self._domain, msgid, msgid_plural, n)

    def pgettext(self, msgctxt: str, msgid: str) -> str:
        """
        Translate a singular string with context.

        Parameters
        ----------
        msgctxt: str
            The message context.
        msgid: str
            The singular string to translate.

        Returns
        -------
        str
            The translated string.
        """
        return gettext.dpgettext(self._domain, msgctxt, msgid)

    def npgettext(
        self, msgctxt: str, msgid: str, msgid_plural: str, n: int
    ) -> str:
        """
        Translate a singular string with context and pluralization.

        Parameters
        ----------
        msgctxt: str
            The message context.
        msgid: str
            The singular string to translate.
        msgid_plural: str
            The plural string to translate.
        n: int
            The number for pluralization.

        Returns
        -------
        str
            The translated string.
        """
        return gettext.dnpgettext(
            self._domain, msgctxt, msgid, msgid_plural, n
        )

    # Shorthands
    def _(self, msgid: str) -> str:
        """
        Shorthand for gettext.

        Parameters
        ----------
        msgid: str
            The singular string to translate.

        Returns
        -------
        str
            The translated string.
        """
        return self.gettext(msgid)

    def _n(self, msgid: str, msgid_plural: str, n: int) -> str:
        """
        Shorthand for ngettext.

        Parameters
        ----------
        msgid: str
            The singular string to translate.
        msgid_plural: str
            The plural string to translate.
        n: int
            The number for pluralization.

        Returns
        -------
        str
            The translated string.
        """
        return self.ngettext(msgid, msgid_plural, n)

    def _p(self, msgctxt: str, msgid: str) -> str:
        """
        Shorthand for pgettext.

        Parameters
        ----------
        msgctxt: str
            The message context.
        msgid: str
            The singular string to translate.

        Returns
        -------
        str
            The translated string.
        """
        return self.pgettext(msgctxt, msgid)

    def _np(self, msgctxt: str, msgid: str, msgid_plular: str, n: str) -> str:
        """
        Shorthand for npgettext.

        Parameters
        ----------
        msgctxt: str
            The message context.
        msgid: str
            The singular string to translate.
        msgid_plural: str
            The plural string to translate.
        n: int
            The number for pluralization.

        Returns
        -------
        str
            The translated string.
        """
        return self.npgettext(msgctxt, msgid, msgid_plular, n)


class _Translator:
    """
    Translations manager.
    """

    _TRANSLATORS = {}
    _LOCALE = DEFAULT_LOCALE

    @staticmethod
    def _update_env(locale: str):
        """
        Update the locale environment variables based on the settings.

        Parameters
        ----------
        locale: str
            The language name to use.
        """
        for key in ["LANGUAGE", "LANG"]:
            os.environ[key] = f"{locale}.UTF-8"

    @classmethod
    def _set_locale(cls, locale: str):
        """
        Set locale for the translation bundles based on the settings.

        Parameters
        ----------
        locale: str
            The language name to use.
        """
        if _is_valid_locale(locale):
            cls._LOCALE = locale

            if locale.split("_")[0] != DEFAULT_LOCALE:
                translator._update_env(locale)

            for __, bundle in cls._TRANSLATORS.items():
                bundle._update_locale(locale)

    @classmethod
    def load(cls, domain: str = "napari") -> TranslationBundle:
        """
        Load translation domain.

        The domain is usually the normalized ``package_name``.

        Parameters
        ----------
        domain: str
            The translations domain. The normalized python package name.

        Returns
        -------
        Translator
            A translator instance bound to the domain.
        """
        if domain in cls._TRANSLATORS:
            trans = cls._TRANSLATORS[domain]
        else:
            trans = TranslationBundle(domain, cls._LOCALE)
            cls._TRANSLATORS[domain] = trans

        return trans


translator = _Translator


if __name__ == "__main__":
    _main()
