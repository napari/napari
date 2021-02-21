"""
Localization utilities to find available language packs and packages with
localization data.
"""

import gettext
import os
from pathlib import Path

# Entry points
NAPARI_LANGUAGEPACK_ENTRY = "napari.languagepack"

# Constants
DEFAULT_LOCALE = "en"
LOCALE_DIR = "locale"


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
        locale : str
            The language name to use.
        """
        self._locale = locale
        localedir = None
        if locale.split("_")[0] != DEFAULT_LOCALE:
            from napari_plugin_engine.manager import iter_available_plugins

            lang_packs = iter_available_plugins(NAPARI_LANGUAGEPACK_ENTRY)
            data = {k: v for (k, v, _) in lang_packs}
            if locale not in data:
                import warnings

                warnings.warn("Requested locale not available: {locale}")
            else:
                import importlib

                mod = importlib.import_module(data[locale])
                localedir = Path(mod.__file__).parent / LOCALE_DIR

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
