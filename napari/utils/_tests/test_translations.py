"""Test the translations API."""

from copy import deepcopy
from pathlib import Path

import pytest

from napari.utils.translations import (
    TranslationString,
    _get_display_name,
    _is_valid_locale,
    _load_language,
    get_language_packs,
    translator,
)

TEST_LOCALE = "es_CO"
HERE = Path(__file__).parent
TEST_LANGUAGE_PACK_PATH = HERE / "napari-language-pack-es_CO"


es_CO_po = r"""msgid ""
msgstr ""
"Project-Id-Version: \n"
"POT-Creation-Date: 2021-02-18 19:00\n"
"PO-Revision-Date: 2021-04-08 08:14-0500\n"
"Language-Team: \n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"
"Content-Transfer-Encoding: 8bit\n"
"X-Generator: Poedit 2.4.2\n"
"Last-Translator: \n"
"Plural-Forms: nplurals=2; plural=(n != 1);\n"
"Language: es_CO\n"

#: Test for singular
msgid "More about napari"
msgstr "Más sobre napari"

#: Test for singular with context
msgctxt "singular-context"
msgid "More about napari with context"
msgstr "Más sobre napari con contexto"

#: Test for singular with variables
msgid "More about napari with {variable}"
msgstr "Más sobre napari con {variable}"

#: Test for singular with and context variables
msgctxt "singular-context-variables"
msgid "More about napari with context and {variable}"
msgstr "Más sobre napari con contexto y {variable}"

#: Test for plural
msgid "I have napari"
msgid_plural "I have naparis"
msgstr[0] "Tengo napari"
msgstr[1] "Tengo naparis"

#: Test for singular with context
msgctxt "plural-context"
msgid "I have napari with context"
msgid_plural "I have naparis with context"
msgstr[0] "Tengo napari con contexto"
msgstr[1] "Tengo naparis con contexto"

#: Test for plural with variables
msgid "I have {n} napari with {variable}"
msgid_plural "I have {n} naparis with {variable}"
msgstr[0] "Tengo {n} napari con {variable}"
msgstr[1] "Tengo {n} naparis con {variable}"

#: Test for singular with and context variables
msgctxt "plural-context-variables"
msgid "I have {n} napari with {variable} and context"
msgid_plural "I have {n} naparis with {variable} and context"
msgstr[0] "Tengo {n} napari con {variable} y contexto"
msgstr[1] "Tengo {n} naparis con {variable} y contexto"
"""

es_CO_mo = b'\xde\x12\x04\x95\x00\x00\x00\x00\t\x00\x00\x00\x1c\x00\x00\x00d\x00\x00\x00\r\x00\x00\x00\xac\x00\x00\x00\x00\x00\x00\x00\xe0\x00\x00\x00\x1c\x00\x00\x00\xe1\x00\x00\x00D\x00\x00\x00\xfe\x00\x00\x00\x11\x00\x00\x00C\x01\x00\x00!\x00\x00\x00U\x01\x00\x00E\x00\x00\x00w\x01\x00\x00u\x00\x00\x00\xbd\x01\x00\x00/\x00\x00\x003\x02\x00\x00H\x00\x00\x00c\x02\x00\x00\x0e\x01\x00\x00\xac\x02\x00\x00\x1a\x00\x00\x00\xbb\x03\x00\x00@\x00\x00\x00\xd6\x03\x00\x00\x11\x00\x00\x00\x17\x04\x00\x00 \x00\x00\x00)\x04\x00\x004\x00\x00\x00J\x04\x00\x00V\x00\x00\x00\x7f\x04\x00\x00\x1e\x00\x00\x00\xd6\x04\x00\x00+\x00\x00\x00\xf5\x04\x00\x00\x01\x00\x00\x00\t\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x03\x00\x00\x00\x08\x00\x00\x00\x06\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x07\x00\x00\x00\x02\x00\x00\x00\x00I have napari\x00I have naparis\x00I have {n} napari with {variable}\x00I have {n} naparis with {variable}\x00More about napari\x00More about napari with {variable}\x00plural-context\x04I have napari with context\x00I have naparis with context\x00plural-context-variables\x04I have {n} napari with {variable} and context\x00I have {n} naparis with {variable} and context\x00singular-context\x04More about napari with context\x00singular-context-variables\x04More about napari with context and {variable}\x00Project-Id-Version: \nPO-Revision-Date: 2021-04-08 08:14-0500\nLanguage-Team: \nMIME-Version: 1.0\nContent-Type: text/plain; charset=UTF-8\nContent-Transfer-Encoding: 8bit\nX-Generator: Poedit 2.4.2\nLast-Translator: \nPlural-Forms: nplurals=2; plural=(n != 1);\nLanguage: es_CO\n\x00Tengo napari\x00Tengo naparis\x00Tengo {n} napari con {variable}\x00Tengo {n} naparis con {variable}\x00M\xc3\xa1s sobre napari\x00M\xc3\xa1s sobre napari con {variable}\x00Tengo napari con contexto\x00Tengo naparis con contexto\x00Tengo {n} napari con {variable} y contexto\x00Tengo {n} naparis con {variable} y contexto\x00M\xc3\xa1s sobre napari con contexto\x00M\xc3\xa1s sobre napari con contexto y {variable}\x00'


@pytest.fixture
def trans(tmp_path):
    """A good plugin that uses entry points."""
    distinfo = tmp_path / "napari_language_pack_es_CO-0.1.0.dist-info"
    distinfo.mkdir()
    (distinfo / "top_level.txt").write_text('napari_language_pack_es_CO')
    (distinfo / "entry_points.txt").write_text(
        "[napari.languagepack]\nes_CO = napari_language_pack_es_CO\n"
    )
    (distinfo / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: napari-language-pack-es-CO\n"
        "Version: 0.1.0\n"
    )
    pkgdir = tmp_path / 'napari_language_pack_es_CO'
    msgs = pkgdir / 'locale' / 'es_CO' / 'LC_MESSAGES'
    msgs.mkdir(parents=True)
    (pkgdir / '__init__.py').touch()
    (msgs / "napari.po").write_text(es_CO_po)
    (msgs / "napari.mo").write_bytes(es_CO_mo)

    from napari_plugin_engine.manager import temp_path_additions

    with temp_path_additions(tmp_path):
        # Load translator and force a locale for testing
        translator._set_locale(TEST_LOCALE)
        return translator.load()


def test_get_language_packs(trans):
    result = get_language_packs()
    assert result == {
        'en': {'displayName': 'English', 'nativeName': 'English'}
    }


def test_get_display_name_valid():
    assert _get_display_name("en", "en") == "English"
    assert _get_display_name("en", "es") == "Inglés"
    assert _get_display_name("en", "es_CO") == "Inglés"
    assert _get_display_name("en", "fr") == "Anglais"
    assert _get_display_name("es", "en") == "Spanish"
    assert _get_display_name("fr", "en") == "French"


def test_get_display_name_invalid():
    assert _get_display_name("en", "foo") == "English"
    assert _get_display_name("foo", "en") == "English"
    assert _get_display_name("foo", "bar") == "English"


def test_is_valid_locale_valid():
    assert _is_valid_locale("en")
    assert _is_valid_locale("es")
    assert _is_valid_locale("es_CO")


def test_is_valid_locale_invalid():
    assert not _is_valid_locale("foo_SPAM")
    assert not _is_valid_locale("bar")


def test_load_language_valid(tmp_path):
    # This is valid content
    data = """
application:
  language: es_ES
"""
    temp_config_path = tmp_path / "tempconfig.yml"
    with open(temp_config_path, "w") as fh:
        fh.write(data)

    result = _load_language(temp_config_path)
    assert result == "es_ES"


def test_load_language_invalid(tmp_path):
    # This is invalid content
    data = ":"
    temp_config_path = tmp_path / "tempconfig.yml"
    with open(temp_config_path, "w") as fh:
        fh.write(data)

    with pytest.warns(UserWarning):
        _load_language(temp_config_path)


def test_locale_invalid():
    with pytest.warns(UserWarning):
        translator._set_locale(TEST_LOCALE)
        trans = translator.load()
        result = trans._("BOO")
        assert result == "BOO"


# Test trans methods
# ------------------
def test_locale_singular(trans):
    expected_result = "Más sobre napari"
    result = trans._("More about napari")
    assert result == expected_result


def test_locale_singular_with_format(trans):
    variable = 1
    singular = "More about napari with {variable}"
    expected_result = f"Más sobre napari con {variable}"
    result = trans._(singular, variable=variable)
    assert result == expected_result


def test_locale_singular_deferred_with_format(trans):
    variable = 1
    singular = "More about napari with {variable}"
    original_result = f"More about napari with {variable}"
    translated_result = f"Más sobre napari con {variable}"
    result = trans._(singular, deferred=True, variable=variable)
    assert isinstance(result, TranslationString)
    assert result.translation() == translated_result
    assert result.value() == original_result
    assert str(result) == original_result


def test_locale_singular_context(trans):
    context = "singular-context"
    singular = "More about napari with context"

    result = trans._p(context, singular)
    assert result == "Más sobre napari con contexto"


def test_locale_singular_context_with_format(trans):
    context = "singular-context-variables"
    variable = 1
    singular = "More about napari with context and {variable}"

    result = trans._p(context, singular, variable=variable)
    assert result == f"Más sobre napari con contexto y {variable}"


def test_locale_singular_context_deferred_with_format(trans):
    context = "singular-context-variables"
    variable = 1
    singular = "More about napari with context and {variable}"
    original_result = f"More about napari with context and {variable}"

    translated_result = f"Más sobre napari con contexto y {variable}"

    result = trans._p(context, singular, deferred=True, variable=variable)
    assert isinstance(result, TranslationString)
    assert result.translation() == translated_result
    assert result.value() == original_result
    assert str(result) == original_result


def test_locale_plural(trans):
    singular = "I have napari"
    plural = "I have naparis"

    n = 1
    result = trans._n(singular, plural, n=n)
    assert result == "Tengo napari"

    n = 2
    result_plural = trans._n(singular, plural, n=n)
    assert result_plural == "Tengo naparis"


def test_locale_plural_with_format(trans):
    singular = "I have {n} napari with {variable}"
    plural = "I have {n} naparis with {variable}"
    variable = 1

    n = 1
    result = trans._n(singular, plural, n=n, variable=variable)
    expected_result = f"Tengo {n} napari con {variable}"
    assert result == expected_result

    n = 2
    result_plural = trans._n(singular, plural, n=n, variable=variable)
    expected_result_plural = f"Tengo {n} naparis con {variable}"
    assert result_plural == expected_result_plural


def test_locale_plural_deferred_with_format(trans):
    variable = 1
    singular = "I have {n} napari with {variable}"
    plural = "I have {n} naparis with {variable}"

    n = 1
    original_result = singular.format(n=n, variable=variable)
    result = trans._n(singular, plural, n=n, deferred=True, variable=variable)
    expected_result = f"Tengo {n} napari con {variable}"
    assert isinstance(result, TranslationString)
    assert result.translation() == expected_result
    assert result.value() == original_result
    assert str(result) == original_result

    n = 2
    original_result_plural = plural.format(n=n, variable=variable)
    result_plural = trans._n(
        singular, plural, n=n, deferred=True, variable=variable
    )
    expected_result_plural = f"Tengo {n} naparis con {variable}"
    assert isinstance(result, TranslationString)
    assert result_plural.translation() == expected_result_plural
    assert result_plural.value() == original_result_plural
    assert str(result_plural) == original_result_plural


def test_locale_plural_context(trans):
    context = "plural-context"
    singular = "I have napari with context"
    plural = "I have naparis with context"

    n = 1
    result = trans._np(context, singular, plural, n=n)
    assert result == "Tengo napari con contexto"

    n = 2
    result_plural = trans._np(context, singular, plural, n=n)
    assert result_plural == "Tengo naparis con contexto"


def test_locale_plural_context_with_format(trans):
    context = "plural-context-variables"
    singular = "I have {n} napari with {variable} and context"
    plural = "I have {n} naparis with {variable} and context"
    variable = 1

    n = 1
    result = trans._np(context, singular, plural, n=n, variable=variable)
    assert result == f"Tengo {n} napari con {variable} y contexto"

    n = 2
    result_plural = trans._np(
        context, singular, plural, n=n, variable=variable
    )
    assert result_plural == f"Tengo {n} naparis con {variable} y contexto"


def test_locale_plural_context_deferred_with_format(trans):
    context = "plural-context-variables"
    variable = 1
    singular = "I have {n} napari with {variable} and context"
    plural = "I have {n} naparis with {variable} and context"

    n = 1
    original_result = singular.format(n=n, variable=variable)
    result = trans._np(
        context, singular, plural, n=n, deferred=True, variable=variable
    )
    expected_result = f"Tengo {n} napari con {variable} y contexto"

    assert isinstance(result, TranslationString)
    assert result.translation() == expected_result
    assert result.value() == original_result
    assert str(result) == original_result

    n = 2
    original_result_plural = plural.format(n=n, variable=variable)
    result_plural = trans._np(
        context, singular, plural, n=n, deferred=True, variable=variable
    )
    expected_result_plural = f"Tengo {n} naparis con {variable} y contexto"

    assert isinstance(result, TranslationString)
    assert result_plural.translation() == expected_result_plural
    assert result_plural.value() == original_result_plural
    assert str(result_plural) == original_result_plural


# Deferred strings in exceptions
# ------------------------------
def test_exception_string(trans):
    expected_result = "Más sobre napari"
    result = trans._("MORE ABOUT NAPARI", deferred=True)
    assert str(result) != expected_result
    assert str(result) == "MORE ABOUT NAPARI"

    with pytest.raises(ValueError) as err:
        raise ValueError(result)

    assert isinstance(err.value.args[0], TranslationString)


# Test TranslationString
# ----------------------
def test_translation_string_exceptions():
    with pytest.raises(ValueError):
        TranslationString()


def test_bundle_exceptions(trans):
    with pytest.raises(ValueError):
        trans._dnpgettext()


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            'msgid': 'huhu',
        },
        {
            'msgid': 'Convert to {dtype}',
            'dtype': 'uint16',
        },
        {
            'msgid': 'Convert to {dtype}',
            'dtype': 'uint16',
            'deferred': True,
        },
        {
            'msgid': 'Convert to {dtype}',
            'dtype': 'uint16',
            'deferred': False,
        },
        {
            'msgid': 'Convert to {dtype}',
            'msgid_plural': 'Convert to {dtype}s',
            'n': 1,
            'dtype': 'uint16',
        },
        {
            'msgid': 'Convert to {dtype}',
            'msgid_plural': 'Convert to {dtype}s',
            'n': 2,
            'dtype': 'uint16',
        },
    ],
)
def test_deepcopy(kwargs):
    """Object containing translation strings can't bee deep-copied.

    See:
      - https://github.com/napari/napari/issues/2911
      - https://github.com/napari/napari/issues/4736    
    """
    t = TranslationString(**kwargs)
    u = deepcopy(t)
    assert t is not u
    assert t == u
