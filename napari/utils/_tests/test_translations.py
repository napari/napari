"""Test the translations API."""

import sys
from pathlib import Path

import pytest

from napari.utils.translations import (
    _get_display_name,
    _is_valid_locale,
    get_language_packs,
    translator,
)

TEST_LOCALE = "es_CO"
HERE = Path(__file__).parent
TEST_LANGUAGE_PACK_PATH = HERE / "napari-language-pack-es_CO"
PY37_OR_LOWER = sys.version_info[:2] <= (3, 7)


es_CO_po = r"""msgid ""
msgstr ""
"Project-Id-Version: \n"
"POT-Creation-Date: 2021-02-18 19:00\n"
"PO-Revision-Date:  2021-02-18 19:00\n"
"Language-Team: \n"
"MIME-Version: 1.0\n"
"Content-Type: text/plain; charset=UTF-8\n"
"Content-Transfer-Encoding: 8bit\n"
"X-Generator: Poedit 2.4.2\n"
"Last-Translator: \n"
"Plural-Forms: nplurals=2; plural=(n != 1);\n"
"Language: es_CO\n"

#: /
msgid "MORE ABOUT NAPARI"
msgstr "Más sobre napari"
"""

es_CO_mo = (
    b"\xde\x12\x04\x95\x00\x00\x00\x00\x02\x00\x00\x00\x1c\x00\x00\x00,"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00<\x00"
    b"\x00\x00\x11\x00\x00\x00=\x00\x00\x00[\x01\x00\x00O\x00\x00\x00\x11"
    b"\x00\x00\x00\xab\x01\x00\x00\x00"
    b"MORE ABOUT NAPARI\x00Project-Id-Version:  \n"
    b"Report-Msgid-Bugs-To: EMAIL@ADDRESS\n"
    b"POT-Creation-Date: 2021-02-18 19:00+0000\n"
    b"PO-Revision-Date: 2021-02-18 19:00+0000\n"
    b"Last-Translator: \nLanguage: es_CO\nLanguage-Team: \n"
    b"Plural-Forms: nplurals=2; plural=(n != 1)\nMIME-Version: 1.0\n"
    b"Content-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: 8bit"
    b"\nGenerated-By: Babel 2.9.0\n\x00M\xc3\xa1s sobre napari\x00"
)


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


def test_locale_valid_singular(trans):
    # Test singular method
    expected_result = "Más sobre napari"
    result = trans.gettext("MORE ABOUT NAPARI")
    assert result == expected_result

    # Test singular method shorthand
    result = trans._("MORE ABOUT NAPARI")
    assert result == expected_result


def test_locale_invalid():
    with pytest.warns(UserWarning):
        translator._set_locale(TEST_LOCALE)
        trans = translator.load()
        result = trans._("BOO")
        assert result == "BOO"


def test_locale_n_runs(trans):
    # Test plural method
    n = 2
    string = "MORE ABOUT NAPARI"
    plural = "MORE ABOUT NAPARIS"
    result = trans.ngettext(string, plural, n)
    assert result == plural

    # Test plural method shorthand
    result = trans._n(string, plural, n)
    assert result == plural


def test_locale_p_runs(trans):
    # Test context singular method
    context = "context"
    string = "MORE ABOUT NAPARI"
    py37_result = "Más sobre napari"
    result = trans.pgettext(context, string)

    # Python 3.7 or lower does not offer translations based on context
    # so pgettext, or npgettext are not available. We fallback to the
    # singular and plural versions without context. For these cases:
    # `pgettext` falls back to `gettext` and `npgettext` to `gettext`
    if PY37_OR_LOWER:
        assert result == py37_result
    else:
        assert result == string

    # Test context singular method shorthand
    result = trans._p(context, string)
    if PY37_OR_LOWER:
        assert result == py37_result
    else:
        assert result == string


def test_locale_np_runs(trans):
    # Test plural context method
    n = 2
    context = "context"
    string = "MORE ABOUT NAPARI"
    plural = "MORE ABOUT NAPARIS"
    result = trans.npgettext(context, string, plural, n)
    assert result == plural

    # Test plural context method shorthand
    result = trans._np(context, string, plural, n)
    assert result == plural
