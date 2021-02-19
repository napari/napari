"""Test the translations API."""

import subprocess
import sys
from pathlib import Path

import pytest

from ...utils import translations
from ..translations import (
    _get_display_name,
    _is_valid_locale,
    _run_process_and_parse,
    get_installed_packages_locales,
    translator,
)

TEST_LOCALE = "es_CO"
HERE = Path(__file__).parent
TEST_LANGUAGE_PACK_PATH = HERE / "napari-language-pack-es_CO"


@pytest.fixture(scope="module")
def trans():
    # Install test language package
    p = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"{TEST_LANGUAGE_PACK_PATH}",
            # "--force-reinstall",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    print(stdout, stderr)

    # Compile the *.po catalog to a *.mo file
    import napari_language_pack_es_CO

    package_dir = Path(napari_language_pack_es_CO.__file__).parent
    p = subprocess.Popen(
        [
            "pybabel",
            "compile",
            "--domain=napari",
            f"--dir={package_dir / 'locale'}",
            f"--locale={TEST_LOCALE}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = p.communicate()
    print(stdout, stderr)

    print(
        " ".join(
            [
                "pybabel",
                "compile",
                "--domain=napari",
                f"--dir={package_dir / 'locale'}",
                f"--locale={TEST_LOCALE}",
            ]
        )
    )
    # Load translator and force a locale for testing
    translator._set_locale(TEST_LOCALE)
    return translator.load()


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


def test_run_process_and_parse(trans):
    cmd = [
        sys.executable,
        translations.__file__,
        "_get_installed_language_pack_locales",
    ]
    data, msg = _run_process_and_parse(cmd)
    assert data == {'es_CO': 'napari_language_pack_es_CO'}
    assert msg == ""


def test_get_installed_packages_locales(trans):
    data, msg = get_installed_packages_locales()
    assert data == {'es_CO': 'napari_language_pack_es_CO'}
    assert msg == ""


def test_locale_valid_singular(trans):
    # Test singular method
    expected_result = "Más sobre napari"
    result = trans.gettext("MORE ABOUT NAPARI")
    assert result == expected_result

    # Test singular method shorthand
    result = trans._("MORE ABOUT NAPARI")
    assert result == expected_result


def test_locale_invalid():
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
    result = trans.pgettext(context, string)
    assert result == string

    # Test context singular method shorthand
    result = trans._p(context, string)
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
