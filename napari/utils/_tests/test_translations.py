"""Test the translations API."""

import subprocess
import sys
from pathlib import Path

from ..translations import _get_display_name, _is_valid_locale, translator

TEST_LOCALE = "es_CO"
HERE = Path(__file__).parent
TEST_LANGUAGE_PACK_PATH = HERE / "napari-language-pack-es_CO"


def test_locale_valid():
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
    trans = translator.load()

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


def test_is_valid_locale_valid():
    assert _is_valid_locale("en")
    assert _is_valid_locale("es")
    assert _is_valid_locale("es_CO")


def test_is_valid_locale_invalid():
    assert not _is_valid_locale("foo_SPAM")
    assert not _is_valid_locale("bar")


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
