import datetime as dt
import importlib

import humanize
import pytest


def test_i18n():
    three_seconds = dt.timedelta(seconds=3)

    assert humanize.naturaltime(three_seconds) == "3 seconds ago"
    assert humanize.ordinal(5) == "5th"

    try:
        humanize.i18n.activate("ru_RU")
        assert humanize.naturaltime(three_seconds) == "3 секунды назад"
        assert humanize.ordinal(5) == "5ый"
    finally:
        humanize.i18n.deactivate()
        assert humanize.naturaltime(three_seconds) == "3 seconds ago"
        assert humanize.ordinal(5) == "5th"


def test_default_locale_path_defined__file__():
    i18n = importlib.import_module("humanize.i18n")
    assert i18n._get_default_locale_path() is not None


def test_default_locale_path_null__file__():
    i18n = importlib.import_module("humanize.i18n")
    i18n.__file__ = None
    assert i18n._get_default_locale_path() is None


def test_default_locale_path_undefined__file__():
    i18n = importlib.import_module("humanize.i18n")
    del i18n.__file__
    i18n._get_default_locale_path() is None


class TestActivate:
    expected_msg = (
        "Humanize cannot determinate the default location of the"
        " 'locale' folder. You need to pass the path explicitly."
    )

    def test_default_locale_path_null__file__(self):
        i18n = importlib.import_module("humanize.i18n")
        i18n.__file__ = None

        with pytest.raises(Exception) as excinfo:
            i18n.activate("ru_RU")
        assert str(excinfo.value) == self.expected_msg

    def test_default_locale_path_undefined__file__(self):
        i18n = importlib.import_module("humanize.i18n")
        del i18n.__file__

        with pytest.raises(Exception) as excinfo:
            i18n.activate("ru_RU")
        assert str(excinfo.value) == self.expected_msg
