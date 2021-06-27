# humanize

[![PyPI version](https://img.shields.io/pypi/v/humanize.svg?logo=pypi&logoColor=FFE873)](https://pypi.org/project/humanize/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/humanize.svg?logo=python&logoColor=FFE873)](https://pypi.org/project/humanize/)
[![Documentation Status](https://readthedocs.org/projects/python-humanize/badge/?version=latest)](https://python-humanize.readthedocs.io/en/latest/?badge=latest)
[![PyPI downloads](https://img.shields.io/pypi/dm/humanize.svg)](https://pypistats.org/packages/humanize)
[![Travis CI Status](https://travis-ci.com/hugovk/humanize.svg?branch=master)](https://travis-ci.com/hugovk/humanize)
[![GitHub Actions status](https://github.com/jmoiron/humanize/workflows/Test/badge.svg)](https://github.com/jmoiron/humanize/actions)
[![codecov](https://codecov.io/gh/hugovk/humanize/branch/master/graph/badge.svg)](https://codecov.io/gh/hugovk/humanize)
[![MIT License](https://img.shields.io/github/license/jmoiron/humanize.svg)](LICENCE)
[![Tidelift](https://tidelift.com/badges/package/pypi/humanize)](https://tidelift.com/subscription/pkg/pypi-humanize?utm_source=pypi-humanize&utm_medium=badge)

This modest package contains various common humanization utilities, like turning
a number into a fuzzy human readable duration ("3 minutes ago") or into a human
readable size or throughput. It is localized to:

* Brazilian Portuguese
* Dutch
* European Portuguese
* Finnish
* French
* German
* Indonesian
* Italian
* Japanese
* Korean
* Persian
* Polish
* Russian
* Simplified Chinese
* Slovak
* Spanish
* Turkish
* Ukrainian
* Vietnamese

## Usage

### Integer humanization

```pycon
>>> import humanize
>>> humanize.intcomma(12345)
'12,345'
>>> humanize.intword(123455913)
'123.5 million'
>>> humanize.intword(12345591313)
'12.3 billion'
>>> humanize.apnumber(4)
'four'
>>> humanize.apnumber(41)
'41'
```

### Date & time humanization

```pycon
>>> import humanize
>>> import datetime as dt
>>> humanize.naturalday(dt.datetime.now())
'today'
>>> humanize.naturaldelta(dt.timedelta(seconds=1001))
'16 minutes'
>>> humanize.naturalday(dt.datetime.now() - dt.timedelta(days=1))
'yesterday'
>>> humanize.naturalday(dt.date(2007, 6, 5))
'Jun 05'
>>> humanize.naturaldate(dt.date(2007, 6, 5))
'Jun 05 2007'
>>> humanize.naturaltime(dt.datetime.now() - dt.timedelta(seconds=1))
'a second ago'
>>> humanize.naturaltime(dt.datetime.now() - dt.timedelta(seconds=3600))
'an hour ago'
```

### Precise time delta

```pycon
>>> import humanize
>>> import datetime as dt
>>> delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
>>> humanize.precisedelta(delta)
'2 days, 1 hour and 33.12 seconds'
>>> humanize.precisedelta(delta, minimum_unit="microseconds")
'2 days, 1 hour, 33 seconds and 123 milliseconds'
>>> humanize.precisedelta(delta, suppress=["days"], format="%0.4f")
'49 hours and 33.1230 seconds'
```

#### Smaller units

If seconds are too large, set `minimum_unit` to milliseconds or microseconds:

```pycon
>>> import humanize
>>> import datetime as dt
>>> humanize.naturaldelta(dt.timedelta(seconds=2))
'2 seconds'
```
```pycon
>>> delta = dt.timedelta(milliseconds=4)
>>> humanize.naturaldelta(delta)
'a moment'
>>> humanize.naturaldelta(delta, minimum_unit="milliseconds")
'4 milliseconds'
>>> humanize.naturaldelta(delta, minimum_unit="microseconds")
'4000 microseconds'
```
```pycon
>>> humanize.naturaltime(delta)
'now'
>>> humanize.naturaltime(delta, minimum_unit="milliseconds")
'4 milliseconds ago'
>>> humanize.naturaltime(delta, minimum_unit="microseconds")
'4000 microseconds ago'
```

### File size humanization

```pycon
>>> import humanize
>>> humanize.naturalsize(1_000_000)
'1.0 MB'
>>> humanize.naturalsize(1_000_000, binary=True)
'976.6 KiB'
>>> humanize.naturalsize(1_000_000, gnu=True)
'976.6K'
```

### Human-readable floating point numbers

```pycon
>>> import humanize
>>> humanize.fractional(1/3)
'1/3'
>>> humanize.fractional(1.5)
'1 1/2'
>>> humanize.fractional(0.3)
'3/10'
>>> humanize.fractional(0.333)
'333/1000'
>>> humanize.fractional(1)
'1'
```

### Scientific notation

```pycon
>>> import humanize
>>> humanize.scientific(0.3)
'3.00 x 10⁻¹'
>>> humanize.scientific(500)
'5.00 x 10²'
>>> humanize.scientific("20000")
'2.00 x 10⁴'
>>> humanize.scientific(1**10)
'1.00 x 10⁰'
>>> humanize.scientific(1**10, precision=1)
'1.0 x 10⁰'
>>> humanize.scientific(1**10, precision=0)
'1 x 10⁰'
```

## Localization

How to change locale at runtime:

```pycon
>>> import humanize
>>> import datetime as dt
>>> humanize.naturaltime(dt.timedelta(seconds=3))
'3 seconds ago'
>>> _t = humanize.i18n.activate("ru_RU")
>>> humanize.naturaltime(dt.timedelta(seconds=3))
'3 секунды назад'
>>> humanize.i18n.deactivate()
>>> humanize.naturaltime(dt.timedelta(seconds=3))
'3 seconds ago'
```

You can pass additional parameter `path` to `activate` to specify a path to search
locales in.

```pycon
>>> import humanize
>>> humanize.i18n.activate("xx_XX")
<...>
FileNotFoundError: [Errno 2] No translation file found for domain: 'humanize'
>>> humanize.i18n.activate("pt_BR", path="path/to/my/portuguese/translation/")
<gettext.GNUTranslations instance ...>
```

How to add new phrases to existing locale files:

```console
$ xgettext --from-code=UTF-8 -o humanize.pot -k'_' -k'N_' -k'P_:1c,2' -l python src/humanize/*.py  # extract new phrases
$ msgmerge -U src/humanize/locale/ru_RU/LC_MESSAGES/humanize.po humanize.pot # add them to locale files
$ msgfmt --check -o src/humanize/locale/ru_RU/LC_MESSAGES/humanize{.mo,.po} # compile to binary .mo
```

How to add a new locale:

```console
$ msginit -i humanize.pot -o humanize/locale/<locale name>/LC_MESSAGES/humanize.po --locale <locale name>
```

Where `<locale name>` is a locale abbreviation, eg. `en_GB`, `pt_BR` or just `ru`, `fr`
etc.

List the language at the top of this README.
