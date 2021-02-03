#!/usr/bin/env python

"""Time humanizing functions. These are largely borrowed from Django's
`contrib.humanize`."""

import datetime as dt
import math
from enum import Enum
from functools import total_ordering

from .i18n import gettext as _
from .i18n import ngettext

__all__ = [
    "naturaldelta",
    "naturaltime",
    "naturalday",
    "naturaldate",
    "precisedelta",
]


@total_ordering
class Unit(Enum):
    MICROSECONDS = 0
    MILLISECONDS = 1
    SECONDS = 2
    MINUTES = 3
    HOURS = 4
    DAYS = 5
    MONTHS = 6
    YEARS = 7

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


def _now():
    return dt.datetime.now()


def abs_timedelta(delta):
    """Return an "absolute" value for a timedelta, always representing a
    time distance.

    Args:
        delta (datetime.timedelta): Input timedelta.

    Returns:
        datetime.timedelta: Absolute timedelta.
    """
    if delta.days < 0:
        now = _now()
        return now - (now + delta)
    return delta


def date_and_delta(value, *, now=None):
    """Turn a value into a date and a timedelta which represents how long ago it was.

    If that's not possible, return `(None, value)`.
    """
    if not now:
        now = _now()
    if isinstance(value, dt.datetime):
        date = value
        delta = now - value
    elif isinstance(value, dt.timedelta):
        date = now - value
        delta = value
    else:
        try:
            value = int(value)
            delta = dt.timedelta(seconds=value)
            date = now - delta
        except (ValueError, TypeError):
            return None, value
    return date, abs_timedelta(delta)


def naturaldelta(value, months=True, minimum_unit="seconds"):
    """Return a natural representation of a timedelta or number of seconds.

    This is similar to `naturaltime`, but does not add tense to the result.

    Args:
        value (datetime.timedelta): A timedelta or a number of seconds.
        months (bool): If `True`, then a number of months (based on 30.5 days) will be
            used for fuzziness between years.
        minimum_unit (str): The lowest unit that can be used.

    Returns:
        str: A natural representation of the amount of time elapsed.
    """
    tmp = Unit[minimum_unit.upper()]
    if tmp not in (Unit.SECONDS, Unit.MILLISECONDS, Unit.MICROSECONDS):
        raise ValueError(f"Minimum unit '{minimum_unit}' not supported")
    minimum_unit = tmp

    date, delta = date_and_delta(value)
    if date is None:
        return value

    use_months = months

    seconds = abs(delta.seconds)
    days = abs(delta.days)
    years = days // 365
    days = days % 365
    months = int(days // 30.5)

    if not years and days < 1:
        if seconds == 0:
            if minimum_unit == Unit.MICROSECONDS and delta.microseconds < 1000:
                return (
                    ngettext("%d microsecond", "%d microseconds", delta.microseconds)
                    % delta.microseconds
                )
            elif minimum_unit == Unit.MILLISECONDS or (
                minimum_unit == Unit.MICROSECONDS
                and 1000 <= delta.microseconds < 1_000_000
            ):
                milliseconds = delta.microseconds / 1000
                return (
                    ngettext("%d millisecond", "%d milliseconds", milliseconds)
                    % milliseconds
                )
            return _("a moment")
        elif seconds == 1:
            return _("a second")
        elif seconds < 60:
            return ngettext("%d second", "%d seconds", seconds) % seconds
        elif 60 <= seconds < 120:
            return _("a minute")
        elif 120 <= seconds < 3600:
            minutes = seconds // 60
            return ngettext("%d minute", "%d minutes", minutes) % minutes
        elif 3600 <= seconds < 3600 * 2:
            return _("an hour")
        elif 3600 < seconds:
            hours = seconds // 3600
            return ngettext("%d hour", "%d hours", hours) % hours
    elif years == 0:
        if days == 1:
            return _("a day")
        if not use_months:
            return ngettext("%d day", "%d days", days) % days
        else:
            if not months:
                return ngettext("%d day", "%d days", days) % days
            elif months == 1:
                return _("a month")
            else:
                return ngettext("%d month", "%d months", months) % months
    elif years == 1:
        if not months and not days:
            return _("a year")
        elif not months:
            return ngettext("1 year, %d day", "1 year, %d days", days) % days
        elif use_months:
            if months == 1:
                return _("1 year, 1 month")
            else:
                return (
                    ngettext("1 year, %d month", "1 year, %d months", months) % months
                )
        else:
            return ngettext("1 year, %d day", "1 year, %d days", days) % days
    else:
        return ngettext("%d year", "%d years", years) % years


def naturaltime(value, future=False, months=True, minimum_unit="seconds"):
    """Return a natural representation of a time in a resolution that makes sense.

    This is more or less compatible with Django's `naturaltime` filter.

    Args:
        value (datetime.datetime, int): A `datetime` or a number of seconds.
        future (bool): Ignored for `datetime`s, where the tense is always figured out
            based on the current time. For integers, the return value will be past tense
            by default, unless future is `True`.
        months (bool): If `True`, then a number of months (based on 30.5 days) will be
            used for fuzziness between years.
        minimum_unit (str): The lowest unit that can be used.

    Returns:
        str: A natural representation of the input in a resolution that makes sense.
    """
    now = _now()
    date, delta = date_and_delta(value, now=now)
    if date is None:
        return value
    # determine tense by value only if datetime/timedelta were passed
    if isinstance(value, (dt.datetime, dt.timedelta)):
        future = date > now

    ago = _("%s from now") if future else _("%s ago")
    delta = naturaldelta(delta, months, minimum_unit)

    if delta == _("a moment"):
        return _("now")

    return ago % delta


def naturalday(value, format="%b %d"):
    """For date values that are tomorrow, today or yesterday compared to
    present day returns representing string. Otherwise, returns a string
    formatted according to `format`."""
    try:
        value = dt.date(value.year, value.month, value.day)
    except AttributeError:
        # Passed value wasn't date-ish
        return value
    except (OverflowError, ValueError):
        # Date arguments out of range
        return value
    delta = value - dt.date.today()
    if delta.days == 0:
        return _("today")
    elif delta.days == 1:
        return _("tomorrow")
    elif delta.days == -1:
        return _("yesterday")
    return value.strftime(format)


def naturaldate(value):
    """Like `naturalday`, but append a year for dates more than about five months away.
    """
    try:
        value = dt.date(value.year, value.month, value.day)
    except AttributeError:
        # Passed value wasn't date-ish
        return value
    except (OverflowError, ValueError):
        # Date arguments out of range
        return value
    delta = abs_timedelta(value - dt.date.today())
    if delta.days >= 5 * 365 / 12:
        return naturalday(value, "%b %d %Y")
    return naturalday(value)


def _quotient_and_remainder(value, divisor, unit, minimum_unit, suppress):
    """Divide `value` by `divisor` returning the quotient and
    the remainder as follows:

    If `unit` is `minimum_unit`, makes the quotient a float number
    and the remainder will be zero. The rational is that if unit
    is the unit of the quotient, we cannot
    represent the remainder because it would require a unit smaller
    than the minimum_unit.

    >>> from humanize.time import _quotient_and_remainder, Unit
    >>> _quotient_and_remainder(36, 24, Unit.DAYS, Unit.DAYS, [])
    (1.5, 0)

    If unit is in suppress, the quotient will be zero and the
    remainder will be the initial value. The idea is that if we
    cannot use unit, we are forced to use a lower unit so we cannot
    do the division.

    >>> _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [Unit.DAYS])
    (0, 36)

    In other case return quotient and remainder as `divmod` would
    do it.

    >>> _quotient_and_remainder(36, 24, Unit.DAYS, Unit.HOURS, [])
    (1, 12)

    """

    if unit == minimum_unit:
        return (value / divisor, 0)
    elif unit in suppress:
        return (0, value)
    else:
        return divmod(value, divisor)


def _carry(value1, value2, ratio, unit, min_unit, suppress):
    """Return a tuple with two values as follows:

    If the unit is in suppress multiplies value1
    by ratio and add it to value2 (carry to right).
    The idea is that if we cannot represent value1 we need
    to represent it in a lower unit.

    >>> from humanize.time import _carry, Unit
    >>> _carry(2, 6, 24, Unit.DAYS, Unit.SECONDS, [Unit.DAYS])
    (0, 54)

    If the unit is the minimum unit, value2 is divided
    by ratio and added to value1 (carry to left).
    We assume that value2 has a lower unit so we need to
    carry it to value1.

     >>> _carry(2, 6, 24, Unit.DAYS, Unit.DAYS, [])
     (2.25, 0)

    Otherwise, just return the same input:

    >>> _carry(2, 6, 24, Unit.DAYS, Unit.SECONDS, [])
    (2, 6)
    """
    if unit == min_unit:
        return (value1 + value2 / ratio, 0)
    elif unit in suppress:
        return (0, value2 + value1 * ratio)
    else:
        return (value1, value2)


def _suitable_minimum_unit(min_unit, suppress):
    """Return a minimum unit suitable that is not suppressed.

    If not suppressed, return the same unit:

    >>> from humanize.time import _suitable_minimum_unit, Unit
    >>> _suitable_minimum_unit(Unit.HOURS, [])
    <Unit.HOURS: 4>

    But if suppressed, find a unit greather than the original one
    that is not suppressed:

    >>> _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS])
    <Unit.DAYS: 5>

    >>> _suitable_minimum_unit(Unit.HOURS, [Unit.HOURS, Unit.DAYS])
    <Unit.MONTHS: 6>
    """
    if min_unit in suppress:
        for unit in Unit:
            if unit > min_unit and unit not in suppress:
                return unit

        raise ValueError(
            "Minimum unit is suppressed and no suitable replacement was found"
        )

    return min_unit


def _suppress_lower_units(min_unit, suppress):
    """Extend the suppressed units (if any) with all the units that are
    lower than the minimum unit.

    >>> from humanize.time import _suppress_lower_units, Unit
    >>> list(sorted(_suppress_lower_units(Unit.SECONDS, [Unit.DAYS])))
    [<Unit.MICROSECONDS: 0>, <Unit.MILLISECONDS: 1>, <Unit.DAYS: 5>]
    """
    suppress = set(suppress)
    for u in Unit:
        if u == min_unit:
            break
        suppress.add(u)

    return suppress


def precisedelta(value, minimum_unit="seconds", suppress=(), format="%0.2f"):
    """Return a precise representation of a timedelta.

    ```pycon
    >>> import datetime as dt
    >>> from humanize.time import precisedelta

    >>> delta = dt.timedelta(seconds=3633, days=2, microseconds=123000)
    >>> precisedelta(delta)
    '2 days, 1 hour and 33.12 seconds'
    ```

    A custom `format` can be specified to control how the fractional part
    is represented:

    ```pycon
    >>> precisedelta(delta, format="%0.4f")
    '2 days, 1 hour and 33.1230 seconds'
    ```

    Instead, the `minimum_unit` can be changed to have a better resolution;
    the function will still readjust the unit to use the greatest of the
    units that does not lose precision.

    For example setting microseconds but still representing the date with milliseconds:

    ```pycon
    >>> precisedelta(delta, minimum_unit="microseconds")
    '2 days, 1 hour, 33 seconds and 123 milliseconds'
    ```

    If desired, some units can be suppressed: you will not see them represented and the
    time of the other units will be adjusted to keep representing the same timedelta:

    ```pycon
    >>> precisedelta(delta, suppress=['days'])
    '49 hours and 33.12 seconds'
    ```

    Note that microseconds precision is lost if the seconds and all
    the units below are suppressed:

    ```pycon
    >>> delta = dt.timedelta(seconds=90, microseconds=100)
    >>> precisedelta(delta, suppress=['seconds', 'milliseconds', 'microseconds'])
    '1.50 minutes'
    ```
    """

    date, delta = date_and_delta(value)
    if date is None:
        return value

    suppress = [Unit[s.upper()] for s in suppress]

    # Find a suitable minimum unit (it can be greater the one that the
    # user gave us if it is suppressed).
    min_unit = Unit[minimum_unit.upper()]
    min_unit = _suitable_minimum_unit(min_unit, suppress)
    del minimum_unit

    # Expand the suppressed units list/set to include all the units
    # that are below the minimum unit
    suppress = _suppress_lower_units(min_unit, suppress)

    # handy aliases
    days = delta.days
    secs = delta.seconds
    usecs = delta.microseconds

    MICROSECONDS, MILLISECONDS, SECONDS, MINUTES, HOURS, DAYS, MONTHS, YEARS = list(
        Unit
    )

    # Given DAYS compute YEARS and the remainder of DAYS as follows:
    #   if YEARS is the minimum unit, we cannot use DAYS so
    #   we will use a float for YEARS and 0 for DAYS:
    #       years, days = years/days, 0
    #
    #   if YEARS is suppressed, use DAYS:
    #       years, days = 0, days
    #
    #   otherwise:
    #       years, days = divmod(years, days)
    #
    # The same applies for months, hours, minutes and milliseconds below
    years, days = _quotient_and_remainder(days, 365, YEARS, min_unit, suppress)
    months, days = _quotient_and_remainder(days, 30.5, MONTHS, min_unit, suppress)

    # If DAYS is not in suppress, we can represent the days but
    # if it is a suppressed unit, we need to carry it to a lower unit,
    # seconds in this case.
    #
    # The same applies for secs and usecs below
    days, secs = _carry(days, secs, 24 * 3600, DAYS, min_unit, suppress)

    hours, secs = _quotient_and_remainder(secs, 3600, HOURS, min_unit, suppress)
    minutes, secs = _quotient_and_remainder(secs, 60, MINUTES, min_unit, suppress)

    secs, usecs = _carry(secs, usecs, 1e6, SECONDS, min_unit, suppress)

    msecs, usecs = _quotient_and_remainder(
        usecs, 1000, MILLISECONDS, min_unit, suppress
    )

    # if _unused != 0 we had lost some precision
    usecs, _unused = _carry(usecs, 0, 1, MICROSECONDS, min_unit, suppress)

    fmts = [
        ("%d year", "%d years", years),
        ("%d month", "%d months", months),
        ("%d day", "%d days", days),
        ("%d hour", "%d hours", hours),
        ("%d minute", "%d minutes", minutes),
        ("%d second", "%d seconds", secs),
        ("%d millisecond", "%d milliseconds", msecs),
        ("%d microsecond", "%d microseconds", usecs),
    ]

    texts = []
    for unit, fmt in zip(reversed(Unit), fmts):
        singular_txt, plural_txt, value = fmt
        if value > 0:
            fmt_txt = ngettext(singular_txt, plural_txt, value)
            if unit == min_unit and math.modf(value)[0] > 0:
                fmt_txt = fmt_txt.replace("%d", format)

            texts.append(fmt_txt % value)

        if unit == min_unit:
            break

    if len(texts) == 1:
        return texts[0]

    head = ", ".join(texts[:-1])
    tail = texts[-1]

    return " and ".join((head, tail))
