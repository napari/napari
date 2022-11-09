#!/usr/bin/env python

"""Humanizing functions for numbers."""

import re
from fractions import Fraction

from napari._vendor.experimental.humanize.src.humanize.i18n import gettext as _
from napari._vendor.experimental.humanize.src.humanize.i18n import gettext_noop as N_
from napari._vendor.experimental.humanize.src.humanize.i18n import pgettext as P_


def ordinal(value):
    """Converts an integer to its ordinal as a string.

    For example, 1 is "1st", 2 is "2nd", 3 is "3rd", etc. Works for any integer or
    anything `int()` will turn into an integer. Anything other value will have nothing
    done to it.

    Args:
        value (int, str, float): Integer to convert.

    Returns:
        str: Ordinal string.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    t = (
        P_("0", "th"),
        P_("1", "st"),
        P_("2", "nd"),
        P_("3", "rd"),
        P_("4", "th"),
        P_("5", "th"),
        P_("6", "th"),
        P_("7", "th"),
        P_("8", "th"),
        P_("9", "th"),
    )
    if value % 100 in (11, 12, 13):  # special case
        return f"{value}{t[0]}"
    return f"{value}{t[value % 10]}"


def intcomma(value, ndigits=None):
    """Converts an integer to a string containing commas every three digits.

    For example, 3000 becomes "3,000" and 45000 becomes "45,000". To maintain some
    compatibility with Django's `intcomma`, this function also accepts floats.

    Args:
        value (int, float, str): Integer or float to convert.
        ndigits (int, None): Digits of precision for rounding after the decimal point.

    Returns:
        str: string containing commas every three digits.
    """
    try:
        if isinstance(value, str):
            float(value.replace(",", ""))
        else:
            float(value)
    except (TypeError, ValueError):
        return value

    if ndigits:
        orig = "{0:.{1}f}".format(value, ndigits)
    else:
        orig = str(value)

    new = re.sub(r"^(-?\d+)(\d{3})", r"\g<1>,\g<2>", orig)
    if orig == new:
        return new
    else:
        return intcomma(new)


powers = [10 ** x for x in (6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 100)]
human_powers = (
    N_("million"),
    N_("billion"),
    N_("trillion"),
    N_("quadrillion"),
    N_("quintillion"),
    N_("sextillion"),
    N_("septillion"),
    N_("octillion"),
    N_("nonillion"),
    N_("decillion"),
    N_("googol"),
)


def intword(value, format="%.1f"):
    """Converts a large integer to a friendly text representation.

    Works best for numbers over 1 million. For example, 1_000_000 becomes "1.0 million",
    1200000 becomes "1.2 million" and "1_200_000_000" becomes "1.2 billion". Supports up
    to decillion (33 digits) and googol (100 digits).

    Args:
        value (int, float, str): Integer to convert.
        format (str): To change the number of decimal or general format of the number
            portion.

    Returns:
        str: Friendly text representation as a string, unless the value passed could not
        be coaxed into an `int`.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value

    if value < powers[0]:
        return str(value)
    for ordinal, power in enumerate(powers[1:], 1):
        if value < power:
            chopped = value / float(powers[ordinal - 1])
            if float(format % chopped) == float(10 ** 3):
                chopped = value / float(powers[ordinal])
                return (" ".join([format, _(human_powers[ordinal])])) % chopped
            else:
                return (" ".join([format, _(human_powers[ordinal - 1])])) % chopped
    return str(value)


def apnumber(value):
    """Converts an integer to Associated Press style.

    Args:
        value (int, float, str): Integer to convert.

    Returns:
        str: For numbers 0-9, the number spelled out. Otherwise, the number. This always
        returns a string unless the value was not `int`-able, unlike the Django filter.
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    if not 0 <= value < 10:
        return str(value)
    return (
        _("zero"),
        _("one"),
        _("two"),
        _("three"),
        _("four"),
        _("five"),
        _("six"),
        _("seven"),
        _("eight"),
        _("nine"),
    )[value]


def fractional(value):
    """Convert to fractional number.

    There will be some cases where one might not want to show ugly decimal places for
    floats and decimals.

    This function returns a human-readable fractional number in form of fractions and
    mixed fractions.

    Pass in a string, or a number or a float, and this function returns:

    * a string representation of a fraction
    * or a whole number
    * or a mixed fraction

    Examples:
        ```pycon
        >>> fractional(0.3)
        '3/10'
        >>> fractional(1.3)
        '1 3/10'
        >>> fractional(float(1/3))
        '1/3'
        >>> fractional(1)
        '1'
        ```
    Args:
        value (int, float, str): Integer to convert.

    Returns:
        str: Fractional number as a string.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    whole_number = int(number)
    frac = Fraction(number - whole_number).limit_denominator(1000)
    numerator = frac._numerator
    denominator = frac._denominator
    if whole_number and not numerator and denominator == 1:
        # this means that an integer was passed in
        # (or variants of that integer like 1.0000)
        return f"{whole_number:.0f}"
    elif not whole_number:
        return f"{numerator:.0f}/{denominator:.0f}"
    else:
        return f"{whole_number:.0f} {numerator:.0f}/{denominator:.0f}"


def scientific(value, precision=2):
    """Return number in string scientific notation z.wq x 10ⁿ.

    Examples:
        ```pycon
        >>> scientific(float(0.3))
        '3.00 x 10⁻¹'
        >>> scientific(int(500))
        '5.00 x 10²'
        ```

    Args:
        value (int, float, str): Input number.
        precision (int): Number of decimal for first part of the number.

    Returns:
        str: Number in scientific notation z.wq x 10ⁿ.
    """
    exponents = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "+": "⁺",
        "-": "⁻",
    }
    negative = False
    try:
        if "-" in str(value):
            value = str(value).replace("-", "")
            negative = True

        if isinstance(value, str):
            value = float(value)

        fmt = "{:.%se}" % str(int(precision))
        n = fmt.format(value)

    except (ValueError, TypeError):
        return value

    part1, part2 = n.split("e")
    if "-0" in part2:
        part2 = part2.replace("-0", "-")

    if "+0" in part2:
        part2 = part2.replace("+0", "")

    new_part2 = []
    if negative:
        new_part2.append(exponents["-"])

    for char in part2:
        new_part2.append(exponents[char])

    final_str = part1 + " x 10" + "".join(new_part2)

    return final_str
