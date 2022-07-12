"""Wildmatching based off of git's implementation.

See https://github.com/git/git/blob/master/wildmatch.c
"""

import string
from enum import IntFlag

LITERAL_ERROR = "invalid pattern: no character followed by literal symbol '\\'"
BRACKET_ERROR = "invalid pattern: no closing bracket ']'"
PAREN_ERROR = "invalid pattern: no closing parenthesis ')'"
PAREN_OP_ERROR = "missing parenthesis operator '?'"

POSIX_CLASSES = {
    'alnum': str.isalnum,
    'alpha': str.isalpha,
    'ascii': str.isascii,
    'blank': lambda c: c == ' ' or c == '\t',
    'cntrl': lambda c: '\x00' <= c <= '\x1F' or c == '\x7F',
    'digit': str.isdigit,
    'graph': lambda c: POSIX_CLASSES['print'](c)
    and not POSIX_CLASSES['space'](c),
    'lower': str.islower,
    'print': str.isprintable,
    'punct': lambda c: c in string.punctuation,
    'space': str.isspace,
    'upper': str.isupper,
    'word': lambda c: c == '_' or POSIX_CLASSES['alnum'](c),
    'xdigit': lambda c: c in string.hexdigits,
}


def _wildmatch(pattern, text, casefold):
    pattern_left = True

    if not pattern and text:
        return False

    p_ch = pattern[(p_i := 0)]

    if len(text) > (t_i := 0):
        t_ch = text[t_i]
    else:
        t_ch = None

    def p():
        nonlocal p_ch, p_i
        if (p_i := p_i + 1) >= len(pattern):
            return (p_ch := None)
        return (p_ch := pattern[p_i])

    def t():
        nonlocal t_ch, t_i
        if (t_i := t_i + 1) >= len(text):
            return (t_ch := None)
        return (t_ch := text[t_i])

    while pattern_left:

        # TODO: replace with case matching in 3.10
        if t_ch is None and p_ch != '*' and p_ch != '(':
            return False

        if p_ch == '\\':
            # literal match, skip to next character
            if p() is None:
                raise ValueError(LITERAL_ERROR)

            # do not apply lower casing in literal comparisons

            if t_ch != p_ch:
                return False
        elif p_ch == '?':
            # matches one character
            if t_ch == '/':
                # that is not /
                return False
        elif p_ch == '*':
            # one asterisk
            doublestar = False
            if p() is None:
                # pattern ends in singular *, match all files non-recursively
                return '/' not in text[t_i:]
            elif p_ch == '*':
                # two or more asterisks
                doublestar = True
                prev_p_i = p_i - 2
                while p() == '*':
                    pass  # advance until next non-asterisk character reached
                if p_ch is None:  # end of pattern, match everything
                    return True
                elif (
                    prev_p_i < 0 or pattern[prev_p_i] == '/'
                ) and p_ch == '/':  # equivalent to /**/
                    if p_i + 1 < len(pattern) and _wildmatch(
                        pattern[p_i + 1 :], text[t_i:], casefold
                    ):
                        # check if **/ can be collapsed as it can match nothing
                        return True

            while t_ch is not None:
                if not doublestar and t_ch == '/':
                    break

                if _wildmatch(pattern[p_i:], text[t_i:], casefold):
                    return True

                t()
            else:
                return False

        elif p_ch == '[':
            p()
            if negated := p_ch == '!':
                p()

            needs_end_bracket = True

            prev_ord = 0
            matched = False

            while needs_end_bracket:
                if p_ch is None:
                    raise ValueError(BRACKET_ERROR)

                if p_ch == '\\':
                    if p() is None:
                        raise ValueError(LITERAL_ERROR)
                    if t_ch == p_ch:
                        matched = True
                elif (
                    p_ch == '-'
                    and prev_ord
                    and p_i + 1 < len(pattern)
                    and pattern[p_i + 1] != ']'
                ):
                    if p() == '\\':
                        if p() is None:
                            raise ValueError(LITERAL_ERROR)
                    if ord(t_ch) <= ord(p_ch) and ord(t_ch) >= prev_ord:
                        matched = True
                    elif casefold and t_ch.islower():
                        t_ch_upper = t_ch.upper()
                        if (
                            ord(t_ch_upper) <= ord(p_ch)
                            and ord(t_ch_upper) >= prev_ord
                        ):
                            matched = True
                    else:
                        p_ch = '\0'  # resets prev_ord to 0
                elif (
                    p_ch == '['
                    and p_i + 1 < len(pattern)
                    and pattern[p_i + 1] == ':'
                ):
                    s = p_i + 2  # skip past '[:'
                    i = pattern.find(']', s)

                    if i < 0:  # end bracket not found
                        raise ValueError(BRACKET_ERROR)

                    if i - s - 1 < 0 or pattern[i - 1] != ':':
                        # didn't find ':]', treat like normal set
                        assert p_ch == '['

                        if t_ch == p_ch:
                            matched = True
                    else:
                        p_i = i

                        if (
                            posix_class := pattern[s : i - 1]
                        ) not in POSIX_CLASSES:
                            raise ValueError(
                                f"invalid pattern: malformed posix class '{posix_class}'"
                            )

                        if casefold and posix_class == 'upper':
                            if t_ch in string.ascii_letters:
                                matched = True
                        elif POSIX_CLASSES[posix_class](t_ch):
                            matched = True

                        p_ch = '\0'  # resets prev_ord to 0

                elif t_ch == p_ch:
                    matched = True

                prev_ord = ord(p_ch)
                needs_end_bracket = p() != ']'

            if matched == negated or t_ch == '/':
                return False

        elif p_ch == '(':
            if p() != '?':
                # TODO: add other operators e.g. +, *, {a,b}
                raise ValueError(PAREN_OP_ERROR)

            # match zero or one times

            patt = ''

            while True:
                if p() is None:
                    raise ValueError(PAREN_ERROR)
                if p_ch == '\\':
                    if p() is None:
                        raise ValueError(LITERAL_ERROR)
                    patt += '\\' + p_ch
                elif p_ch == ')':
                    break
                else:
                    patt += p_ch

            if p() is None and t_i >= len(text):
                return True

            if _wildmatch(pattern[p_i:], text[t_i:], casefold):
                # check for zero time match
                return True

            return _wildmatch(patt + pattern[p_i:], text[t_i:], casefold)

        else:  # default case
            if casefold:
                p_ch = p_ch.lower()
                t_ch = t_ch.lower()

            if t_ch != p_ch:
                return False

        pattern_left = p() is not None
        t()

    return t_i >= len(text)  # is there remaining text?


def wildmatch(pattern, path, *, casefold=True, any_depth=False):
    """Match a pattern against a path. Path delimiters are always '/'.

    Special Characters:
    - ! at the beginning of the pattern negates it
    - \\ marks the following character as a literal
    - ? matches any non-slash character
    - * matches multiple non-slash characters
    - ** matches multiple characters with infinite depth

    Sub-patterns:
    - brackets [] denote a set of characters that can be matched:
        - loose set of characters, e.g. [cake] which matches the characters c, a, k, and e
        - range of characters, e.g. [0-5] which matches all numbers from 0 to 5
        - posix class, e.g. [[:word:]] which match certain pre-defined rules
        - ! at the beginning of the set inverts it, e.g. [!.] matches anything that isn't a period
    - parentheses () denote a certain amount of matches:
        - ? matches 0 or 1 times
        - TODO: * matches 0 or more times
        - TODO: + matches 1 or more times
        - TODO: {a,b} matches between a and b times, e.g. {3,5} matches 3-5 times

    Parameters
    ----------
    pattern : str
        Pattern to match with.
    path : str
        Path to match against.
    casefold : bool, optional, kwonly
        Whether to ignore case when matching, does not apply to sets
        except for the posix class `upper`.
    any_depth : bool, optional, kwonly
        Whether relative paths can be matched at any depth (effectively pre-pends **).

    Returns
    -------
    match : bool
        Whether the pattern matches the path provided.

    Raises
    ------
    ValueError
        When the pattern is invalid.
    """
    if pattern.startswith('!'):
        return not wildmatch(
            pattern[1:], path, casefold=casefold, any_depth=any_depth
        )
    if any_depth and not pattern.startswith('/'):
        pattern = '**' + pattern
    return _wildmatch(pattern, path, casefold)


class Match(IntFlag):
    NONE = 0
    MAYBE = 1
    SET = 2
    ANY = 4
    MULTI_RANGE = 8  # e.g. ({3,5}pattern), to be implemented
    MULTI = 16  # e.g. (+pattern) or (*pattern), to be implemented
    STAR = 32
    DOUBLESTAR = 64


def score_pattern_specificity(pattern):
    """Score a pattern's segments' specificities. Lower score means higher specificity.

    Parameters
    ----------
    pattern : str
        Pattern to score.

    Returns
    -------
    score : list of int
        Specificity scores for each pattern segment.

    Raises
    ------
    ValueError
        When the pattern is invalid.
    """
    if not pattern:
        return []

    score = [Match.NONE]

    p_ch = pattern[(p_i := 0)]

    def p():
        nonlocal p_ch, p_i
        if (p_i := p_i + 1) >= len(pattern):
            return (p_ch := None)
        return (p_ch := pattern[p_i])

    pattern_left = True

    while pattern_left:
        # TODO: replace with case matching in 3.10
        if p_ch == '\\':
            # literal match, skip to next character
            if p() is None:
                raise ValueError(LITERAL_ERROR)

        elif p_ch == '/':
            score.append(Match.NONE)

        elif p_ch == '?':
            # matches one character
            score[-1] |= Match.ANY

        elif p_ch == '*':
            if p() == '*':
                # two or more asterisks
                prev_p_i = p_i - 2
                while p() == '*':
                    pass  # advance until next non-asterisk character reached
                if (
                    not score[-1] & Match.DOUBLESTAR
                    and prev_p_i >= 0
                    and pattern[prev_p_i] == '/'
                ):
                    score.append(Match.NONE)
                score[-1] |= Match.DOUBLESTAR
                if p_ch == '/':
                    s = score_pattern_specificity(pattern[p_i + 1 :])
                    if not s[0] & Match.DOUBLESTAR:
                        score.append(Match.NONE)
                elif p_ch is not None:
                    score.append(Match.STAR)
            else:
                # one asterisk
                score[-1] |= Match.STAR

        elif p_ch == '[':
            if p() == '!':
                p()

            needs_end_bracket = True

            prev_ord = False

            while needs_end_bracket:
                if p_ch is None:
                    raise ValueError(BRACKET_ERROR)

                if p_ch == '\\':
                    if p() is None:
                        raise ValueError(LITERAL_ERROR)
                elif (
                    p_ch == '-'
                    and prev_ord
                    and p_i + 1 < len(pattern)
                    and pattern[p_i + 1] != ']'
                ):
                    if p() == '\\':
                        if p() is None:
                            raise ValueError(LITERAL_ERROR)

                elif (
                    p_ch == '['
                    and p_i + 1 < len(pattern)
                    and pattern[p_i + 1] == ':'
                ):
                    s = p_i + 2  # skip past '[:'
                    i = pattern.find(']', s)

                    if i < 0:  # end bracket not found
                        raise ValueError(BRACKET_ERROR)

                    if i - s - 1 < 0 or pattern[i - 1] != ':':
                        # didn't find ':]', treat like normal set
                        assert p_ch == '['
                    else:
                        p_i = i

                        if (
                            posix_class := pattern[s : i - 1]
                        ) not in POSIX_CLASSES:
                            raise ValueError(
                                f"invalid pattern: malformed posix class '{posix_class}'"
                            )

                prev_ord = True
                needs_end_bracket = p() != ']'

            score[-1] |= Match.SET

        elif p_ch == '(':
            if p() != '?':
                # TODO: add other operators e.g. +, *, {a,b}
                raise ValueError("missing parenthesis operator '?'")

            # match zero or one times

            patt = ''

            while True:
                if p() is None:
                    raise ValueError(PAREN_ERROR)
                if p_ch == '\\':
                    if p() is None:
                        raise ValueError(LITERAL_ERROR)
                    patt += '\\' + p_ch
                elif p_ch == ')':
                    break
                else:
                    patt += p_ch

            score[-1] |= Match.MAYBE

            for e in score_pattern_specificity(patt):
                score[-1] |= e

        pattern_left = p() is not None

    return score


def score_key(pattern):
    """Extracts a comparable specificity score key from a pattern.
    Lower score means more specificity.

    Absolute paths have highest specificity,
    followed by paths with the most nesting,
    then by path segments with the least ambiguity.

    Parameters
    ----------
    pattern : str
        Pattern to score.

    Returns
    -------
    score : list of int
        Comparable score.

    Raises
    ------
    ValueError
        When the pattern is invalid.
    """
    rel_path = False
    if not pattern.startswith('/'):
        pattern = '**/' + pattern
        rel_path = True
    score = score_pattern_specificity(pattern)
    return [rel_path, -len(score)] + score
