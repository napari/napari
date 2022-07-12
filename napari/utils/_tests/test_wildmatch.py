from ..wildmatch import Match, score_key, score_pattern_specificity, wildmatch


def test_match_literal():
    assert wildmatch('a', 'a')
    assert not wildmatch('a', 'b')
    assert wildmatch('\\(', '(')
    assert not wildmatch('abc', 'a')


def test_match_any():
    assert wildmatch('?', 'b')
    assert wildmatch('a?c', 'abc')
    assert not wildmatch('a?c', 'abdc')
    assert wildmatch('file_??', 'file_01')


def test_match_star_simple():
    assert wildmatch('*', 'abc')
    assert not wildmatch('*', 'a/bc')
    assert wildmatch('a*d', 'ad')
    assert wildmatch('a*d', 'abcd')
    assert wildmatch('a*cd', 'abcd')
    assert wildmatch('a*cd', 'acdeabcd')
    assert not wildmatch('a*d', 'abcde')
    assert wildmatch('a*d', 'abcdabcd')
    assert not wildmatch('a*d', 'abcdeabcde')
    assert wildmatch('file_*', 'file_100')
    assert wildmatch('a*', 'a')


def test_match_star_complex():
    assert wildmatch('a/*/d', 'a/bc/d')
    assert wildmatch('a/b*g/z', 'a/bcdefg/z')
    assert not wildmatch('a/*/d', 'a/b/c/d')
    assert not wildmatch('a/*/d', 'a/d')


def test_match_doublestar_collapse():
    assert wildmatch('a/**/d', 'a/d')
    assert wildmatch('a/**/d', 'a/bc/d')
    assert wildmatch('a/**/d', 'a/b/c/d')
    assert not wildmatch('a/**/d', 'a/b/c/d/e')
    assert wildmatch('a/**/d/e', 'a/b/c/d/e')
    assert wildmatch('a/**/d', 'a/b/c/d/a/b/c/d')
    assert not wildmatch('a/**/d', 'a/b/c/d/a/b/c/d/e')
    assert not wildmatch('a/**/b', 'a/')
    assert not wildmatch('a/**/', 'a/b')


def test_match_doublestar_expand():
    assert wildmatch('a/b**/e', 'a/b/c/d/e')
    assert wildmatch('a/b**/e', 'a/bc/d/e')
    assert not wildmatch('a/b**/e', 'a/bc/de')
    assert wildmatch('a/b**e', 'a/b/c/d/e')
    assert not wildmatch('a/**c/e', 'a/b/c/d/e')
    assert wildmatch('**.zarr', 'a/b/c/d/efg.zarr')
    assert not wildmatch('a/**b/', 'a/')
    assert wildmatch('a**', 'a')
    assert wildmatch('**', 'a/b/c')


def test_match_set():
    assert wildmatch('[abcd]', 'c')
    assert not wildmatch('[abcd]', 'f')

    assert wildmatch('[a-c]', 'c')
    assert not wildmatch('[a-c]', 'd')
    assert wildmatch('[0-9]', '0')
    assert not wildmatch('[3-5]', '1')
    assert wildmatch('[!3-5]', '1')
    assert wildmatch('[!0-9]', 'A')
    assert not wildmatch('[!0-9]', '9')

    assert wildmatch('[[:space:]]', ' ')
    assert wildmatch('[[:]', '[')
    assert wildmatch('[[:]', ':')
    assert wildmatch('[[:abc]', 'c')
    assert not wildmatch('[[:abc]', 'd')
    assert wildmatch('[[:graph:]]', 'A')
    assert wildmatch('[[:upper:][:space:]]', ' ')
    assert wildmatch('[[:upper:][:space:]]', 'A')
    assert wildmatch('[[:cntrl:]]', '\0')
    assert wildmatch('[[:cntrl:]]', '\x1F')
    assert wildmatch('[[:cntrl:]]', '\x0F')
    assert wildmatch('[[:cntrl:]]', '\x7F')
    assert not wildmatch('[[:cntrl:]]', '\x5F')


def test_match_paren():
    assert wildmatch('a(?bc)', 'a')
    assert wildmatch('a(?bc)', 'abc')
    assert wildmatch('a(?b)c', 'ac')
    assert wildmatch('a(?b)c', 'abc')
    assert not wildmatch('a(?b)c', 'a')


def test_match_paren_subpatt():
    assert wildmatch('(?[a-f])', 'd')
    assert not wildmatch('(?[a-f])', 'z')
    assert wildmatch('(?[a-f])', '')

    assert wildmatch('a(?*)b', 'ab')
    assert wildmatch('a(?b*)c', 'ac')
    assert wildmatch('a(?b*)c', 'abc')
    assert wildmatch('a(?b*)c', 'abwer2341243c')
    assert not wildmatch('a(?b*)c', 'abcd')


def test_match_folder():
    assert wildmatch('**.zarr/', 'a/b/c/d/efg.zarr/')
    assert not wildmatch('**.zarr/', 'a/b/c/d/efg.zarr')

    assert wildmatch('*/', 'abcdefg.zarr/')
    assert not wildmatch('*/', 'abcdefg.zarr')

    assert wildmatch('**.zarr(?/)', 'a/b/c/d/efg.zarr')
    assert wildmatch('**.zarr(?/)', 'a/b/c/d/efg.zarr/')

    assert wildmatch('**/', 'a/b/c/d/efg.zarr/')
    assert wildmatch('a**/', 'a/b/c/d/efg.zarr/')

    assert not wildmatch('a/**/', 'a/')
    assert wildmatch('a/**/', 'a/b/c/d/efg.zarr/')

    assert not wildmatch('a/**/', 'a/c/d/e/b')
    assert wildmatch('a/**/', 'a/b/c/d/e/')


def test_match_casefold():
    assert wildmatch('abcde', 'AbCdE', casefold=True)
    assert not wildmatch('abcde', 'AbCdE', casefold=False)

    assert not wildmatch('\\Dockerfile', 'dockerfile', casefold=True)
    assert wildmatch('\\Dockerfile', 'Dockerfile', casefold=True)

    assert wildmatch('[a-z]', 'g', casefold=False)
    assert not wildmatch('[a-z]', 'G', casefold=False)
    assert wildmatch('[a-zA-Z]', 'G', casefold=False)
    assert not wildmatch('[a-zA-Z]', '0', casefold=False)

    assert not wildmatch('[[:upper:][:space:]]', 'a', casefold=False)
    assert wildmatch('[[:upper:][:space:]]', 'a', casefold=True)


def test_match_any_depth():
    assert wildmatch('*.zarr', 'a/b/c/d/efg.zarr', any_depth=True)
    assert wildmatch('.zarr', 'a/b/c/d/efg.zarr', any_depth=True)
    assert not wildmatch('/*.zarr', 'a/b/c/d/efg.zarr', any_depth=True)
    assert wildmatch('!/*.zarr', 'a/b/c/d/efg.zarr', any_depth=True)


def test_specificity_individual():
    assert score_pattern_specificity('\\*') == [Match.NONE]
    assert score_pattern_specificity('(?abc)') == [Match.MAYBE]
    assert score_pattern_specificity('[:upper:]') == [Match.SET]
    assert score_pattern_specificity('a*') == [Match.STAR]
    assert score_pattern_specificity('**') == [Match.DOUBLESTAR]


def test_specificity_complex():
    assert score_pattern_specificity('**abc') == [Match.DOUBLESTAR, Match.STAR]
    assert score_pattern_specificity('**abc/d') == [
        Match.DOUBLESTAR,
        Match.STAR,
        Match.NONE,
    ]
    assert score_pattern_specificity('(?ab*c)') == [Match.STAR | Match.MAYBE]

    assert score_pattern_specificity('(?ab/c)') == [Match.MAYBE]
    assert score_pattern_specificity('/abc') == [Match.NONE, Match.NONE]

    assert score_pattern_specificity('**/**/**/**a(?b)c') == [
        Match.DOUBLESTAR,
        Match.STAR | Match.MAYBE,
    ]
    assert score_pattern_specificity('****foo') == [
        Match.DOUBLESTAR,
        Match.STAR,
    ]


def test_sort_by_specificity():
    paths = [
        'foo',
        '/foo',
        '/foo/',
        '/foo/bar',
        '/foo/bar/',
        '/foo/*.txt',
        '*.txt',
        'foo/bar',
        'foo/*.txt',
        '**foo',
        '*foo',
    ]
    paths_ordered = [
        '/foo/bar/',
        '/foo/',
        '/foo/bar',
        '/foo/*.txt',
        '/foo',
        'foo/bar',
        'foo/*.txt',
        'foo',
        '*.txt',
        '**foo',
        '*foo',
    ]

    assert sorted(paths, key=score_key) == paths_ordered
