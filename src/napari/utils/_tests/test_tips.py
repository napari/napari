import sys

import pytest

from napari.utils.tips import _link_color, format_tip, urls_to_html


@pytest.mark.parametrize(
    ('raw', 'formatted_linux', 'formatted_win', 'formatted_mac'),
    [
        (
            'press {Meta+K} to jump',
            'press Super+K to jump',
            'press ⊞+K to jump',
            'press ⌘K to jump',
        ),
        ('{Space} or {Alt}', '␣ or Alt', '␣ or Alt', '␣ or ⌥'),  # multiple
        ('{napari.viewer.fit_to_view}', 'Ctrl+0', 'Ctrl+0', '⌘0'),  # appmodel
        ('{napari:reset_view}', 'Ctrl+R', 'Ctrl+R', '⌘R'),  # action manager
    ],
)
def test_format_tips(raw, formatted_linux, formatted_win, formatted_mac):
    match sys.platform:
        case 'linux':
            expected = formatted_linux
        case 'win32':
            expected = formatted_win
        case 'darwin':
            expected = formatted_mac

    assert format_tip(raw) == expected


@pytest.mark.parametrize(
    ('text', 'expected_body'),
    [
        (
            'Check out https://napari.org for more info',
            'Check out <a href="https://napari.org">napari.org</a> for more info',
        ),
        (
            'Visit http://example.com now',
            'Visit <a href="http://example.com">example.com</a> now',
        ),
        (
            'Chat at https://napari.zulipchat.com! Join us!',
            'Chat at <a href="https://napari.zulipchat.com">napari.zulipchat.com</a>! Join us!',
        ),
        (
            'Visit https://forum.image.sc',
            'Visit <a href="https://forum.image.sc">forum.image.sc</a>',
        ),
    ],
)
def test_urls_to_html_converts_urls(text, expected_body):
    theme = 'dark'
    style = f'<style>a{{color:{_link_color(theme)}}}</style>'
    assert urls_to_html(text, theme) == f'{style}{expected_body}'


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        (
            'Just a regular tip without any links',
            'Just a regular tip without any links',
        ),
        (
            'Press Ctrl+> to zoom',
            'Press Ctrl+&gt; to zoom',
        ),
        (
            'A&B Sandwich',
            'A&amp;B Sandwich',
        ),
    ],
)
def test_urls_to_html_no_urls_escaping(text, expected):
    assert urls_to_html(text) == expected


def test_urls_to_html_multiple_urls():
    theme = 'dark'
    style = f'<style>a{{color:{_link_color(theme)}}}</style>'
    result = urls_to_html(
        'Site1: https://napari.org Site2: https://forum.image.sc', theme
    )
    assert 'href="https://napari.org"' in result
    assert 'href="https://forum.image.sc"' in result
    assert style in result


def test_urls_to_html_escapes_around_urls():
    theme = 'dark'
    style = f'<style>a{{color:{_link_color(theme)}}}</style>'
    result = urls_to_html(
        'Use Ctrl+> then visit https://napari.org and see & more', theme
    )
    assert result == (
        f'{style}Use Ctrl+&gt; then visit '
        '<a href="https://napari.org">napari.org</a>'
        ' and see &amp; more'
    )
