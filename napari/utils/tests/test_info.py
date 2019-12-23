from ..info import sys_info, citation_text


def test_sys_info():
    str_info = sys_info()
    assert isinstance(str_info, str)
    assert '<br>' not in str_info
    assert '<b>' not in str_info

    html_info = sys_info(as_html=True)
    assert isinstance(html_info, str)
    assert '<br>' in html_info
    assert '<b>' in html_info


def test_citation_text():
    assert isinstance(citation_text, str)
    assert 'doi' in citation_text
