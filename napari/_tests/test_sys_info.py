from napari.utils.info import sys_info


# vispy use_app tries to start Qt, which can cause segfaults when running
# sys_info on CI unless we provide a pytest Qt app
def test_sys_info(qapp):
    str_info = sys_info()
    assert isinstance(str_info, str)
    assert '<br>' not in str_info
    assert '<b>' not in str_info

    html_info = sys_info(as_html=True)
    assert isinstance(html_info, str)
    assert '<br>' in html_info
    assert '<b>' in html_info
