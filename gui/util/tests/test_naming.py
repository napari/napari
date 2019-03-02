from ..naming import numbered_patt, inc_name_count, sep, start


def test_re_base_num():
    assert numbered_patt.search('layer 12').group(0) == '12'


def test_re_base_empty():
    assert numbered_patt.search('layer').group(0) == ''


def test_re_other_digs():
    assert numbered_patt.search('layer3 123').group(0) == '123'


def test_re_no_space():
    assert numbered_patt.search('layer7').group(0) == '7'


def test_re_first_dig():
    assert numbered_patt.search('42').group(0) == '42'


def test_re_sub_base_num():
    assert numbered_patt.sub('8', 'layer 7') == 'layer 8'


def test_re_sub_base_empty():
    assert numbered_patt.sub(' 3', 'layer') == 'layer 3'


def test_inc_name_count():
    assert inc_name_count('layer 7') == 'layer 8'
    assert inc_name_count('layer') == f'layer{sep}{start}'
    assert inc_name_count('41') == '42'
