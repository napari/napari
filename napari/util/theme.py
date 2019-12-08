# syntax_style for the console must be one of the supported styles from
# pygments - see here for examples https://help.farbox.com/pygments.html

palettes = {
    'dark': {
        'folder': 'dark',
        'background': 'rgb(38, 41, 48)',
        'background_darker': 'rgb(33, 36, 42)',  # 2% darker than background
        'foreground': 'rgb(65, 72, 81)',
        'primary': 'rgb(90, 98, 108)',
        'secondary': 'rgb(134, 142, 147)',
        'highlight': 'rgb(106, 115, 128)',
        'text': 'rgb(240, 241, 242)',
        'icon': 'rgb(209, 210, 212)',
        'warning': 'rgb(153, 18, 31)',
        'syntax_style': 'native',
        'console': 'rgb(0, 0, 0)',
        'canvas': 'black',
    },
    'light': {
        'folder': 'light',
        'background': 'rgb(239, 235, 233)',
        'background_darker': 'rgb(234, 230, 228)',  # 2% darker than background
        'foreground': 'rgb(214, 208, 206)',
        'primary': 'rgb(188, 184, 181)',
        'secondary': 'rgb(150, 146, 144)',
        'highlight': 'rgb(163, 158, 156)',
        'text': 'rgb(59, 58, 57)',
        'icon': 'rgb(107, 105, 103)',
        'warning': 'rgb(255, 18, 31)',
        'syntax_style': 'default',
        'console': 'rgb(255, 255, 255)',
        'canvas': 'white',
    },
}


def template(css, **palette):
    for k, v in palette.items():
        css = css.replace('{{ %s }}' % k, v)
    return css
