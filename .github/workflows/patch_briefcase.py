import sys


if sys.platform == 'darwin':
    from dmgbuild import core

    with open(core.__file__, 'r') as f:
        source = f.read()
    source = source.replace('max(total_size / 1024', 'max(total_size / 1000')
    with open(core.__file__, 'w') as f:
        f.write(source)
