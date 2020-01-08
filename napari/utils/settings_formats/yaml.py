import yaml


def write_yaml(filename, dct):
    with open(filename, 'w') as fp:
        yaml.safe_dump(dct, fp)


def read_yaml(filename):
    with open(filename, 'r') as fp:
        _yaml = yaml.safe_load(fp)
    return _yaml


YAML_FORMAT = ("yaml", read_yaml, write_yaml)
