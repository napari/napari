import json
from enum import Enum
from importlib import import_module


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            # for "registered" settings, all that is necessary for the
            # preferences window to work properly is:
            # return obj.value

            # this is a more general solution, but makes the output file
            # uglier...
            return {
                "__enum__": obj.__class__.__name__,
                "module": obj.__class__.__module__,
                "value": obj.value,
            }
        return json.JSONEncoder.default(self, obj)


def write_json(filename, dct):
    with open(filename, 'w') as fp:
        json.dump(dct, fp, cls=MyEncoder)


def as_enum(d):
    if "__enum__" in d:
        cls = getattr(import_module(d["module"]), d["__enum__"])
        return cls(d["value"])
    return d


def read_json(filename):
    with open(filename, 'r') as fp:
        _json = json.load(fp, object_hook=as_enum)
    return _json


JSON_FORMAT = ("json", read_json, write_json)
