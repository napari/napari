from ..utils.theme import available_themes
from ..utils.translations import _load_language, get_language_packs, trans


class Theme(str):
    """
    Custom theme type to dynamically load all installed themes.
    """

    # https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # TODO: Provide a way to handle keys so we can display human readable
        # option in the preferences dropdown
        field_schema.update(enum=available_themes())

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise ValueError(trans._('must be a string', deferred=True))

        value = v.lower()
        themes = available_themes()
        if value not in available_themes():
            raise ValueError(
                trans._(
                    '"{value}" is not valid. It must be one of {themes}',
                    deferred=True,
                    value=value,
                    themes=", ".join(themes),
                )
            )

        return value


class Language(str):
    """
    Custom theme type to dynamically load all installed language packs.
    """

    # https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def __modify_schema__(cls, field_schema):
        # TODO: Provide a way to handle keys so we can display human readable
        # option in the preferences dropdown
        language_packs = list(get_language_packs(_load_language()).keys())
        field_schema.update(enum=language_packs)

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise ValueError(trans._('must be a string', deferred=True))

        language_packs = list(get_language_packs(_load_language()).keys())
        if v not in language_packs:
            raise ValueError(
                trans._(
                    '"{value}" is not valid. It must be one of {language_packs}.',
                    deferred=True,
                    value=v,
                    language_packs=", ".join(language_packs),
                )
            )

        return v


class SchemaVersion(str):
    """
    Custom schema version type to handle both tuples and version strings.

    Provides also a `as_tuple` method for convenience when doing version
    comparison.
    """

    def __new__(cls, value):
        if isinstance(value, (tuple, list)):
            value = ".".join(str(item) for item in value)

        return str.__new__(cls, value)

    def __init__(self, value):
        if isinstance(value, (tuple, list)):
            value = ".".join(str(item) for item in value)

        self._value = value

    def __eq__(self, other):
        if isinstance(other, (tuple, list)):
            other = ".".join(str(item) for item in other)
        return self._value == other

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, (tuple, list)):
            v = ".".join(str(item) for item in v)

        if not isinstance(v, str):
            raise ValueError(
                trans._(
                    "A schema version must be a 3 element tuple or string!",
                    deferred=True,
                )
            )

        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(
                trans._(
                    "A schema version must be a 3 element tuple or string!",
                    deferred=True,
                )
            )

        for part in parts:
            try:
                int(part)
            except Exception:
                raise ValueError(
                    trans._(
                        "A schema version subparts must be positive integers or parseable as integers!",
                        deferred=True,
                    )
                )

        return cls(v)

    def __repr__(self):
        return f'SchemaVersion("{self._value}")'

    def __str__(self):
        return f'"{self._value}"'

    def as_tuple(self):
        return tuple(int(p) for p in self._value.split('.'))
