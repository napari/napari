import re
from dataclasses import dataclass
from functools import total_ordering
from typing import Any, SupportsInt

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from napari.utils.logo import available_logos
from napari.utils.theme import available_themes, is_theme_available
from napari.utils.translations import _load_language, get_language_packs, trans


class StrField(str):
    # https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types

    __slots__ = ()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        json_schema = handler(core_schema)
        json_schema.update(enum=cls._available_options())
        return json_schema

    @classmethod
    def validate(cls, v):
        if not isinstance(v, str):
            raise TypeError(trans._('must be a string', deferred=True))

        value = v.lower()
        if not cls._valid_option(v):
            raise ValueError(
                trans._(
                    '"{value}" is not valid. It must be one of {options}',
                    deferred=True,
                    value=value,
                    options=', '.join(cls._available_options()),
                )
            )

        return value

    @classmethod
    def _available_options(cls):
        raise NotImplementedError

    @classmethod
    def _valid_option(cls, v):
        raise NotImplementedError


class Logo(StrField):
    """
    Custom logo type to dynamically load all available logos.
    """

    @classmethod
    def _available_options(cls):
        return available_logos()

    @classmethod
    def _valid_option(cls, v):
        return v in cls._available_options()


class Theme(StrField):
    """
    Custom theme type to dynamically load all installed themes.
    """

    @classmethod
    def _available_options(cls):
        return available_themes()

    @classmethod
    def _valid_option(cls, v):
        return is_theme_available(v)


class Language(StrField):
    """
    Custom theme type to dynamically load all installed language packs.
    """

    @classmethod
    def _available_options(cls):
        return list(get_language_packs(_load_language()).keys())

    @classmethod
    def _valid_option(cls, v):
        return v in cls._available_options()


@total_ordering
@dataclass
class Version:
    """A semver compatible version class.

    mostly vendored from python-semver (BSD-3):
    https://github.com/python-semver/python-semver/
    """

    major: SupportsInt
    minor: SupportsInt = 0
    patch: SupportsInt = 0
    prerelease: bytes | str | int | None = None
    build: bytes | str | int | None = None

    _SEMVER_PATTERN = re.compile(
        r"""
            ^
            (?P<major>0|[1-9]\d*)
            \.
            (?P<minor>0|[1-9]\d*)
            \.
            (?P<patch>0|[1-9]\d*)
            (?:-(?P<prerelease>
                (?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)
                (?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*
            ))?
            (?:\+(?P<build>
                [0-9a-zA-Z-]+
                (?:\.[0-9a-zA-Z-]+)*
            ))?
            $
        """,
        re.VERBOSE,
    )

    @classmethod
    def parse(cls, version: bytes | str) -> 'Version':
        """Convert string or bytes into Version object."""
        if isinstance(version, bytes):
            version = version.decode('UTF-8')
        match = cls._SEMVER_PATTERN.match(version)
        if match is None:
            raise ValueError(
                trans._(
                    '{version} is not valid SemVer string',
                    deferred=True,
                    version=version,
                )
            )
        matched_version_parts: dict[str, Any] = match.groupdict()
        return cls(**matched_version_parts)

    # NOTE: we're only comparing the numeric parts for now.
    # ALSO: the rest of the comparators come  from functools.total_ordering
    def __eq__(self, other) -> bool:
        try:
            return self.to_tuple()[:3] == self._from_obj(other).to_tuple()[:3]
        except TypeError:
            return NotImplemented

    def __lt__(self, other) -> bool:
        try:
            return self.to_tuple()[:3] < self._from_obj(other).to_tuple()[:3]
        except TypeError:
            return NotImplemented

    @classmethod
    def _from_obj(cls, other):
        if isinstance(other, str | bytes):
            other = Version.parse(other)
        elif isinstance(other, dict):
            other = Version(**other)
        elif isinstance(other, tuple | list):
            other = Version(*other)
        elif not isinstance(other, Version):
            raise TypeError(
                trans._(
                    'Expected str, bytes, dict, tuple, list, or {cls} instance, but got {other_type}',
                    deferred=True,
                    cls=cls,
                    other_type=type(other),
                )
            )
        return other

    def to_tuple(self) -> tuple[int, int, int, str | None, str | None]:
        """Return version as tuple (first three are int, last two Opt[str])."""
        return (
            int(self.major),
            int(self.minor),
            int(self.patch),
            str(self.prerelease) if self.prerelease is not None else None,
            str(self.build) if self.build is not None else None,
        )

    def __iter__(self):
        yield from self.to_tuple()

    def __str__(self) -> str:
        v = f'{self.major}.{self.minor}.{self.patch}'
        if self.prerelease:  # pragma: no cover
            v += str(self.prerelease)
        if self.build:  # pragma: no cover
            v += str(self.build)
        return v

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)

    @classmethod
    def validate(cls, v):
        return cls._from_obj(v)

    def _json_encode(self):
        return str(self)
