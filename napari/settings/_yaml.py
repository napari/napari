from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from yaml import Dumper, SafeDumper, dump_all

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import AbstractSet, Any, Dict, Optional, Union

    from pydantic import BaseModel

    IntStr = Union[int, str]
    AbstractSetIntStr = AbstractSet[IntStr]
    DictStrAny = Dict[str, Any]
    MappingIntStrAny = Mapping[IntStr, Any]


class YamlDumper(SafeDumper):
    ...


YamlDumper.add_multi_representer(str, YamlDumper.represent_str)
YamlDumper.add_multi_representer(
    Enum, lambda dumper, data: dumper.represent_str(data.value)
)


class PydanticYamlMixin(BaseModel):
    """Mixin that provides yaml dumping capability to pydantic BaseModel."""

    def yaml(
        self,
        *,
        include: Union[AbstractSetIntStr, MappingIntStrAny] = None,  # type: ignore
        exclude: Union[AbstractSetIntStr, MappingIntStrAny] = None,  # type: ignore
        by_alias: bool = False,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        dumper: Optional[Dumper] = None,
        **dumps_kwargs: Any,
    ) -> str:

        data = self.dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )
        if self.__custom_root_type__:
            from pydantic.utils import ROOT_KEY

            data = data[ROOT_KEY]

        dumper = dumper or getattr(self.__config__, 'yaml_dumper', YamlDumper)
        return dump_all([data], Dumper=dumper, **dumps_kwargs)
