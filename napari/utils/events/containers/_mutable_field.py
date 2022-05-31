from __future__ import annotations

from typing import TYPE_CHECKING

from ....utils.translations import trans

if TYPE_CHECKING:
    from pydantic.fields import ModelField


class MutableFieldMixin:
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field: ModelField):
        """Pydantic validator."""
        from pydantic.utils import sequence_like

        if not sequence_like(v):
            raise TypeError(
                trans._(
                    'Value is not a valid sequence: {value}',
                    deferred=True,
                    value=v,
                )
            )
        if not field.sub_fields:
            return cls(v)

        type_field = field.sub_fields[0]
        errors = []
        for i, v_ in enumerate(v):
            _valid_value, error = type_field.validate(v_, {}, loc=f'[{i}]')
            if error:
                errors.append(error)
        if errors:
            from pydantic import ValidationError

            raise ValidationError(errors, cls)  # type: ignore
        return cls(v)
