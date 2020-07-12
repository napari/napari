from enum import Enum, auto
import re

import numpy as np

from ._text_constants import TextMode


class Anchor(Enum):
    CENTER = auto()


def get_text_anchors(view_data, anchor_pos=Anchor.CENTER) -> np.ndarray:
    if anchor_pos == Anchor.CENTER:
        if view_data.ndim == 2:
            return view_data
        else:
            return np.mean(view_data, axis=0)


def format_text_properties(text: str, n_text: int, properties: dict = {}):

    # If the text value is a property key, the text is the property values
    if text in properties:
        formatted_text = np.array([str(v) for v in properties[text]])
        text_mode = TextMode.PROPERTY
    elif ('{' in text) and ('}' in text):
        format_keys = _get_format_keys(text, properties)
        formatted_text = _format_text_f_string(
            text=text,
            n_text=n_text,
            format_keys=format_keys,
            properties=properties,
        )
        text_mode = TextMode.FORMATTED

    else:
        formatted_text, text_mode = format_text_direct(text, n_text)

    return np.array(formatted_text), text_mode


def format_text_direct(text, n_text: int):
    if isinstance(text, str):
        formatted_text = np.repeat(text, n_text)
    else:
        if len(text) != n_text:
            raise ValueError(
                f'Number of text elements ({len(text)}) should equal the length of the data ({n_text})'
            )

        formatted_text = np.asarray(text)

    text_mode = TextMode.NONE

    return formatted_text, text_mode


def _get_format_keys(text: str, properties: dict):
    format_keys = re.findall('{(.*?)}', text)

    format_keys_in_properties = []

    for format_key in format_keys:
        split_key = format_key.split(':')
        if split_key[0] in properties:
            if len(split_key) == 1:
                format_keys_in_properties.append((split_key[0], ''))
            else:
                format_keys_in_properties.append((split_key[0], split_key[1]))

    return format_keys_in_properties


def _format_text_f_string(
    text: str, n_text: int, format_keys: list, properties: dict
):

    all_formatted_text = []
    for i in range(n_text):
        formatted_text = text
        for format_key in format_keys:
            prop_value = properties[format_key[0]][i]
            string_template = '{' + ':' + format_key[1] + '}'
            formatted_prop_value = string_template.format(prop_value)
            if len(format_key[1]) == 0:
                original_value = '{' + format_key[0] + '}'
            else:
                original_value = (
                    '{' + format_key[0] + ':' + format_key[1] + '}'
                )
            formatted_text = formatted_text.replace(
                original_value, formatted_prop_value
            )
        all_formatted_text.append(formatted_text)

    return all_formatted_text
