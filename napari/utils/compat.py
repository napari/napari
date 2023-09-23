"""compatibility between newer and older python versions
"""
import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    # in 3.11+, using the below class in an f-string would put the enum name instead of its value
    from enum import Enum

    class StrEnum(str, Enum):
        pass
