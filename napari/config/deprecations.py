import warnings
from typing import Dict

#: This dict is used to mark deprecated config keys.
#:
#: The keys of ``deprecations`` are deprecated config values, and the values
#: are the new namespace for the key.  This deprecations are checked when new
#: keys are added to the config in set()
deprecations: Dict[str, str] = {}


def check_deprecations(key: str, deprecations: dict = deprecations):
    """Check if the provided value has been renamed or removed.

    Parameters
    ----------
    key : str
        The configuration key to check
    deprecations : Dict[str, str]
        The mapping of aliases

    Examples
    --------
    >>> deprecations = {"old_key": "new_key", "invalid": None}
    >>> check_deprecations("old_key", deprecations=deprecations)
    UserWarning: Configuration key "old_key" has been deprecated. Please use "new_key" instead.

    >>> check_deprecations("invalid", deprecations=deprecations)
    Traceback (most recent call last):
        ...
    ValueError: Configuration value "invalid" has been removed

    >>> check_deprecations("another_key", deprecations=deprecations)
    'another_key'

    Returns
    -------
    new: str
        The proper key, whether the original (if no deprecation) or the aliased
        value
    """
    if key in deprecations:
        new = deprecations[key]
        if new:
            warnings.warn(
                'Configuration key "{}" has been deprecated. '
                'Please use "{}" instead'.format(key, new)
            )
            return new
        else:
            raise ValueError(
                'Configuration value "{}" has been removed'.format(key)
            )
    else:
        return key
