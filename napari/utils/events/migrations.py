from napari.utils.events.event import WarningEmitter
from napari.utils.translations import trans


def deprecation_warning_event(
    prefix: str,
    previous_name: str,
    new_name: str,
    version: str,
    since_version: str,
) -> WarningEmitter:
    """
    Helper function for event emitter deprecation warning.

    This event still needs to be added to the events group.

    Parameters
    ----------
    prefix:
        Prefix indicating class and event (e.g. layer.event)
    previous_name : str
        Name of deprecated event (e.g. edge_width)
    new_name : str
        Name of new event (e.g. border_width)
    version : str
        Version where deprecated event will be removed.
    since_version : str
        Version when new event name was added.

    Returns
    -------
    WarningEmitter
        Event emitter that prints a deprecation warning.
    """
    previous_path = f"{prefix}.{previous_name}"
    new_path = f"{prefix}.{new_name}"
    return WarningEmitter(
        trans._(
            "{previous_path} is deprecated since {since_version} and will be removed in {version}. Please use {new_path}",
            deferred=True,
            previous_path=previous_path,
            since_version=since_version,
            version=version,
            new_path=new_path,
        ),
        type_name=previous_name,
    )
