from napari.utils.events.event import WarningEmitter
from napari.utils.migrations import _warning_category_for, deprecation_message


def deprecation_warning_event(
    *,
    prefix: str,
    previous_name: str,
    new_name: str,
    since_version: str,
    window: str | None = None,
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
    window : str, optional
        Removal window as a ``YYYY-QN`` token, meaning the event may be removed
        as early as that quarter. Omit it for a soft deprecation.
    since_version : str
        Version when new event name was added.

    Returns
    -------
    WarningEmitter
        Event emitter that prints a deprecation warning.
    """
    previous_path = f'{prefix}.{previous_name}'
    new_path = f'{prefix}.{new_name}'
    return WarningEmitter(
        deprecation_message(
            name=previous_path,
            replacement=new_path,
            since=since_version,
            window=window,
        ),
        type_name=previous_name,
        category=_warning_category_for(window),
    )
