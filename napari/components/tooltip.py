from napari.utils.events import EventedModel


class Tooltip(EventedModel):
    """Tooltip showing additional information on the cursor.

    Attributes
    ----------
    visible : bool
        If tooltip is visible or not.
    text : str
        text of tooltip
    """

    # fields
    visible: bool = False
    text: str = ""
