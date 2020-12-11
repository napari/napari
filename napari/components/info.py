from ..utils.events.dataclass import evented_dataclass


@evented_dataclass
class ViewerInfo:
    """Object modeling basic information for the viewer.

    Attributes
    ----------
    help : str
        Help string displayed in the bottom right of the viewer status bar.
    status : str
        Status string displayed in the bottom left of the viewer status bar.
    title : str
        Title of the viewer window.
    """

    help: str = ''
    status: str = 'Ready'
    title: str = 'napari'
