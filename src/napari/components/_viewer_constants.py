from enum import StrEnum


class CanvasPosition(StrEnum):
    """Canvas overlay position.

    Sets the position of an object in the canvas
            * top_left: Top left of the canvas
            * top_right: Top right of the canvas
            * top_center: Top center of the canvas
            * middle_left: Middle left of the canvas
            * middle_center: Middle center of the canvas
            * middle_right: Middle right of the canvas
            * bottom_right: Bottom right of the canvas
            * bottom_left: Bottom left of the canvas
            * bottom_center: Bottom center of the canvas
    """

    TOP_LEFT = 'top_left'
    TOP_CENTER = 'top_center'
    TOP_RIGHT = 'top_right'
    MIDDLE_LEFT = 'middle_left'
    MIDDLE_CENTER = 'middle_center'
    MIDDLE_RIGHT = 'middle_right'
    BOTTOM_RIGHT = 'bottom_right'
    BOTTOM_CENTER = 'bottom_center'
    BOTTOM_LEFT = 'bottom_left'
