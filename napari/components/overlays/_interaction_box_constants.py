class Box:
    """Box: Constants associated with the vertices of the interaction box"""

    # List of points to include for vertices
    WITH_HANDLE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    WITHOUT_HANDLE = [0, 2, 4, 6]
    # List of points for drawing line. Top center appears twice to close box.
    LINE_HANDLE = [7, 6, 4, 2, 0, 7, 9]
    LINE = [0, 2, 4, 6, 0]
    TOP_LEFT = 0
    TOP_CENTER = 7
    LEFT_CENTER = 1
    BOTTOM_RIGHT = 4
    BOTTOM_LEFT = 2
    CENTER = 8
    HANDLE = 9
    LEN = 9
    LEN_WITHOUT_HANDLE = 8
