from pathlib import Path

from qtpy.QtCore import QPoint, QSize, Qt
from qtpy.QtGui import QMovie
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import napari.resources

from ...utils.translations import trans
from ..widgets.qt_progress_bar import ProgressBar, ProgressBarGroup


class ActivityToggleItem(QWidget):
    """Toggle button for Activity Dialog.

    A progress indicator is displayed when there are active progress
    bars.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self.setLayout(QHBoxLayout())

        self._activityBtn = QToolButton()
        self._activityBtn.setObjectName("QtActivityButton")
        self._activityBtn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._activityBtn.setArrowType(Qt.UpArrow)
        self._activityBtn.setIconSize(QSize(11, 11))
        self._activityBtn.setText(trans._('activity'))
        self._activityBtn.setCheckable(True)

        self._inProgressIndicator = QLabel(trans._("in progress..."), self)
        sp = self._inProgressIndicator.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self._inProgressIndicator.setSizePolicy(sp)
        load_gif = str(Path(napari.resources.__file__).parent / "loading.gif")
        mov = QMovie(load_gif)
        mov.setScaledSize(QSize(18, 18))
        self._inProgressIndicator.setMovie(mov)
        self._inProgressIndicator.hide()

        self.layout().addWidget(self._inProgressIndicator)
        self.layout().addWidget(self._activityBtn)
        self.layout().setContentsMargins(0, 0, 0, 0)


class ActivityDialog(QDialog):
    """Activity Dialog for Napari progress bars."""

    MIN_WIDTH = 250
    MIN_HEIGHT = 185

    def __init__(self, parent=None, toggle_button=None):
        super().__init__(parent)
        self._toggleButton = toggle_button

        self.setObjectName('Activity')
        self.setMinimumWidth(self.MIN_WIDTH)
        self.setMinimumHeight(self.MIN_HEIGHT)
        self.setMaximumHeight(self.MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.setWindowFlags(Qt.SubWindow | Qt.WindowStaysOnTopHint)
        self.setModal(False)

        opacityEffect = QGraphicsOpacityEffect(self)
        opacityEffect.setOpacity(0.8)
        self.setGraphicsEffect(opacityEffect)

        self._baseWidget = QWidget()

        self._activityLayout = QVBoxLayout()
        self._activityLayout.addStretch()
        self._baseWidget.setLayout(self._activityLayout)
        self._baseWidget.layout().setContentsMargins(0, 0, 0, 0)

        self._scrollArea = QScrollArea()
        self._scrollArea.setWidgetResizable(True)
        self._scrollArea.setWidget(self._baseWidget)

        self._titleBar = QLabel()

        title = QLabel('activity', self)
        title.setObjectName('QtCustomTitleLabel')
        title.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        )
        line = QFrame(self)
        line.setObjectName("QtCustomTitleBarLine")
        titleLayout = QHBoxLayout()
        titleLayout.setSpacing(4)
        titleLayout.setContentsMargins(8, 1, 8, 0)
        line.setFixedHeight(1)
        titleLayout.addWidget(line)
        titleLayout.addWidget(title)
        self._titleBar.setLayout(titleLayout)

        self._baseLayout = QVBoxLayout()
        self._baseLayout.addWidget(self._titleBar)
        self._baseLayout.addWidget(self._scrollArea)
        self.setLayout(self._baseLayout)

    def add_progress_bar(self, pbar, nest_under=None):
        """Add progress bar to the activity_dialog, making ProgressBarGroup if needed.

        Check whether pbar is nested and create ProgressBarGroup if it is, removing
        existing separators and creating new ones. Show and start inProgressIndicator
        to highlight to user the existence of a progress bar in the dock even when
        the dock is hidden.

        Parameters
        ----------
        pbar : ProgressBar
            progress bar to add to activity dialog
        nest_under : Optional[ProgressBar]
            parent progress bar pbar should be nested under, by default None
        """
        if nest_under is None:
            self._activityLayout.addWidget(pbar)
        else:
            # this is going to be nested, remove separators
            # as the group will have its own
            parent_pbar = nest_under._pbar
            current_pbars = [parent_pbar, pbar]
            remove_separators(current_pbars)

            parent_widg = parent_pbar.parent()
            # if we are already in a group, add pbar to existing group
            if isinstance(parent_widg, ProgressBarGroup):
                nested_layout = parent_widg.layout()
            # create ProgressBarGroup for this pbar
            else:
                new_group = ProgressBarGroup(nest_under._pbar)
                new_group.destroyed.connect(self.maybe_hide_progress_indicator)
                nested_layout = new_group.layout()
                self._activityLayout.addWidget(new_group)
            new_pbar_index = nested_layout.count() - 1
            nested_layout.insertWidget(new_pbar_index, pbar)

        # show progress indicator and start gif
        self._toggleButton._inProgressIndicator.movie().start()
        self._toggleButton._inProgressIndicator.show()
        pbar.destroyed.connect(self.maybe_hide_progress_indicator)

    def move_to_bottom_right(self, offset=(8, 8)):
        """Position widget at the bottom right edge of the parent."""
        if not self.parent():
            return
        sz = self.parent().size() - self.size() - QSize(*offset)
        self.move(QPoint(sz.width(), sz.height()))

    def maybe_hide_progress_indicator(self):
        """Hide progress indicator when all progress bars have finished."""
        pbars = self._baseWidget.findChildren(ProgressBar)
        pbar_groups = self._baseWidget.findChildren(ProgressBarGroup)

        progress_visible = any([pbar.isVisible() for pbar in pbars])
        progress_group_visible = any(
            [pbar_group.isVisible() for pbar_group in pbar_groups]
        )
        if not progress_visible and not progress_group_visible:
            self._toggleButton._inProgressIndicator.movie().stop()
            self._toggleButton._inProgressIndicator.hide()


def get_pbar(prog, nest_under=None, **kwargs):
    """Adds ProgressBar to viewer ActivityDialog and returns it.

    If nest_under is valid ProgressBar, nests new bar underneath
    parent in a ProgressBarGroup

    Parameters
    ----------
    prog : Progress
        progress iterable this ProgressBar will belong to
    nest_under : Optional[ProgressBar]
        parent ProgressBar to nest under, by default None

    Returns
    -------
    ProgressBar
        progress bar to associate with iterable
    """
    from ..qt_main_window import _QtMainWindow

    current_window = _QtMainWindow.current()
    if current_window is None:
        return
    viewer_instance = current_window.qt_viewer
    activity_dialog = viewer_instance.window()._activity_dialog

    pbar = ProgressBar(**kwargs)
    activity_dialog.add_progress_bar(pbar, nest_under)
    viewer_instance.destroyed.connect(prog.close_pbar)

    return pbar


def remove_separators(current_pbars):
    """Remove any existing line separators from current_pbars
    as they will get a separator from the group

    Parameters
    ----------
    current_pbars : List[ProgressBar]
        parent and new progress bar to remove separators from
    """
    for current_pbar in current_pbars:
        line_widg = current_pbar.findChild(QFrame, "QtCustomTitleBarLine")
        if line_widg:
            current_pbar.layout().removeWidget(line_widg)
            line_widg.hide()
            line_widg.deleteLater()
