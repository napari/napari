from pathlib import Path

from qtpy.QtCore import QPoint, QSize, Qt
from qtpy.QtGui import QMovie
from qtpy.QtWidgets import (
    QApplication,
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
from napari._qt.widgets.qt_progress_bar import (
    QtLabeledProgressBar,
    QtProgressBarGroup,
)
from napari.utils.progress import progress
from napari.utils.translations import trans


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
        self._activityBtn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._activityBtn.setArrowType(Qt.ArrowType.UpArrow)
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


class QtActivityDialog(QDialog):
    """Activity Dialog for Napari progress bars."""

    MIN_WIDTH = 250
    MIN_HEIGHT = 185

    def __init__(self, parent=None, toggle_button=None) -> None:
        super().__init__(parent)
        self._toggleButton = toggle_button

        self.setObjectName('Activity')
        self.setMinimumWidth(self.MIN_WIDTH)
        self.setMinimumHeight(self.MIN_HEIGHT)
        self.setMaximumHeight(self.MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.setWindowFlags(
            Qt.WindowType.SubWindow | Qt.WindowType.WindowStaysOnTopHint
        )
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
        self.resize(520, self.MIN_HEIGHT)
        self.move_to_bottom_right()

        # TODO: what do we do with any existing progress objects in action?
        # connect callback to handle new progress objects being added/removed
        progress._all_instances.events.changed.connect(
            self.handle_progress_change
        )

    def handle_progress_change(self, event):
        """Handle addition and/or removal of new progress objects

        Parameters
        ----------
        event : Event
            EventedSet `changed` event with `added` and `removed` objects
        """
        for prog in event.removed:
            self.close_progress_bar(prog)
        for prog in event.added:
            self.make_new_pbar(prog)

    def make_new_pbar(self, prog):
        """Make new `QtLabeledProgressBar` for this `progress` object and add to viewer.

        Parameters
        ----------
        prog : progress
            progress object to associated with new progress bar
        """
        prog.gui = True
        prog.leave = False

        # make and add progress bar
        pbar = QtLabeledProgressBar(prog=prog)
        self.add_progress_bar(pbar, nest_under=prog.nest_under)

        # connect progress object events to updating progress bar
        prog.events.value.connect(pbar._set_value)
        prog.events.description.connect(pbar._set_description)
        prog.events.overflow.connect(pbar._make_indeterminate)
        prog.events.eta.connect(pbar._set_eta)
        prog.events.total.connect(pbar._set_total)

        # connect pbar close method if we're closed
        self.destroyed.connect(prog.close)

        # set its range etc. based on progress object
        if prog.total is not None:
            pbar.setRange(prog.n, prog.total)
            pbar.setValue(prog.n)
        else:
            pbar.setRange(0, 0)
            prog.total = 0
        pbar.setDescription(prog.desc)

    def add_progress_bar(self, pbar, nest_under=None):
        """Add progress bar to activity_dialog,in QtProgressBarGroup if needed.

        Check if pbar needs nesting and create QtProgressBarGroup, removing
        existing separators and creating new ones. Show and start
        inProgressIndicator to highlight the existence of a progress bar
        in the dock even when the dock is hidden.

        Parameters
        ----------
        pbar : QtLabeledProgressBar
            progress bar to add to activity dialog
        nest_under : Optional[progress]
            parent `progress` whose QtLabeledProgressBar we need to nest under
        """
        if nest_under is None:
            self._activityLayout.addWidget(pbar)
        else:
            # TODO: can parent be non gui pbar?
            parent_pbar = self.get_pbar_from_prog(nest_under)
            current_pbars = [parent_pbar, pbar]
            remove_separators(current_pbars)

            parent_widg = parent_pbar.parent()
            # if we are already in a group, add pbar to existing group
            if isinstance(parent_widg, QtProgressBarGroup):
                nested_layout = parent_widg.layout()
            # create QtProgressBarGroup for this pbar
            else:
                new_group = QtProgressBarGroup(parent_pbar)
                new_group.destroyed.connect(self.maybe_hide_progress_indicator)
                nested_layout = new_group.layout()
                self._activityLayout.addWidget(new_group)
            # progress bar needs to go before separator
            new_pbar_index = nested_layout.count() - 1
            nested_layout.insertWidget(new_pbar_index, pbar)

        # show progress indicator and start gif
        self._toggleButton._inProgressIndicator.movie().start()
        self._toggleButton._inProgressIndicator.show()
        pbar.destroyed.connect(self.maybe_hide_progress_indicator)
        QApplication.processEvents()

    def get_pbar_from_prog(self, prog):
        """Given prog `progress` object, find associated `QtLabeledProgressBar`

        Parameters
        ----------
        prog : progress
            progress object with associated progress bar

        Returns
        -------
        QtLabeledProgressBar
            QtLabeledProgressBar widget associated with this progress object
        """
        if pbars := self._baseWidget.findChildren(QtLabeledProgressBar):
            for potential_parent in pbars:
                if potential_parent.progress is prog:
                    return potential_parent
        return None

    def close_progress_bar(self, prog):
        """Close `QtLabeledProgressBar` and parent `QtProgressBarGroup` if needed

        Parameters
        ----------
        prog : progress
            progress object whose QtLabeledProgressBar to close
        """
        current_pbar = self.get_pbar_from_prog(prog)
        if not current_pbar:
            return
        parent_widget = current_pbar.parent()
        current_pbar.close()
        current_pbar.deleteLater()
        if isinstance(parent_widget, QtProgressBarGroup):
            pbar_children = [
                child
                for child in parent_widget.children()
                if isinstance(child, QtLabeledProgressBar)
            ]
            # only close group if it has no visible progress bars
            if not any(child.isVisible() for child in pbar_children):
                parent_widget.close()

    def move_to_bottom_right(self, offset=(8, 8)):
        """Position widget at the bottom right edge of the parent."""
        if not self.parent():
            return
        sz = self.parent().size() - self.size() - QSize(*offset)
        self.move(QPoint(sz.width(), sz.height()))

    def maybe_hide_progress_indicator(self):
        """Hide progress indicator when all progress bars have finished."""
        pbars = self._baseWidget.findChildren(QtLabeledProgressBar)
        pbar_groups = self._baseWidget.findChildren(QtProgressBarGroup)

        progress_visible = any(pbar.isVisible() for pbar in pbars)
        progress_group_visible = any(
            pbar_group.isVisible() for pbar_group in pbar_groups
        )
        if not progress_visible and not progress_group_visible:
            self._toggleButton._inProgressIndicator.movie().stop()
            self._toggleButton._inProgressIndicator.hide()


def remove_separators(current_pbars):
    """Remove any existing line separators from current_pbars
    as they will get a separator from the group

    Parameters
    ----------
    current_pbars : List[QtLabeledProgressBar]
        parent and new progress bar to remove separators from
    """
    for current_pbar in current_pbars:
        if line_widg := current_pbar.findChild(QFrame, "QtCustomTitleBarLine"):
            current_pbar.layout().removeWidget(line_widg)
            line_widg.hide()
            line_widg.deleteLater()
