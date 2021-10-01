import html

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtWidgets import QWidget

from ...utils.translations import trans

class QTooltipEventFilter(QObject):
    def eventFilter(self, widget: QObject, event: QEvent) -> bool:
        '''
        Tooltip-specific event filter handling the passed Qt object and event.

        This code is used from https://stackoverflow.com/a/46212292 under CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/
        '''
        # If this is a tooltip event...
        if event.type() == QEvent.ToolTipChange:
            # If the target Qt object containing this tooltip is *NOT* a widget,
            # raise a human-readable exception. While this should *NEVER* be the
            # case, edge cases are edge cases because they sometimes happen.
            if not isinstance(widget, QWidget):
                raise ValueError(
                    trans._(
                        'QObject "{widget}" not a widget.',
                        deferred=True,
                        widget=widget,
                    )
                )

            # Tooltip for this widget if any *OR* the empty string otherwise.
            tooltip = widget.toolTip()

            # If this tooltip is both non-empty and not already rich text...
            if tooltip and not Qt.mightBeRichText(tooltip):
                # Convert this plaintext tooltip into a rich text tooltip by:
                #
                # * Escaping all HTML syntax in this tooltip.
                # * Embedding this tooltip in the Qt-specific "<qt>...</qt>" tag.
                tooltip = f'<qt>{html.escape(tooltip)}</qt>'

                # Replace this widget's non-working plaintext tooltip with this
                # working rich text tooltip.
                widget.setToolTip(tooltip)

                # Notify the parent event handler this event has been handled.
                return True

        # Else, defer to the default superclass handling of this event.
        return super().eventFilter(widget, event)
