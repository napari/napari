from qtpy.QtGui import QValidator
from qtpy.QtWidgets import QSpinBox


class QtSpinBox(QSpinBox):
    """Extends QSpinBox validate and stepBy methods in order
    to skip values in spin box."""

    prohibit = None

    def setProhibitValue(self, value: int):
        """Set value that should not be used in QSpinBox.

        Parameters
        ----------
        value : int
            Value to be excluded from QSpinBox.
        """
        self.prohibit = value

    def validate(self, value: str, pos: int):
        if value == str(self.prohibit):
            return QValidator.Invalid, value, pos
        return super().validate(value, pos)

    def stepBy(self, steps: int) -> None:
        if self.value() + steps == self.prohibit:
            steps *= 2
        return super().stepBy(steps)
