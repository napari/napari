from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from app_model.expressions import Constant, Expr
from app_model.types import CommandRule

from napari._app_model import get_app

_R = TypeVar('_R')


@dataclass
class Command(Generic[_R]):
    """A command representation."""

    command_rule: CommandRule
    title: str
    desc: str
    tooltip: str = ''
    enablement: Expr = field(default=Constant(True))

    def exec(self) -> _R:
        app = get_app()
        return app.commands.execute_command(self.command_rule.id).result()

    def fmt(self) -> str:
        """Format command for display in the palette."""
        if self.title:
            return f'{self.title} > {self.desc}'
        return self.desc

    def match_score(self, input_text: str) -> float:
        """Return a match score (between 0 and 1) for the input text."""
        fmt = self.fmt().lower()
        if all(word in fmt for word in input_text.lower().split(' ')):
            return 1.0
        return 0.0

    def enabled(self, context) -> bool:
        """Return True if the command is enabled."""
        try:
            return self.enablement.eval(context)
        except NameError:
            return False
