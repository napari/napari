from app_model.types import Action
from pydantic import Field


class RepeatableAction(Action):
    repeatable: bool = Field(
        True,
        description="Whether this command is triggered repeatedly when its keybinding is held down.",
    )
