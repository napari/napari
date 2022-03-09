from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, List, NamedTuple

from ._fields import Version

if TYPE_CHECKING:
    from ._napari_settings import NapariSettings

_MIGRATORS: List[Migrator] = []
MigratorF = Callable[[NapariSettings], None]


class Migrator(NamedTuple):
    """Tuple of from-version, to-version, migrator function."""

    from_: Version
    to_: Version
    run: MigratorF


def do_migrations(model: NapariSettings):
    """Migrate (update) a NapariSettings model in place."""
    for migration in sorted(_MIGRATORS, key=lambda m: m.from_):
        if model.schema_version == migration.from_:
            with mutation_allowed(model):
                backup = model.dict()
                try:
                    migration.run(model)
                    model.schema_version = migration.to_
                except Exception as e:
                    model.update(backup)
                    warnings.warn(
                        f"Failed to migrate settings from v{migration.from_} "
                        f"to v{migration.to_}. You may need to reset your "
                        f"settings with `napari --reset`. Error: {e}"
                    )
                    return


@contextmanager
def mutation_allowed(obj: NapariSettings):
    """Temporarily allow mutations on an immutable model."""
    config = obj.__config__
    prev, config.allow_mutation = config.allow_mutation, True
    try:
        yield
    finally:
        config.allow_mutation = prev


def migrator(from_: str, to_: str) -> Callable[[MigratorF], MigratorF]:
    """Decorate function as migrating settings from v `from_` to v `to_`.

    A migrator should mutate a `NapariSettings` model from schema version
    `from_` to schema version `to_` (in place).

    Parameters
    ----------
    from_ : str
        NapariSettings.schema_version version that this migrator expects as
        input
    to_ : str
        NapariSettings.schema_version version after this migrator has been
        executed.

    Returns
    -------
    Callable[ [MigratorF], MigratorF ]
        _description_
    """

    def decorator(migrate_func: MigratorF):
        m = Migrator(Version.parse(from_), Version.parse(to_), migrate_func)
        _MIGRATORS.append(m)

    return decorator


@migrator('0.3.0', '0.4.0')
def _(model: NapariSettings):
    """Migrate from v0.3.0 to v0.4.0.

    Prior to v0.4.0, npe2 plugins were automatically added to disabled plugins.
    This migration removes any npe2 plugins discovered in the environment
    (at migration time) from the "disabled plugins" set.
    """
    from importlib.metadata import distributions

    for dist in distributions():
        for ep in dist.entry_points:
            if ep.group == "napari.manifest":
                model.plugins.disabled_plugins.discard(dist.metadata['Name'])
