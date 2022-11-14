from __future__ import annotations

import warnings
from contextlib import contextmanager
from importlib.metadata import distributions
from typing import TYPE_CHECKING, Callable, List, NamedTuple

from napari.settings._fields import Version

if TYPE_CHECKING:
    from napari.settings._napari_settings import NapariSettings

_MIGRATORS: List[Migrator] = []
MigratorF = Callable[['NapariSettings'], None]


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
                    msg = (
                        f"Failed to migrate settings from v{migration.from_} "
                        f"to v{migration.to_}. Error: {e}. "
                    )
                    try:
                        model.update(backup)
                        msg += 'You may need to reset your settings with `napari --reset`. '
                    except Exception:
                        msg += 'Settings rollback also failed. Please run `napari --reset`.'
                    warnings.warn(msg)
                    return
    model._maybe_save()


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

    def decorator(migrate_func: MigratorF) -> MigratorF:
        _from, _to = Version.parse(from_), Version.parse(to_)
        assert _to >= _from, 'Migrator must increase the version.'
        _MIGRATORS.append(Migrator(_from, _to, migrate_func))
        return migrate_func

    return decorator


@migrator('0.3.0', '0.4.0')
def v030_v040(model: NapariSettings):
    """Migrate from v0.3.0 to v0.4.0.

    Prior to v0.4.0, npe2 plugins were automatically added to disabled plugins.
    This migration removes any npe2 plugins discovered in the environment
    (at migration time) from the "disabled plugins" set.
    """
    for dist in distributions():
        for ep in dist.entry_points:
            if ep.group == "napari.manifest":
                model.plugins.disabled_plugins.discard(dist.metadata['Name'])


@migrator('0.4.0', '0.5.0')
def v040_050(model: NapariSettings):
    """Migrate from v0.4.0 to v0.5.0

    Prior to 0.5.0 existing preferences may have reader extensions
    preferences saved without a leading *.
    fnmatch would fail on these so we coerce them to include a *
    e.g. '.csv' becomes '*.csv'
    """
    from napari.settings._utils import _coerce_extensions_to_globs

    current_settings = model.plugins.extension2reader
    new_settings = _coerce_extensions_to_globs(current_settings)
    model.plugins.extension2reader = new_settings
