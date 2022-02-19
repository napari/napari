from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, List, NamedTuple

from ._fields import Version

if TYPE_CHECKING:
    from ._napari_settings import NapariSettings


class Migration(NamedTuple):
    from_: Version
    to_: Version
    run: Callable[[NapariSettings], None]


MIGRATIONS: List[Migration] = []


def do_migrations(model: NapariSettings):
    # TODO: make atomic?
    for migration in sorted(MIGRATIONS, key=lambda m: m[0]):
        if model.schema_version == migration.from_:
            with mutation_allowed(model):
                migration.run(model)
                model.schema_version = migration.to_


@contextmanager
def mutation_allowed(obj: NapariSettings):
    config = obj.__config__
    prev, config.allow_mutation = config.allow_mutation, True
    try:
        yield
    finally:
        config.allow_mutation = prev


def migrator(
    from_: str, to_: str
) -> Callable[
    [Callable[[NapariSettings], None]], Callable[[NapariSettings], None]
]:
    def decorator(migrate_func: Callable[[NapariSettings], None]):
        m = Migration(Version.parse(from_), Version.parse(to_), migrate_func)
        MIGRATIONS.append(m)

    return decorator


@migrator('0.3.0', '0.4.0')
def _(model: NapariSettings):
    from importlib.metadata import distributions

    # prior to v0.4.0, npe2 plugins were automatically added to
    # disabled plugins.
    for dist in distributions():
        for ep in dist.entry_points:
            if ep.group == "napari.manifest":
                model.plugins.disabled_plugins.discard(dist.metadata['Name'])
