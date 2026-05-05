"""Named processing profiles for the analysis pipeline.

A profile is a small dict of constants that override values in
`pipeline.config` for the duration of a single processing job. This lets the
same pipeline code handle different physical setups (regulation gym vs.
mini-hoop demo) without forking the codebase.

The override is applied via `profile_scope(name)`, a context manager that
mutates `pipeline.config` on entry and restores the originals on exit. Wrap
each pipeline run in `with profile_scope(profile_name):` so a crash can never
leave the global config in an overridden state.

Concurrency caveat: the override is module-level, so concurrent jobs would
clash. Current Flask backend processes jobs serially, so this is safe in
practice. If multi-job concurrency becomes a need, switch to a runtime
config object passed through the call stack instead.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Any, Iterator

from . import config


# An empty dict means "no overrides — use whatever config.py defines".
PROFILES: Dict[str, Dict[str, Any]] = {
    "regulation": {},
    "mini": {
        # Mini ball / hoop geometry. Educated initial guesses — tune after
        # first real test on the mini setup.
        "BALL_DIAMETER_M":        0.13,   # ~5 inch mini basketball
        "BALL_MIN_Y":             0.10,   # mini setups sit lower than gym hoops
        "BALL_MIN_Z":             1.00,   # closer-range demo, OK to allow nearer balls
        "BALL_HOOP_Z_TOLERANCE":  3.00,   # smaller scene, tighter Z agreement
        "MAKE_CYLINDER_RADIUS":   0.10,   # mini rim ~0.15 m diameter → ~half radius
    },
}


def available() -> list[str]:
    return sorted(PROFILES.keys())


@contextmanager
def profile_scope(name: str) -> Iterator[None]:
    """Apply named profile's overrides to `config`, restore on exit.

    Raises ValueError if the name isn't a known profile.
    """
    if name not in PROFILES:
        raise ValueError(
            f"Unknown profile {name!r}. Known profiles: {', '.join(available())}"
        )

    overrides = PROFILES[name]
    saved: Dict[str, Any] = {}
    for key in overrides:
        if not hasattr(config, key):
            raise AttributeError(
                f"Profile {name!r} tried to override config.{key}, "
                f"which doesn't exist in pipeline.config"
            )
        saved[key] = getattr(config, key)

    try:
        for key, value in overrides.items():
            setattr(config, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(config, key, value)
