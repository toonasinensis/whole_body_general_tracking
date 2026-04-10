"""
Python module serving as a project/extension template.
"""

def _try_register_tasks() -> None:
    """Best-effort task registration.

    Importing `whole_body_tracking` shouldn't hard-require Isaac Sim / Omniverse.
    Tasks are only needed when running the RL environments; utility modules (e.g.
    motion datasets) should remain usable in lightweight Python environments.
    """

    try:
        # Register Gym environments (may require IsaacLab / Omniverse).
        from . import tasks as _tasks  # noqa: F401
    except Exception:
        # Intentionally swallow any import errors so `whole_body_tracking.utils.*`
        # can be imported without heavy runtime dependencies.
        return


# Attempt registration, but keep package import lightweight.
_try_register_tasks()
