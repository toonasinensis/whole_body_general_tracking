from __future__ import annotations

import os
import sys


def _append_after_stdlib(path: str) -> None:
    if not path or not os.path.isdir(path):
        return
    normalized = os.path.abspath(path)
    sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry or os.curdir) != normalized]
    sys.path.append(normalized)


for item in os.environ.get("OMNIRETARGET_EXTRA_SITE_PACKAGES", "").split(os.pathsep):
    _append_after_stdlib(item)
