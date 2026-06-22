from __future__ import annotations


class FlatDict(dict):
    """Tiny local subset of flatdict.FlatDict used by IsaacLab SimulationContext."""

    def __init__(self, value=None, delimiter=".", *args, **kwargs):
        super().__init__()
        self.delimiter = delimiter
        if value is not None:
            self._flatten(value)
        if args or kwargs:
            self.update(*args, **kwargs)

    def _flatten(self, value, prefix=""):
        for key, item in dict(value).items():
            flat_key = f"{prefix}{self.delimiter}{key}" if prefix else str(key)
            if isinstance(item, dict):
                self._flatten(item, flat_key)
            else:
                self[flat_key] = item
