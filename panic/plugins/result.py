"""Plugin-defined observations, independent of the plugin registry."""
from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np

@dataclass
class PluginResult:
    score: float
    artifacts: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] | None = None
    row_keys: tuple[str, ...] = ()
    primary_key: tuple[Any, ...] | None = None


def normalize_plugin_result(value):
    if not isinstance(value, PluginResult):
        if isinstance(value, tuple) and len(value) == 2:
            return PluginResult(float(value[0]), dict(value[1]))
        return PluginResult(float(value))
    value.score = float(value.score)
    if value.rows is None:
        if value.row_keys or value.primary_key is not None:
            raise ValueError('Row identity requires explicit rows')
        return value
    if not value.rows or not value.row_keys or len(set(value.row_keys)) != len(value.row_keys):
        raise ValueError('Explicit rows require nonempty rows and unique row_keys')
    seen = set()
    primary = None
    for row in value.rows:
        if not isinstance(row, dict) or 'observed' not in row:
            raise ValueError('Each plugin row requires observed')
        for name, cell in row.items():
            if not isinstance(name, str) or not name or not (cell is None or isinstance(cell, (str, bool, int, float, np.generic))):
                raise ValueError('Plugin rows require named scalar cells')
        row['observed'] = float(row['observed'])
        key = tuple(row[k] for k in value.row_keys)
        if any(k is None or str(k) == '' or (isinstance(k, (float, np.floating)) and not math.isfinite(k)) for k in key):
            raise ValueError('Missing or nonfinite row identity')
        serialized = tuple(str(k) for k in key)
        if serialized in seen:
            raise ValueError(f'Duplicate plugin row identity: {key}')
        seen.add(serialized)
        if key == value.primary_key:
            primary = row['observed']
    if primary is None or not (primary == value.score or (math.isnan(primary) and math.isnan(value.score))):
        raise ValueError('Primary row must exist and match scalar score')
    return value
