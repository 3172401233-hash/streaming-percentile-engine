"""Deterministic streaming quantile engine.

Public API:
    QuantileEngine  -- partitioned, mergeable weighted quantile sketch
    QuantileResult  -- value + candidate rank error + total weight
    TDigest         -- the underlying single-partition sketch
"""

from .engine import QuantileEngine, QuantileResult, TDigest

__all__ = ["QuantileEngine", "QuantileResult", "TDigest"]
__version__ = "0.1.0"
