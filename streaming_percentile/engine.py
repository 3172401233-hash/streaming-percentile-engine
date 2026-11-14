"""Deterministic, mergeable, weighted streaming quantile sketch.

The implementation is a weighted t-digest variant:

* Samples arrive as (value, weight) pairs and are buffered.
* When the buffer fills, it is sorted by value, merged with the existing
  centroids, and greedily re-clustered under a scale-dependent size limit
  (the k1 scale: clusters stay small near the tails, larger near the
  median).  Memory is therefore O(compression), independent of the number
  of samples ingested.
* Merging two sketches feeds the centroids of one into the buffer of the
  other and flushes.  No randomness is used anywhere: every run over the
  same inputs produces byte-identical results.

Input validation policy (explicit, documented behaviour):

* NaN or infinite values            -> ValueError
* NaN, infinite or negative weights -> ValueError
* zero weight                       -> sample silently ignored
* empty sketch / empty partition    -> QuantileResult(value=None, ...)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

__all__ = ["QuantileEngine", "QuantileResult", "TDigest"]


@dataclass(frozen=True)
class QuantileResult:
    """Answer of a quantile query.

    Attributes:
        value:        estimated quantile value, or None when the sketch is
                      empty.
        rank_error:   candidate rank error of ``value``, expressed in weight
                      units.  The true rank of ``value`` in the full weighted
                      sample is expected to lie within
                      ``q * total_weight +/- rank_error``.  This is an
                      approximate guarantee derived from the sketch's cluster
                      structure, not an exact promise.
        total_weight: total weight that was actually ingested (zero-weight
                      samples are not counted).
    """

    value: Optional[float]
    rank_error: float
    total_weight: float

    @property
    def relative_rank_error(self) -> float:
        """Rank error as a fraction of the total weight (0.0 when empty)."""
        if self.total_weight <= 0.0:
            return 0.0
        return self.rank_error / self.total_weight


@dataclass
class _Centroid:
    mean: float
    weight: float


class TDigest:
    """Single-partition weighted t-digest.

    Args:
        compression: accuracy/memory knob.  The number of retained centroids
            grows linearly with ``compression`` and is independent of the
            number of samples.  Larger means more accurate.
    """

    def __init__(self, compression: float = 100.0) -> None:
        compression = float(compression)
        if not math.isfinite(compression) or compression < 1.0:
            raise ValueError("compression must be a finite number >= 1")
        self.compression = compression
        self._centroids: List[_Centroid] = []
        self._buffer: List[Tuple[float, float]] = []
        self._total_weight = 0.0
        # Flush threshold: keeps amortised sorting cheap while bounding the
        # transient memory to O(compression) as well.
        self._buffer_capacity = max(64, int(4 * compression))

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def add(self, value: float, weight: float = 1.0) -> None:
        """Add one weighted sample.  See module docstring for validation."""
        value = float(value)
        weight = float(weight)
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"value must be finite, got {value!r}")
        if math.isnan(weight) or math.isinf(weight) or weight < 0.0:
            raise ValueError(f"weight must be a finite number >= 0, got {weight!r}")
        if weight == 0.0:
            return
        self._buffer.append((value, weight))
        self._total_weight += weight
        if len(self._buffer) >= self._buffer_capacity:
            self._flush()

    def merge(self, other: "TDigest") -> "TDigest":
        """Merge ``other`` into this sketch (in place).  Returns self."""
        if not isinstance(other, TDigest):
            raise TypeError("can only merge another TDigest")
        if other is self:
            raise ValueError("cannot merge a sketch into itself")
        other = other.copy()
        other._flush()
        if other._total_weight == 0.0:
            return self
        for centroid in other._centroids:
            self._buffer.append((centroid.mean, centroid.weight))
        self._total_weight += other._total_weight
        if len(self._buffer) >= self._buffer_capacity:
            self._flush()
        return self

    def copy(self) -> "TDigest":
        clone = TDigest(self.compression)
        clone._centroids = [_Centroid(c.mean, c.weight) for c in self._centroids]
        clone._buffer = list(self._buffer)
        clone._total_weight = self._total_weight
        return clone

    # ------------------------------------------------------------------ #
    # compression
    # ------------------------------------------------------------------ #
    def _flush(self) -> None:
        if not self._buffer:
            return
        points: List[Tuple[float, float]] = [
            (c.mean, c.weight) for c in self._centroids
        ]
        points.extend(self._buffer)
        self._buffer = []
        # Deterministic: pure sort, no shuffling.  Python's sort is stable,
        # so ties on equal means resolve in a reproducible order.
        points.sort(key=lambda p: p[0])

        total = self._total_weight
        compression = self.compression

        def max_cluster_weight(q0: float) -> float:
            # k1 scale function from the t-digest paper:
            #   k(q) = compression / (2*pi) * asin(2q - 1)
            # A cluster starting at quantile q0 may extend until k grows by
            # one unit.  This keeps clusters tiny near the tails and bounds
            # the total number of centroids to roughly compression / 2.
            angle = math.asin(2.0 * q0 - 1.0) + 2.0 * math.pi / compression
            if angle >= math.pi / 2.0:
                return total  # reaches the end of the distribution
            q1 = (math.sin(angle) + 1.0) / 2.0
            return (q1 - q0) * total

        merged: List[_Centroid] = []
        weight_so_far = 0.0
        cur_mean, cur_weight = points[0]
        for mean, weight in points[1:]:
            projected = cur_weight + weight
            limit = max_cluster_weight(weight_so_far / total)
            if projected <= limit:
                cur_mean = (cur_mean * cur_weight + mean * weight) / projected
                cur_weight = projected
            else:
                merged.append(_Centroid(cur_mean, cur_weight))
                weight_so_far += cur_weight
                cur_mean, cur_weight = mean, weight
        merged.append(_Centroid(cur_mean, cur_weight))
        self._centroids = merged

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    @property
    def total_weight(self) -> float:
        return self._total_weight

    @property
    def centroid_count(self) -> int:
        """Number of retained centroids after flushing pending samples."""
        self._flush()
        return len(self._centroids)

    def quantile(self, q: float) -> Optional[Tuple[float, float]]:
        """Return (value, candidate_rank_error) for quantile ``q``.

        Returns None when the sketch is empty.  ``q`` must lie in [0, 1].
        """
        q = float(q)
        if math.isnan(q) or not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q!r}")
        self._flush()
        centroids = self._centroids
        if not centroids:
            return None
        total = self._total_weight
        if len(centroids) == 1:
            return centroids[0].mean, 0.0

        # Prefix sums of centroid weights; each centroid "owns" the rank
        # interval [prefix[i], prefix[i] + w_i).
        prefix = [0.0] * len(centroids)
        running = 0.0
        for i, centroid in enumerate(centroids):
            prefix[i] = running
            running += centroid.weight

        target = q * total
        first = centroids[0]
        last = centroids[-1]
        if target <= first.weight / 2.0:
            return first.mean, first.weight / 2.0
        if target >= total - last.weight / 2.0:
            return last.mean, last.weight / 2.0

        # Locate adjacent centroids whose centres bracket the target rank
        # and interpolate linearly between their means.
        center_prev = prefix[0] + first.weight / 2.0
        for i in range(1, len(centroids)):
            center = prefix[i] + centroids[i].weight / 2.0
            if target <= center:
                span = center - center_prev
                t = (target - center_prev) / span if span > 0.0 else 0.0
                value = centroids[i - 1].mean + t * (
                    centroids[i].mean - centroids[i - 1].mean
                )
                # Candidate rank error: the interpolated value is a mixture
                # of two clusters; any rank inside the union of their rank
                # intervals is plausible, so half that span bounds the error.
                low = prefix[i - 1]
                high = prefix[i] + centroids[i].weight
                return value, (high - low) / 2.0
            center_prev = center
        return last.mean, last.weight / 2.0


class QuantileEngine:
    """Partitioned, mergeable streaming quantile engine.

    Samples are tagged with a partition id; each partition keeps its own
    TDigest so shards can be built independently (e.g. in a batch/MapReduce
    stage) and merged later in a reduce stage.
    """

    def __init__(self, compression: float = 100.0) -> None:
        # Validate eagerly by constructing a probe digest.
        TDigest(compression)
        self.compression = float(compression)
        self._partitions: Dict[Hashable, TDigest] = {}

    def add(
        self,
        value: float,
        weight: float = 1.0,
        partition: Hashable = "default",
    ) -> None:
        """Add one weighted sample to a partition."""
        digest = self._partitions.get(partition)
        if digest is None:
            digest = TDigest(self.compression)
            self._partitions[partition] = digest
        digest.add(value, weight)

    def merge(self, other: "QuantileEngine") -> "QuantileEngine":
        """Merge all partitions of ``other`` into this engine (in place)."""
        if not isinstance(other, QuantileEngine):
            raise TypeError("can only merge another QuantileEngine")
        if other is self:
            raise ValueError("cannot merge an engine into itself")
        for partition, digest in other._partitions.items():
            target = self._partitions.get(partition)
            if target is None:
                self._partitions[partition] = digest.copy()
            else:
                target.merge(digest)
        return self

    def partitions(self) -> List[Hashable]:
        """Ids of all partitions that have received at least one sample."""
        return list(self._partitions.keys())

    def total_weight(self, partition: Optional[Hashable] = None) -> float:
        """Total ingested weight for one partition, or across all of them."""
        if partition is not None:
            digest = self._partitions.get(partition)
            return digest.total_weight if digest is not None else 0.0
        return sum(d.total_weight for d in self._partitions.values())

    def quantile(
        self, q: float, partition: Optional[Hashable] = None
    ) -> QuantileResult:
        """Query a quantile for one partition or for the merged whole.

        Unknown or empty partitions yield ``QuantileResult(value=None,
        rank_error=0.0, total_weight=0.0)`` rather than raising.
        """
        if partition is not None:
            digest = self._partitions.get(partition)
            if digest is None:
                return QuantileResult(None, 0.0, 0.0)
            return self._query(digest, q)
        merged = TDigest(self.compression)
        for digest in self._partitions.values():
            merged.merge(digest)
        return self._query(merged, q)

    @staticmethod
    def _query(digest: TDigest, q: float) -> QuantileResult:
        answer = digest.quantile(q)
        if answer is None:
            return QuantileResult(None, 0.0, 0.0)
        value, rank_error = answer
        return QuantileResult(value, rank_error, digest.total_weight)
