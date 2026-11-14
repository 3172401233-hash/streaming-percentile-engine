"""Tests for the deterministic streaming quantile engine.

Strategy: generate deterministic datasets (seeded RNG, fixed at import
time so every test run sees identical data), query the sketch, and check
the *rank* of the returned value against an exact weighted sort.  The
observed rank error must stay within the candidate rank error reported by
the engine itself.
"""

import bisect
import math
import random
import unittest

from streaming_percentile import QuantileEngine, QuantileResult, TDigest

QUANTILES = (0.5, 0.9, 0.99)


# --------------------------------------------------------------------- #
# deterministic datasets
# --------------------------------------------------------------------- #
def _make_datasets():
    rng = random.Random(20261114)
    n = 4000
    return {
        "uniform": [rng.uniform(-50.0, 50.0) for _ in range(n)],
        "long_tail": [rng.expovariate(1.0) for _ in range(n)],
        "skewed": [rng.lognormvariate(0.0, 1.5) for _ in range(n)],
        "constant": [7.25] * n,
        "negative": [-abs(rng.gauss(0.0, 10.0)) - 1.0 for _ in range(n)],
        "duplicates": [
            rng.choice([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 100.0])
            for _ in range(n)
        ],
    }


DATASETS = _make_datasets()


def exact_rank_interval(data, value):
    """[rank of first >= value, rank of last <= value] in unit weights."""
    ordered = sorted(data)
    lo = bisect.bisect_left(ordered, value)
    hi = bisect.bisect_right(ordered, value)
    return lo, hi


def rank_error_of(data, value, q):
    """Distance from the ideal rank q*n to the value's true rank interval."""
    lo, hi = exact_rank_interval(data, value)
    target = q * len(data)
    if hi < target:
        return target - hi
    if lo > target:
        return lo - target
    return 0.0


def check_within_declared_error(testcase, data, engine, slack=1.0):
    for q in QUANTILES:
        result = engine.quantile(q)
        testcase.assertIsNotNone(result.value)
        testcase.assertAlmostEqual(result.total_weight, float(len(data)))
        observed = rank_error_of(data, result.value, q)
        testcase.assertLessEqual(
            observed,
            result.rank_error * slack + 1e-9,
            msg=f"q={q}: observed rank error {observed} exceeds declared "
            f"{result.rank_error}",
        )


# --------------------------------------------------------------------- #
# accuracy across distributions
# --------------------------------------------------------------------- #
class TestAccuracy(unittest.TestCase):
    COMPRESSION = 100.0

    def test_distributions_within_declared_error(self):
        for name, data in DATASETS.items():
            with self.subTest(dataset=name):
                engine = QuantileEngine(compression=self.COMPRESSION)
                for value in data:
                    engine.add(value)
                check_within_declared_error(self, data, engine)

    def test_relative_accuracy_sanity(self):
        # Declared relative error should be well under 5% at compression 100.
        for name, data in DATASETS.items():
            with self.subTest(dataset=name):
                engine = QuantileEngine(compression=self.COMPRESSION)
                for value in data:
                    engine.add(value)
                for q in QUANTILES:
                    result = engine.quantile(q)
                    self.assertLessEqual(result.relative_rank_error, 0.05)

    def test_weighted_samples(self):
        # One heavy sample dominates the total weight; the sketch must
        # respect weights, and the returned value must honour the declared
        # rank error against the exactly-expanded dataset.
        engine = QuantileEngine(compression=50.0)
        engine.add(1.0, weight=1.0)
        engine.add(1000.0, weight=999.0)
        expanded = [1.0] + [1000.0] * 999
        check_within_declared_error(self, expanded, engine)
        result = engine.quantile(0.5)
        self.assertAlmostEqual(result.total_weight, 1000.0)
        self.assertGreater(result.value, 1.0)

    def test_higher_compression_is_not_worse(self):
        data = DATASETS["skewed"]
        errors = {}
        for compression in (50.0, 200.0):
            engine = QuantileEngine(compression=compression)
            for value in data:
                engine.add(value)
            errors[compression] = sum(
                rank_error_of(data, engine.quantile(q).value, q)
                for q in QUANTILES
            )
        self.assertLessEqual(errors[200.0], errors[50.0] + 1e-9)


# --------------------------------------------------------------------- #
# determinism, insertion order, merging
# --------------------------------------------------------------------- #
class TestDeterminismAndMerging(unittest.TestCase):
    def test_identical_runs_identical_results(self):
        data = DATASETS["uniform"]
        answers = []
        for _ in range(2):
            engine = QuantileEngine(compression=100.0)
            for value in data:
                engine.add(value)
            answers.append(tuple(engine.quantile(q) for q in QUANTILES))
        self.assertEqual(answers[0], answers[1])

    def test_insertion_order_stays_within_declared_error(self):
        data = DATASETS["long_tail"]
        shuffled = list(data)
        random.Random(7).shuffle(shuffled)
        reversed_data = list(reversed(data))
        for name, ordering in (
            ("original", data),
            ("shuffled", shuffled),
            ("reversed", reversed_data),
        ):
            with self.subTest(order=name):
                engine = QuantileEngine(compression=100.0)
                for value in ordering:
                    engine.add(value)
                check_within_declared_error(self, data, engine)

    def test_sharded_merge_within_declared_error(self):
        data = DATASETS["skewed"]
        shards = [QuantileEngine(compression=100.0) for _ in range(4)]
        for i, value in enumerate(data):
            shards[i % 4].add(value, partition=f"shard-{i % 4}")
        merged = QuantileEngine(compression=100.0)
        for shard in shards:
            merged.merge(shard)
        check_within_declared_error(self, data, merged)

    def test_partitioned_engine_aggregates(self):
        data = DATASETS["uniform"]
        engine = QuantileEngine(compression=100.0)
        for i, value in enumerate(data):
            engine.add(value, partition=f"p{i % 3}")
        self.assertEqual(sorted(engine.partitions()), ["p0", "p1", "p2"])
        self.assertAlmostEqual(engine.total_weight(), float(len(data)))
        check_within_declared_error(self, data, engine)
        # Per-partition queries only see their own shard.
        shard_data = data[0::3]
        result = engine.quantile(0.5, partition="p0")
        self.assertAlmostEqual(result.total_weight, float(len(shard_data)))

    def test_merge_is_order_dependent_but_bounded(self):
        # Merging shards in different orders may flip low-order bits, but
        # every order must honour the declared error bound.
        data = DATASETS["duplicates"]
        shards = []
        for shard_id in range(5):
            engine = QuantileEngine(compression=80.0)
            for i, value in enumerate(data):
                if i % 5 == shard_id:
                    engine.add(value)
            shards.append(engine)
        for order in (range(5), reversed(range(5))):
            merged = QuantileEngine(compression=80.0)
            for idx in order:
                merged.merge(shards[idx])
            check_within_declared_error(self, data, merged)


# --------------------------------------------------------------------- #
# memory bound
# --------------------------------------------------------------------- #
class TestMemoryBound(unittest.TestCase):
    def test_centroid_count_scales_with_compression_not_samples(self):
        data = DATASETS["uniform"]
        counts = {}
        for compression in (50.0, 100.0, 200.0):
            digest = TDigest(compression=compression)
            for value in data:
                digest.add(value)
            counts[compression] = digest.centroid_count
            # Retained state must stay proportional to compression and far
            # below the sample count.
            self.assertLessEqual(counts[compression], 3 * compression + 64)
            self.assertLess(counts[compression], len(data) // 4)
        self.assertLess(counts[50.0], counts[200.0])

    def test_many_samples_do_not_grow_state(self):
        digest = TDigest(compression=100.0)
        rng = random.Random(3)
        for _ in range(200_000):
            digest.add(rng.gauss(0.0, 1.0))
        self.assertLessEqual(digest.centroid_count, 3 * 100 + 64)


# --------------------------------------------------------------------- #
# edge cases and validation
# --------------------------------------------------------------------- #
class TestEdgeCases(unittest.TestCase):
    def test_nan_and_infinite_values_rejected(self):
        engine = QuantileEngine()
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    engine.add(bad)

    def test_invalid_weights_rejected(self):
        engine = QuantileEngine()
        for bad in (-1.0, -0.5, math.nan, math.inf, -math.inf):
            with self.subTest(weight=bad):
                with self.assertRaises(ValueError):
                    engine.add(1.0, weight=bad)

    def test_zero_weight_ignored(self):
        engine = QuantileEngine()
        engine.add(5.0, weight=0.0)
        result = engine.quantile(0.5)
        self.assertIsNone(result.value)
        self.assertEqual(result.total_weight, 0.0)

    def test_empty_partition_and_unknown_partition(self):
        engine = QuantileEngine()
        engine.add(1.0, partition="real")
        for partition in ("empty", "never-seen"):
            with self.subTest(partition=partition):
                result = engine.quantile(0.9, partition=partition)
                self.assertIsInstance(result, QuantileResult)
                self.assertIsNone(result.value)
                self.assertEqual(result.rank_error, 0.0)
                self.assertEqual(result.total_weight, 0.0)

    def test_constant_stream(self):
        engine = QuantileEngine()
        for _ in range(10_000):
            engine.add(3.5)
        for q in QUANTILES:
            result = engine.quantile(q)
            self.assertEqual(result.value, 3.5)

    def test_single_sample(self):
        engine = QuantileEngine()
        engine.add(-42.0, weight=2.5)
        result = engine.quantile(0.99)
        self.assertEqual(result.value, -42.0)
        self.assertAlmostEqual(result.total_weight, 2.5)

    def test_extreme_duplicate_mix(self):
        engine = QuantileEngine(compression=100.0)
        data = [0.0] * 5000 + [1e12] * 5 + [-1e12] * 5
        for value in data:
            engine.add(value)
        check_within_declared_error(self, data, engine)

    def test_invalid_quantile_rejected(self):
        engine = QuantileEngine()
        engine.add(1.0)
        for bad in (-0.1, 1.1, math.nan):
            with self.subTest(q=bad):
                with self.assertRaises(ValueError):
                    engine.quantile(bad)

    def test_invalid_compression_rejected(self):
        for bad in (0.0, -5.0, math.nan, math.inf):
            with self.subTest(compression=bad):
                with self.assertRaises(ValueError):
                    QuantileEngine(compression=bad)

    def test_merge_validation(self):
        engine = QuantileEngine()
        with self.assertRaises(TypeError):
            engine.merge("not-an-engine")
        with self.assertRaises(ValueError):
            engine.merge(engine)

    def test_merging_empty_is_noop(self):
        engine = QuantileEngine()
        engine.add(1.0)
        before = engine.quantile(0.5)
        engine.merge(QuantileEngine())
        self.assertEqual(engine.quantile(0.5), before)


if __name__ == "__main__":
    unittest.main()
