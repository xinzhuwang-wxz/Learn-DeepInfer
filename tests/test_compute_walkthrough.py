"""Numerical contracts for chapter 08+; no upstream or device imports."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from s10_tensor_path import attention, paged_view, route, combine, kv_bytes


class ComputeWalkthrough(unittest.TestCase):
    def setUp(self):
        self.keys = [[1., 0.], [0., 1.], [1., 1.], [9., 9.]]
        self.values = [[1., 0.], [0., 1.], [2., 2.], [99., 99.]]
        self.stored = [self.values[2:], self.values[:2]]

    def test_exact_shared_walkthrough(self):
        restored = paged_view(self.stored, [1, 0], 4)
        self.assertEqual(restored, self.values)
        out = attention([1., 0.], self.keys, restored, [0, 1])
        expected_p = 1 / (1 + math.exp(-1 / math.sqrt(2)))
        self.assertAlmostEqual(out[0], expected_p)
        self.assertAlmostEqual(out[1], 1 - expected_p)
        routes = route([.6, .3, .1], [0., 0., .5], 2, {0: [3], 1: [1], 2: [0, 2]})
        self.assertEqual([(l, p) for l, p, _ in routes], [(0, 3), (2, 0)])
        self.assertAlmostEqual(routes[0][2], 6 / 7)
        self.assertAlmostEqual(routes[1][2], 1 / 7)
        result = combine(out, routes, {0: [2., 0.], 1: [0., 2.], 2: [1., 1.]})
        self.assertAlmostEqual(result[0], 2.526904406469514)
        self.assertAlmostEqual(result[1], .473095593530486)

    def test_wrong_table_preserves_shape_but_corrupts_identity(self):
        wrong = paged_view(self.stored, [0, 1], 4)
        self.assertEqual([len(row) for row in wrong], [2] * 4)
        result = attention([1., 0.], self.keys, wrong, [0, 1])
        p = 1 / (1 + math.exp(-1 / math.sqrt(2)))
        for value in result:
            self.assertAlmostEqual(value, p * 2 + (1 - p) * 99)
            self.assertEqual(round(value, 6), 34.03313)

    def test_invisible_finite_mutation_does_not_leak(self):
        expected = attention([1., 0.], self.keys, self.values, [0, 1])
        self.keys[3] = [1e100, 1e100]
        self.values[3] = [1e100, 1e100]
        self.assertEqual(attention([1., 0.], self.keys, self.values, [0, 1]), expected)

    def test_switch_second_expert_replica(self):
        a = route([.6, .3, .1], [0, 0, .5], 2, {0: [3], 1: [1], 2: [0, 2]})
        b = route([.6, .3, .1], [0, 0, .5], 2, {0: [3], 1: [1], 2: [2, 0]})
        self.assertNotEqual(a, b)
        outputs = {0: [2., 0.], 2: [1., 1.]}
        self.assertEqual(combine([.5, .5], a, outputs), combine([.5, .5], b, outputs))

    def test_missing_logical_mapping_is_not_silent(self):
        with self.assertRaises(KeyError) as caught:
            route([.6, .3, .1], [0, 0, .5], 2, {0: [3], 1: [1]})
        self.assertEqual(caught.exception.args, (2,))

    def test_length_and_format_contract(self):
        self.assertEqual(paged_view(self.stored, [1, 0], 3), self.values[:3])
        for length in (-1, 5):
            with self.assertRaises(ValueError):
                paged_view(self.stored, [1, 0], length)
        with self.assertRaises(ValueError):
            kv_bytes(3, 4, 32, 1)
        payload = 2 * kv_bytes(128, 4, 32, 1)
        self.assertEqual(payload, 136)
        self.assertEqual(math.ceil(payload / 32) * 32, 160)


if __name__ == '__main__':
    unittest.main()
