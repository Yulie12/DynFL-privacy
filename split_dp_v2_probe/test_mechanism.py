import math
import unittest

import torch
from mechanism import (OneReleaseCache, clip_rows, multiplier_for_zcdp_epsilon,
                       public_fixed_projection, zcdp_epsilon)


class MechanismTests(unittest.TestCase):
    def test_bound_calibration(self):
        for num in (1, 3, 10):
            m = multiplier_for_zcdp_epsilon(4.0, 1e-6, num)
            self.assertAlmostEqual(zcdp_epsilon(m, 1e-6, num), 4.0, places=10)
        self.assertEqual(zcdp_epsilon(1, 1e-5, 0), 0)

    def test_projection_and_clipping(self):
        q = public_fixed_projection(128, 64, seed=1)
        self.assertTrue(torch.allclose(q.T @ q, torch.eye(64), atol=1e-5))
        x = torch.randn((5, 128))
        y = clip_rows(x @ q, 0.25)
        self.assertTrue(bool((torch.linalg.vector_norm(y, dim=1) <= 0.250001).all()))

    def test_reuse_in_different_order(self):
        c = OneReleaseCache(0.25, 2.0, seed=4)
        x = torch.randn((3, 8))
        first = c.release('encoder-A/projection-8', ['a', 'b', 'c'], x)
        new_inputs = x.flip(0) + 9999.0  # cache must not recompute on reuse
        second = c.release('encoder-A/projection-8', ['c', 'b', 'a'], new_inputs)
        self.assertTrue(torch.equal(first.flip(0), second))
        self.assertEqual(c.stats.new_releases, 3)
        self.assertEqual(c.stats.cache_hits, 3)
        c.release('encoder-B/projection-8', ['a'], x[:1])
        self.assertEqual(c.stats.new_releases, 4)

    def test_fail_closed(self):
        c = OneReleaseCache(1.0, 2.0)
        with self.assertRaises(ValueError):
            c.release('a', ['x','x'], torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            c.release('a', ['x'], torch.tensor([[float('nan'),0,0]]))
        c.release('a', ['x'], torch.zeros(1, 3))
        with self.assertRaises(ValueError):
            c.release('a', ['x'], torch.zeros(1, 4))
        with self.assertRaises(ValueError):
            multiplier_for_zcdp_epsilon(0, 1e-5)


if __name__ == '__main__':
    unittest.main()
