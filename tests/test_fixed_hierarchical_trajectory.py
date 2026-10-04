"""Arithmetic checks for the fixed, in-process edge/cloud parity diagnostic."""
import unittest
import torch

from dynfed.privacy import PrivacyAccountant, calibrate_gaussian_noise
from experiments.validate_trusted_aggregate_dp import release_scales
from experiments.validate_trusted_aggregate_trajectory import (
    hierarchical_mean_from_edge_sums, trajectory_noise_seed, apply_vector_update,
)


class FixedHierarchicalTests(unittest.TestCase):
    def test_40_clients_4_edges_parity_before_and_after_same_dp_noise(self):
        generator = torch.Generator().manual_seed(11)
        updates = [torch.randn(5130, generator=generator) for _ in range(40)]
        # Clip individual clients before either aggregation path.
        clipped = [u * min(1., 0.1 / max(float(u.norm()), 1e-12)) for u in updates]
        direct = torch.stack(clipped).mean(dim=0)
        edge_sums = [torch.zeros_like(direct) for _ in range(4)]
        counts = [0] * 4
        for i, u in enumerate(clipped):
            edge_sums[i % 4].add_(u)
            counts[i % 4] += 1
        cloud = hierarchical_mean_from_edge_sums(edge_sums, counts)
        self.assertEqual(counts, [10, 10, 10, 10])
        self.assertTrue(torch.allclose(cloud, direct, rtol=2e-5, atol=2e-7))
        std = 0.01
        gen = torch.Generator().manual_seed(trajectory_noise_seed(42, 0))
        noise = torch.randn(cloud.shape, generator=gen) * std
        self.assertTrue(torch.allclose(cloud + noise, direct + noise, atol=2e-7))

    def test_unequal_edge_sizes_are_weighted_by_client_count(self):
        # Unweighted edge means produce [2., 3.]; correct client-weighted mean is [2.5, 3.5].
        sums = [torch.tensor([1., 2.]), torch.tensor([9., 12.])]
        counts = [1, 3]
        result = hierarchical_mean_from_edge_sums(sums, counts)
        self.assertTrue(torch.allclose(result, torch.tensor([2.5, 3.5])))

    def test_single_cloud_release_dp_noise_calibration(self):
        sigma = calibrate_gaussian_noise(8., 1e-5, 20)
        std = release_scales(40, 0.1, sigma)["aggregate_noise_std"]
        self.assertAlmostEqual(std, 2 * 0.1 * sigma / 40)
        ledger = PrivacyAccountant(8., 1e-5)
        for _ in range(20):
            self.assertTrue(ledger.can_add_event(sigma))
            ledger.add_event(sigma)  # One final cloud release, never one release/edge.
        self.assertLessEqual(ledger.current_epsilon(), 8. + 1e-9)

    def test_edge_validation(self):
        with self.assertRaises(ValueError):
            hierarchical_mean_from_edge_sums([torch.ones(2), torch.ones(2)], [1, 0])
        with self.assertRaises(ValueError):
            hierarchical_mean_from_edge_sums([torch.ones(2), torch.ones(3)], [1, 1])

    def test_server_step_applied_once_after_cloud_release(self):
        base = {"end": {"weight": torch.zeros(3)}, "edge": {}}
        released = torch.tensor([1., -2., 3.])
        updated = apply_vector_update(base, [("end", "weight")], released, 0.5)
        self.assertTrue(torch.equal(updated["end"]["weight"], released * 0.5))
        self.assertTrue(torch.equal(base["end"]["weight"], torch.zeros(3)))


if __name__ == "__main__":
    unittest.main()
