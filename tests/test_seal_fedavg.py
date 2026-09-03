from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dynfed.fmnist_lenet5_dynamic import (
    _measure_he_wall_time,
    _seal_process_aggregate_chunk_batch,
    fedavg_split_seal,
)
from dynfed.he_backend import HEOperationMetrics, check_he_backend, decode_seal_vector
from dynfed.split_learning import fedavg_split


class HEOperationMetricsTests(unittest.TestCase):
    def test_wall_time_and_worker_time_have_distinct_output_fields(self) -> None:
        metrics = HEOperationMetrics(
            backend="seal",
            encryption_time_sec=1.5,
            addition_time_sec=0.25,
            decryption_time_sec=0.5,
        )

        @_measure_he_wall_time
        def measured_operation(*, he_metrics: HEOperationMetrics) -> str:
            return "ok"

        self.assertEqual(measured_operation(he_metrics=metrics), "ok")
        output = metrics.as_dict()
        self.assertGreater(output["he_wall_time_sec"], 0.0)
        self.assertEqual(output["he_worker_cpu_time_sec"], 2.25)
        self.assertEqual(
            output["he_ciphertext_bytes_semantics"],
            "seal_save_size_upper_bound",
        )

    def test_process_task_batch_preserves_chunk_order(self) -> None:
        bounds = ((0, 4096), (4096, 8192), (8192, 9000))
        with patch(
            "dynfed.fmnist_lenet5_dynamic._seal_process_aggregate_chunk",
            side_effect=lambda item: item,
        ):
            results = _seal_process_aggregate_chunk_batch(bounds)

        self.assertEqual(results, list(bounds))


class SealFedAvgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.availability = check_he_backend("seal")
        if not cls.availability.available:
            raise unittest.SkipTest(cls.availability.detail)

    def test_vector_round_trip_preserves_slots(self) -> None:
        import seal

        parms = seal.EncryptionParameters(seal.scheme_type.ckks)
        parms.set_poly_modulus_degree(8192)
        parms.set_coeff_modulus(seal.CoeffModulus.Create(8192, [40, 40, 40, 40]))
        encoder = seal.CKKSEncoder(seal.SEALContext(parms))
        expected = np.array([0.1, 0.2, -0.3, 0.4], dtype=np.float64)

        actual = decode_seal_vector(encoder, encoder.encode(expected, 2 ** 40))[: expected.size]

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)

    def test_encrypted_fedavg_matches_plaintext_fedavg(self) -> None:
        torch.manual_seed(7)
        device = torch.device("cpu")
        base_end = torch.nn.Linear(7, 3, bias=True)
        base_edge = torch.nn.Linear(3, 2, bias=True)
        plain_end, plain_edge = copy.deepcopy(base_end), copy.deepcopy(base_edge)
        seal_end, seal_edge = copy.deepcopy(base_end), copy.deepcopy(base_edge)

        state_diffs = []
        for scale in (0.01, -0.025, 0.04):
            state_diffs.append(
                {
                    "end": {
                        name: torch.randn_like(param) * scale
                        for name, param in base_end.named_parameters()
                    },
                    "edge": {
                        name: torch.randn_like(param) * scale
                        for name, param in base_edge.named_parameters()
                    },
                }
            )
        sample_counts = [3, 5, 11]

        fedavg_split(state_diffs, sample_counts, plain_end, plain_edge, device)
        fedavg_split_seal(state_diffs, sample_counts, seal_end, seal_edge, device)

        for plain_param, seal_param in zip(plain_end.parameters(), seal_end.parameters()):
            torch.testing.assert_close(seal_param, plain_param, rtol=0.0, atol=2e-6)
        for plain_param, seal_param in zip(plain_edge.parameters(), seal_edge.parameters()):
            torch.testing.assert_close(seal_param, plain_param, rtol=0.0, atol=2e-6)

    def test_mixed_encrypted_and_plaintext_fedavg_matches_plaintext(self) -> None:
        torch.manual_seed(19)
        device = torch.device("cpu")
        base_end = torch.nn.Linear(5, 3, bias=True)
        base_edge = torch.nn.Linear(3, 2, bias=True)
        plain_end, plain_edge = copy.deepcopy(base_end), copy.deepcopy(base_edge)
        seal_end, seal_edge = copy.deepcopy(base_end), copy.deepcopy(base_edge)
        state_diffs = [
            {
                "end": {
                    name: torch.randn_like(param) * scale
                    for name, param in base_end.named_parameters()
                },
                "edge": {
                    name: torch.randn_like(param) * scale
                    for name, param in base_edge.named_parameters()
                },
            }
            for scale in (0.015, -0.02, 0.035)
        ]
        sample_counts = [2, 7, 13]
        metrics = HEOperationMetrics(backend="seal")

        fedavg_split(state_diffs, sample_counts, plain_end, plain_edge, device)
        fedavg_split_seal(
            state_diffs,
            sample_counts,
            seal_end,
            seal_edge,
            device,
            encrypted_mask=[True, False, True],
            he_metrics=metrics,
        )

        for plain_param, seal_param in zip(plain_end.parameters(), seal_end.parameters()):
            torch.testing.assert_close(seal_param, plain_param, rtol=0.0, atol=2e-6)
        for plain_param, seal_param in zip(plain_edge.parameters(), seal_edge.parameters()):
            torch.testing.assert_close(seal_param, plain_param, rtol=0.0, atol=2e-6)
        self.assertEqual(metrics.encrypted_updates, 2)
        self.assertEqual(
            metrics.encrypted_parameter_values,
            2 * sum(param.numel() for model in (seal_end, seal_edge) for param in model.parameters()),
        )
        self.assertGreaterEqual(metrics.ciphertext_count, 2)
        self.assertGreater(metrics.ciphertext_bytes, 0)
        self.assertLess(metrics.max_abs_error, 2e-6)

    def test_partial_update_encryption_is_rejected(self) -> None:
        device = torch.device("cpu")
        end = torch.nn.Linear(4, 2)
        edge = torch.nn.Linear(2, 2)
        state_diff = {
            "end": {name: torch.zeros_like(param) for name, param in end.named_parameters()},
            "edge": {name: torch.zeros_like(param) for name, param in edge.named_parameters()},
        }

        with self.assertRaisesRegex(ValueError, "Partial CKKS aggregation is disabled"):
            fedavg_split_seal(
                [state_diff],
                [1],
                end,
                edge,
                device,
                he_aggregation_size=1,
            )

    def test_process_parallel_encrypted_fedavg_matches_plaintext(self) -> None:
        torch.manual_seed(31)
        device = torch.device("cpu")
        base_end = torch.nn.Linear(9, 4, bias=True)
        base_edge = torch.nn.Linear(4, 3, bias=True)
        plain_end, plain_edge = copy.deepcopy(base_end), copy.deepcopy(base_edge)
        seal_end, seal_edge = copy.deepcopy(base_end), copy.deepcopy(base_edge)
        state_diffs = [
            {
                "end": {
                    name: torch.randn_like(param) * scale
                    for name, param in base_end.named_parameters()
                },
                "edge": {
                    name: torch.randn_like(param) * scale
                    for name, param in base_edge.named_parameters()
                },
            }
            for scale in (0.01, -0.03, 0.025)
        ]
        sample_counts = [3, 7, 12]
        metrics = HEOperationMetrics(backend="seal")

        fedavg_split(state_diffs, sample_counts, plain_end, plain_edge, device)
        fedavg_split_seal(
            state_diffs,
            sample_counts,
            seal_end,
            seal_edge,
            device,
            encrypted_mask=[True, False, True],
            he_workers=2,
            he_metrics=metrics,
        )

        for plain_param, seal_param in zip(plain_end.parameters(), seal_end.parameters()):
            torch.testing.assert_close(seal_param, plain_param, rtol=0.0, atol=2e-6)
        for plain_param, seal_param in zip(plain_edge.parameters(), seal_edge.parameters()):
            torch.testing.assert_close(seal_param, plain_param, rtol=0.0, atol=2e-6)
        self.assertEqual(metrics.encrypted_updates, 2)
        self.assertGreater(metrics.ciphertext_bytes, 0)
        self.assertGreater(metrics.mapped_update_bytes, 0)
        self.assertGreater(metrics.process_tasks, 0)
        self.assertLess(metrics.max_abs_error, 2e-6)


if __name__ == "__main__":
    unittest.main()
