from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.monitor_lenet5_training import (
    _config_matches_resume_target,
    _config_value_matches,
    _json_response,
    _parameter_suffix,
    _resume_he_runtime_is_compatible,
    configured_training_hyperparams,
    configured_preset,
    html_page,
)


class MonitorResumeTests(unittest.TestCase):
    def test_status_json_replaces_non_finite_metrics_with_null(self) -> None:
        payload = json.loads(
            _json_response(
                {
                    "test_loss": float("nan"),
                    "nested": [float("inf"), 0.5],
                }
            ).decode("utf-8")
        )

        self.assertIsNone(payload["test_loss"])
        self.assertEqual(payload["nested"], [None, 0.5])

    def test_merge_suffix_distinguishes_seed_and_privacy_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "selection": {
                            "rounds": 200,
                            "num_clients": 100,
                            "num_edges": 10,
                            "seed": 42,
                            "dp_feature_epsilon_budget": 8.0,
                            "dp_update_epsilon_budget": 8.0,
                        },
                        "training": {
                            "dataset_name": "cifar10",
                            "model_name": "resnet18_pretrained",
                            "selection_period": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            suffix = _parameter_suffix({"run_dir": str(run_dir)})

        self.assertIn("100c_10e_200r_s42_sp1", suffix)
        self.assertTrue(suffix.endswith("eps8"))

    def test_pretrained_cifar_resnet18_defaults_match_paper(self) -> None:
        self.assertEqual(
            configured_training_hyperparams("cifar10", "resnet18_pretrained"),
            ("3", "0.01"),
        )

    def test_main_comparison_defaults_match_the_cifar_paper_run(self) -> None:
        page = html_page().decode("utf-8")

        self.assertIn(
            '<option value="cifar10" selected>CIFAR-10</option>', page
        )
        self.assertIn(
            '<option value="resnet18_pretrained" selected>', page
        )
        self.assertIn(
            'id="selectionPeriodInput" type="number" min="1" max="100" '
            'step="1" value="1"',
            page,
        )
        self.assertIn(
            '<option value="cifar_resnet" selected>', page
        )
        self.assertIn(
            '<input type="checkbox" value="fixed_dp">Fixed-DP', page
        )
        self.assertIn(
            '{policy: "ours_fixed_liieiiic", label: "Fixed Mode LIIEIIIC"}',
            page,
        )

    def test_configured_command_has_unique_options_and_total_rdp_targets(self) -> None:
        preset = configured_preset(
            mode="rounds",
            rounds=100,
            time_limit=123.0,
            train_limit=12000,
            test_limit=2000,
            policies=["ours"],
            figure_axis="round",
            partition_mode="extreme_edge_label_skew",
            clients=100,
            edges=10,
            seed=42,
            client_heterogeneity=2.0,
            edge_heterogeneity=1.5,
            selection_period=5,
            aggregation_fraction=1.0,
            pareto_archive_size=8,
            pareto_max_iters=10,
            pareto_neighbor_top_k=0,
            pareto_conflict_only=False,
            cloud_fusion_xi=0.2,
            cloud_fusion_eps=0.05,
            min_edge_cloud_fusion_ratio=0.5,
            resource_limit=1.35,
            risk_limit=0.5,
            executor="serial",
            executor_workers=None,
            local_epochs=1,
            learning_rate=0.15,
            initial_epsilon=8.0,
            dp_emb_epsilon=8.0,
            dp_upd_epsilon=8.0,
            he_backend="seal",
            he_aggregation_size=0,
            require_real_he=True,
            dp_profile="balanced",
            dataset="fmnist",
            model="lenet5",
            device="cuda",
        )
        args = preset["args"]
        options = [arg for arg in args if arg.startswith("--")]
        self.assertEqual(len(options), len(set(options)))

        def value(option: str) -> str:
            return args[args.index(option) + 1]

        self.assertEqual(value("--dp-feature-epsilon-budget"), "8")
        self.assertEqual(value("--dp-update-epsilon-budget"), "8")
        self.assertEqual(value("--time-limit"), "123")
        self.assertEqual(value("--min-edge-cloud-fusion-ratio"), "0.5")
        self.assertIn("--require-feasible", args)

    def test_old_execution_revision_cannot_resume(self) -> None:
        expected = {
            "selection": {"rounds": 100, "num_clients": 10},
            "training": {"execution_revision": "paper_flow_v13_pareto_search_budget"},
            "train_limit": 100,
            "test_limit": 20,
        }
        old_config = {
            "selection": {"rounds": 50, "num_clients": 10},
            "training": {"execution_revision": "paper_flow_v4"},
            "train_limit": 100,
            "test_limit": 20,
        }

        self.assertFalse(_config_matches_resume_target(old_config, expected))

    def test_monitor_polling_does_not_overlap_status_requests(self) -> None:
        page = html_page().decode("utf-8")

        self.assertIn("if (refreshInFlight) return refreshInFlight", page)
        self.assertIn("window.setTimeout(pollStatus", page)
        self.assertNotIn("setInterval(refresh, 1500)", page)

    def test_optional_worker_count_matches_none_only(self) -> None:
        self.assertTrue(_config_value_matches(None, None))
        self.assertTrue(_config_value_matches("", None))
        self.assertFalse(_config_value_matches(4, None))

    def test_seal_resume_requires_validated_runtime(self) -> None:
        expected = {"training": {"he_backend": "seal", "require_real_he": True}}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            policy_dir = run_dir / "individual_optimal"
            policy_dir.mkdir()

            (policy_dir / "summary.json").write_text(
                json.dumps({"he_status": "Python module 'seal' is importable."}),
                encoding="utf-8",
            )
            self.assertFalse(
                _resume_he_runtime_is_compatible(run_dir, ["individual_optimal"], expected)
            )

            (policy_dir / "summary.json").write_text(
                json.dumps(
                    {"he_status": "SEAL CKKS encrypted-vector validation passed (max error 2e-9)."}
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                _resume_he_runtime_is_compatible(run_dir, ["individual_optimal"], expected)
            )


if __name__ == "__main__":
    unittest.main()
