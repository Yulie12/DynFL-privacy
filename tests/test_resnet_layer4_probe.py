"""Offline CPU code tests; run: python -m unittest tests.test_resnet_layer4_probe"""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.models import resnet18

from dynfed.split_learning import build_full_model
from tools.run_resnet18_layer4_liic_probe import (
    MODEL_NAME, check_forward_equivalence, freeze_bn,
    trainable_tail, frozen_prefix, cache_prefix_features,
)


class ResNetProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with patch('dynfed.split_learning._make_torchvision_resnet',
                   side_effect=lambda *_args, **_kwargs: resnet18(weights=None)):
            cls.model = build_full_model(MODEL_NAME, torch.device('cpu'), 3, 32, 10)
        freeze_bn(cls.model)

    def test_frozen_prefix_outputs_identical_to_full_forward(self):
        data = np.random.default_rng(42).normal(size=(2, 3*32*32)).astype('float32')
        diff = check_forward_equivalence(self.model, data, torch.device('cpu'))
        self.assertLessEqual(diff, 1e-5)

    def test_only_layer4_and_classifier_trainable(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        self.assertEqual(sum(p.numel() for p in self.model.parameters() if p.requires_grad), 8_393_738)
        self.assertTrue(all(n.startswith(('layer4.', 'classifier.')) for n in names))

    def test_cached_features_allow_parameter_backward(self):
        # Inference-mode tensors cannot be saved for Conv2d weight backward.
        # Cached float32 tensors MUST be regular no-grad tensors instead.
        data = np.random.default_rng(7).normal(size=(2, 3 * 32 * 32)).astype('float32')
        features = cache_prefix_features(self.model, data, 2, torch.device('cpu'), 'unit')
        self.assertFalse(features.is_inference())
        self.model.train()
        freeze_bn(self.model)
        loss = trainable_tail(self.model, features).mean()
        loss.backward()
        self.assertIsNotNone(self.model.layer4[0].conv1.weight.grad)

    def test_bn_stays_frozen_when_training(self):
        before = [(id(m), m.running_mean.detach().clone())
                  for m in self.model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        self.model.train()
        freeze_bn(self.model)
        x = torch.randn(2, 3, 32, 32)
        out = trainable_tail(self.model, frozen_prefix(self.model, x))
        out.mean().backward()
        after = [(id(m), m.running_mean.detach().clone())
                 for m in self.model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        self.assertTrue(all(torch.equal(a[1], b[1]) for a,b in zip(before, after)))
        self.assertTrue(all(not m.training for m in self.model.modules()
                            if isinstance(m, torch.nn.BatchNorm2d)))


if __name__ == '__main__':
    unittest.main()
