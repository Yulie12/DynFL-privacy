"""Real LeNet5 model gradient and checkpoint regression, no private data."""
import math,unittest
import torch
import torch.nn.functional as F
from dynfed.split_learning import build_split_pair,build_full_model
from dynfed.online_j_liic_envelope import (
    lenet5_layer_norm_upper_from_states,
    fmnist_lenet5_gradient_norm_cap_by_layers,
    fmnist_lenet5_gradient_norm_cap,
)
from examples.check_lenet5_checkpoint_gradient_cap import checkpoint_gradient_caps

class LayerwiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(18)
        cls.end,cls.edge=build_split_pair('lenet5',torch.device('cpu'))

    def norms(self):
        return lenet5_layer_norm_upper_from_states(self.end.state_dict(),self.edge.state_dict())

    def test_strictly_tighter_actual_initial_model(self):
        norms=self.norms()
        R=math.sqrt(sum(v*v for e in norms.values() for v in e.values()))
        old=fmnist_lenet5_gradient_norm_cap(model_parameter_norm_upper=R)
        new=fmnist_lenet5_gradient_norm_cap_by_layers(layer_norm_uppers=norms)
        self.assertGreater(old/new,3)

    def test_autograd_gradient_never_exceeds_analytic_bound(self):
        net=build_full_model('lenet5',torch.device('cpu'))
        net.load_state_dict({**self.end.state_dict(),**self.edge.state_dict()})
        cap=fmnist_lenet5_gradient_norm_cap_by_layers(layer_norm_uppers=self.norms())
        for y in (0,5,9):
            net.zero_grad(set_to_none=True)
            x=4.5*torch.rand(1,1,28,28)-2.25
            F.cross_entropy(net(x),torch.tensor([y])).backward()
            got=math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in net.parameters()))
            self.assertLessEqual(got,cap)

    def test_missing_invalid_and_nonfinite_weights_fail_closed(self):
        norms=self.norms()
        norms['fc3']['weight_fro']=float('nan')
        with self.assertRaises(ValueError):
            fmnist_lenet5_gradient_norm_cap_by_layers(layer_norm_uppers=norms)
        wrong=dict(self.end.state_dict())
        wrong['conv1.weight']=torch.zeros(1)
        with self.assertRaisesRegex(ValueError,'shape'):
            lenet5_layer_norm_upper_from_states(wrong,self.edge.state_dict())
        wrong=dict(self.end.state_dict())
        wrong['conv1.bias']=torch.full((6,),float('nan'))
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            lenet5_layer_norm_upper_from_states(wrong,self.edge.state_dict())

    def test_checkpoint_does_not_backfill_earlier_rounds(self):
        got=checkpoint_gradient_caps({'next_round':5,'global_end_state':self.end.state_dict(),
                                      'global_edge_state':self.edge.state_dict()})
        self.assertEqual(got['usable_from_predecision_round'],5)
        self.assertGreater(got['improvement_factor'],3)
        self.assertFalse(got['J_certified'])
        with self.assertRaises(ValueError):
            checkpoint_gradient_caps({'next_round':False})

if __name__=='__main__':unittest.main()
