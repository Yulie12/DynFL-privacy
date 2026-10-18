"""Deterministic regression tests for the transfer formula, not a DP proof."""
import math
import random
import unittest

from dynfl_transfer_j_reference import Client, Step, sgd_stage_coefficients, build_transfer_bound


def one_stage(theta0, steps, lr, mu, wd, inputs):
    m=0.
    theta=theta0
    for x in inputs:
        m=mu*m + x+wd*theta
        theta -= lr*m
    return theta


def run_hierarchy(theta0,clients,xinputs,cloud_step):
    cloud=[c for c in clients if c.mode in ('LIIC','LIIEIIIC')]
    total=sum(c.samples for c in cloud)
    if not cloud: return 0.
    numerator=0.
    for c in cloud:
        if c.mode=='LIIC':
            numerator += c.samples*one_stage(theta0,c.stages[0],c.lr,c.momentum,c.weight_decay,xinputs[c.client_id][0])
    edges={c.edge_id for c in cloud if c.mode=='LIIEIIIC'}
    for edge in edges:
        cohort=[c for c in cloud if c.mode=='LIIEIIIC' and c.edge_id==edge]
        mass=sum(c.samples for c in cohort)
        base=theta0
        for s in range(3):
            base=sum(c.samples*one_stage(base,c.stages[s],c.lr,c.momentum,c.weight_decay,xinputs[c.client_id][s]) for c in cohort)/mass
        numerator += mass*base
    return cloud_step*(numerator/total-theta0)


class TransferTests(unittest.TestCase):
    def make(self,cid,mode,edge,steps=(2,3,1),lr=.03,mu=.9,wd=.1,samples=10):
        stages=(tuple(tuple(Step(4,2.,1.1) for _ in range(K)) for K in steps[:3])
                if mode=='LIIEIIIC' else (tuple(Step(4,2.,1.1) for _ in range(steps[0])),))
        return Client(cid,mode,samples,edge,stages,lr,mu,wd)

    def test_stage_exact_against_optimizer_recurrence(self):
        rng=random.Random(17)
        for k in (1,2,3,9,25):
            for mu in (0.,.5,.9):
                for wd in (0.,.0001,.3):
                    f,beta=sgd_stage_coefficients(steps=k,lr=.04,momentum=mu,weight_decay=wd)
                    inputs=[rng.uniform(-2.,2.) for _ in range(k)]
                    theta0=rng.uniform(-3.,3.)
                    explicit=one_stage(theta0,range(k),.04,mu,wd,inputs)
                    expansion=f*theta0+sum(b*x for b,x in zip(beta,inputs))
                    self.assertAlmostEqual(explicit,expansion,places=11)

    def test_real_torch_sgd_exact(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch is unavailable')
        torch.set_num_threads(1)
        rng=random.Random(9)
        for K,mu,wd in ((1,0.9,0.0001),(3,0.9,0.0001),
                        (9,0.9,0.2),(5,0.,0.),(7,.5,.1)):
            eta=.03
            initial=rng.uniform(-2.,2.)
            inputs=[rng.uniform(-3.,3.) for _ in range(K)]
            parameter=torch.nn.Parameter(torch.tensor(initial,dtype=torch.float64))
            optimizer=torch.optim.SGD([parameter],lr=eta,momentum=mu,weight_decay=wd)
            for x in inputs:
                parameter.grad=torch.tensor(x,dtype=torch.float64)
                optimizer.step()
            f,beta=sgd_stage_coefficients(steps=K,lr=eta,momentum=mu,weight_decay=wd)
            exact=f*initial+sum(a*x for a,x in zip(beta,inputs))
            self.assertAlmostEqual(parameter.item(),exact,places=11)

    def test_mixed_hierarchy_exact_linear_decomposition(self):
        rng=random.Random(91)
        clients=[self.make(0,'LIIC',0), self.make(1,'LIIEIIIC',0,samples=13),
                 self.make(2,'LIIEIIIC',0,steps=(3,1,2)),
                 self.make(3,'LIIEIIIC',1,steps=(1,3,2),samples=7),
                 self.make(4,'LIIE',1),self.make(5,'SKIP',1)]
        gamma=.85
        b=build_transfer_bound(clients=clients,dimension=1,theta_norm=1.,ideal_delta_norm=.1,cloud_step=gamma)
        for _ in range(80):
            theta0=rng.uniform(-3.,3.)
            inputs={c.client_id:[[rng.uniform(-1.,1.) for step in stage] for stage in c.stages] for c in clients}
            actual=run_hierarchy(theta0,clients,inputs,gamma)
            linear=b.displacement_scale*theta0 + sum(ev.coefficient*inputs[ev.client_id][ev.stage][ev.step] for ev in b.events)
            self.assertAlmostEqual(actual,linear,places=10)

    def test_special_cases_and_fail_closed(self):
        c=self.make(0,'LIIC',0)
        b=build_transfer_bound(clients=[c],dimension=5,theta_norm=0.,ideal_delta_norm=0.)
        self.assertGreater(b.noise_rms_radius,0.)
        b0=build_transfer_bound(clients=[self.make(0,'SKIP',0)],dimension=5,theta_norm=1.,ideal_delta_norm=2.)
        self.assertEqual(b0.upper_j,4.)
        with self.assertRaises(ValueError):
            Client(0,'LIIEIIIC',10,0,(c.stages[0],),.01,.9,.001)
        with self.assertRaises(ValueError):
            build_transfer_bound(clients=[c,c],dimension=3,theta_norm=1.,ideal_delta_norm=1.)

    def test_monte_carlo_independent_gaussian(self):
        # This validates the numeric calculation in a simplified Gaussian-only
        # experiment; it DOES NOT replace the analytic Minkowski proof.
        rng=random.Random(342)
        clients=[self.make(0,'LIIC',0,wd=.01),self.make(1,'LIIEIIIC',0,wd=.01)]
        b=build_transfer_bound(clients=clients,dimension=1,theta_norm=1.7,ideal_delta_norm=.04,
                                theta_dot_ideal_delta=-.068)
        accum=0.
        for _ in range(6000):
            z={c.client_id:[[rng.gauss(0.,step.noise_std) for step in stage] for stage in c.stages] for c in clients}
            delta=run_hierarchy(1.7,clients,z,1.)
            accum+=(delta+.04)**2  # ideal delta=-.04
        self.assertLess(accum/6000,b.upper_j)
        theoretical_noise_var=sum((ev.coefficient*ev.noise_std)**2 for ev in b.events)
        self.assertGreater(theoretical_noise_var,0.)
        # Monte-Carlo validates exact scalar linear variance component.
        accum=0.
        for _ in range(6000):
            linear_noise=sum(ev.coefficient*rng.gauss(0.,ev.noise_std) for ev in b.events)
            accum+=linear_noise**2
        self.assertLess(abs(accum/6000-theoretical_noise_var),.12*theoretical_noise_var)


if __name__ == '__main__':
    unittest.main()
