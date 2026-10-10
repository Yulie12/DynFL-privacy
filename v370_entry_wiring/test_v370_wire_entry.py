import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).with_name('v370_wire_entry.py')
spec = importlib.util.spec_from_file_location('v370_wire_entry', MODULE_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PAPER='''def build_command(config):
    system = config["system"]
    command = []
    fast_deadlines = system.get("fast_client_deadlines", {})
    if system.get("fl_first_split_on_demand", False):
        command.append("--fl-first-split-on-demand")
    return command
'''
RUNNER='''def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    selection = SelectionConfig(
        rounds=1,
    )
    return selection
'''
class TestWiring(unittest.TestCase):
    def test_paper_command_and_guards(self):
        patched, changed = m.patch_paper(PAPER)
        self.assertTrue(changed)
        self.assertFalse(m.patch_paper(patched)[1])
        ns = {}
        exec(patched, ns)
        command = ns['build_command']({'system': {'static_pair_admission_manifest':'configs/trust.json','fl_first_split_on_demand':True}})
        self.assertIn('--static-pair-admission-manifest', command)
        self.assertIn('configs/trust.json',command)
        with self.assertRaises(ValueError):
            ns['build_command']({'system': {'static_pair_admission_manifest':'configs/trust.json'}})
        with self.assertRaises(ValueError):
            ns['build_command']({'system': {'static_pair_admission_manifest':'configs/trust.json','fl_first_split_on_demand':True,'edge_only_requires_fast_deadline':True}})
        with self.assertRaises(ValueError):
            ns['build_command']({'system': {'static_pair_admission_manifest':'configs/trust.json','fl_first_split_on_demand':True,'fast_client_deadlines':{'1': 4}}})
    def test_train_injection(self):
        patched, changed = m.patch_train(RUNNER)
        self.assertTrue(changed)
        self.assertIn('selection = enable_static_pair_admission(selection, args.static_pair_admission_manifest)',patched)
        self.assertFalse(m.patch_train(patched)[1])
        compile(patched,'runner.py','exec')
    def test_unknown_layout_fails_closed(self):
        with self.assertRaises(RuntimeError):
            m.patch_paper('def foo(): pass')
        with self.assertRaises(RuntimeError):
            m.patch_train('def foo(): pass')

if __name__ == '__main__':
    unittest.main()
