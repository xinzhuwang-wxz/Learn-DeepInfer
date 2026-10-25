import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]

def load(path):
    name='lesson_'+path.parent.name
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

class Progressive(unittest.TestCase):
    def test_every_stage_has_same_ordered_answer(self):
        for path in sorted((ROOT/'lessons').glob('*/code.py')):
            with self.subTest(stage=path.parent.name):
                m=load(path); e=m.Engine([m.Request('A',7,'true'),m.Request('B',3,'false'),m.Request('C',5,'true')])
                self.assertEqual(e.run(),['true','false','true'])
                self.assertEqual(e.report()['committed_characters'],13)
                if hasattr(e,'pool'):self.assertEqual(sum(e.pool.refs),0)
                if hasattr(e,'pending'):self.assertEqual(e.pending,[])
    def test_empty_at_every_stage(self):
        for path in sorted((ROOT/'lessons').glob('*/code.py')):
            m=load(path);self.assertEqual(m.Engine([]).run(),[])
    def test_duplicate_ids_rejected(self):
        m=load(ROOT/'lessons/09_metrics/code.py')
        with self.assertRaises(ValueError):m.Engine([m.Request('A',2,'true'),m.Request('A',3,'false')])
    def test_memory_pressure_recovers(self):
        m=load(ROOT/'lessons/09_metrics/code.py');e=m.Engine([m.Request(str(i),7,'false') for i in range(12)])
        self.assertEqual(e.run(),['false']*12);self.assertEqual(sum(e.pool.refs),0)
    def test_impossible_request_fails(self):
        m=load(ROOT/'lessons/09_metrics/code.py')
        with self.assertRaises(ValueError):m.Engine([m.Request('A',1000,'true')]).run()
    def test_invalid_grammar_fails(self):
        m=load(ROOT/'lessons/09_metrics/code.py')
        e=m.Engine([m.Request('A',1,'wrong')])
        with self.assertRaises(ValueError):e.run()
        self.assertEqual(sum(e.pool.refs),0);self.assertEqual(e.pending,[])
    def test_mixed_lengths_stay_live(self):
        m=load(ROOT/'lessons/09_metrics/code.py')
        for count in [1,2,5,12]:
            e=m.Engine([m.Request(str(i),1+(i*7)%23,'false' if i%2 else 'true') for i in range(count)])
            self.assertEqual(e.run(),['false' if i%2 else 'true' for i in range(count)])
            self.assertEqual(sum(e.pool.refs),0)
    def test_deterministic_trace(self):
        m=load(ROOT/'lessons/09_metrics/code.py');a=m.Engine([m.Request('A',7,'true')]);b=m.Engine([m.Request('A',7,'true')]);a.run();b.run();self.assertEqual(a.trace,b.trace)
    def test_final_flush_exactly_once(self):
        m=load(ROOT/'lessons/09_metrics/code.py');e=m.Engine([m.Request('A',1,'true')]);e.run();self.assertEqual([x for x in e.trace if x[0]=='flush'],[('flush',1)])

if __name__=='__main__':unittest.main()
