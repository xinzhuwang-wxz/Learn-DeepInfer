import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from s01_request_loop import run
from s02_chunk_budget import Prompt,step
from s03_block_lifecycle import Blocks
from s04_parallel_votes import stage
from s05_verify_budget import choose
from s06_async_lifetime import Pipeline,borrowed_demo
from s07_bounded_replay import visible,compare
from s08_grammar_rollback import Matcher
from s09_metrics import decode_metrics

class Lessons(unittest.TestCase):
    def test_order(self):
        out,order=run([3,1,2]);self.assertEqual(order,[1,2,0]);self.assertEqual(list(map(len,out)),[3,1,2])
    def test_empty(self): self.assertEqual(run([]),([],[]))
    def test_length_validation(self):
        with self.assertRaises(ValueError):run([0])
    def test_chunk_budget(self):
        q=[Prompt('A',7),Prompt('B',3)];self.assertEqual(step(q,4),[('A',4)]);self.assertEqual(step(q,4),[('A',3),('B',1)]);self.assertEqual(step(q,4),[('B',2)])
    def test_chunk_no_duplicate(self):
        p=Prompt('A',2);step([p],4);self.assertEqual(step([p],4),[])
    def test_zero_budget(self):
        with self.assertRaises(ValueError):step([],0)
    def test_allocation_three_states(self):
        p=Blocks(4);self.assertEqual(p.status(4),'NEVER');p.allocate(2);self.assertEqual(p.status(2),'LATER');self.assertEqual(p.status(1),'OK')
    def test_pin_lifetime(self):
        p=Blocks(2);i=p.allocate(1)[0];p.pin(i);p.free(i);self.assertEqual(p.status(1),'LATER');p.free(i);self.assertEqual(p.status(1),'OK')
    def test_double_release(self):
        p=Blocks(2);i=p.allocate(1)[0];p.free(i)
        with self.assertRaises(ValueError):p.free(i)
    def test_zero_reserved(self):
        p=Blocks(4);self.assertEqual(p.allocate(3),[1,2,3])
    def test_all_vote(self): self.assertEqual(stage([True,False],[True,False],2),'decode')
    def test_any_vote(self): self.assertEqual(stage([True,False],[True,False],6),'prefill')
    def test_no_allocation_fallback(self): self.assertEqual(stage([True,True],[False,False],0),'decode')
    def test_any_over_all(self): self.assertEqual(stage([True,False],[True,False],6,2,6),'prefill')
    def test_budget(self):
        got=choose([[.9,.5],[.8,.1]],[(2,1.0),(4,1.3),(8,3.0)]);self.assertEqual(got['query_lengths'],[2,2]);self.assertEqual(got['total_q'],4)
    def test_confidence_bounds(self):
        with self.assertRaises(ValueError):choose([[1.1]],[(2,1)])
    def test_golden_only(self): self.assertEqual(choose([[]],[(1,1)])['query_lengths'],[1])
    def test_no_suitable_graph(self):
        with self.assertRaises(ValueError):choose([[],[]],[(1,1)])
    def test_tail_flush(self):
        p=Pipeline();self.assertEqual(p.step(10),[]);self.assertEqual(p.step(11),[10]);self.assertEqual(p.flush(),[11]);self.assertEqual(p.flush(),[])
    def test_borrowed_copy(self): self.assertEqual(borrowed_demo(),(11,10))
    def test_bounded_visibility(self):self.assertEqual(visible(11,12,4,6,2),(8,9,10,11));self.assertEqual(len(compare()),6)
    def test_valid_start(self):self.assertEqual(visible(11,12,4,6,2,10),(10,11))
    def test_invalid_keep(self):
        with self.assertRaises(ValueError):visible(11,12,4,5,2)
    def test_grammar(self):
        m=Matcher();self.assertEqual(m.allowed(),{'t','f'});self.assertFalse(m.accept('x'));self.assertEqual(m.prefix,'')
    def test_rollback(self):
        m=Matcher();[m.accept(c) for c in 'tru'];m.rollback(2);self.assertEqual(m.allowed(),{'r'});m.rollback(0);self.assertEqual(m.prefix,'t')
    def test_eos(self):
        m=Matcher();[m.accept(c) for c in 'false'];self.assertEqual(m.allowed(),{'<eos>'})
    def test_metric_units(self):self.assertEqual(decode_metrics(20,64,5,1,.75)['tokens_per_s_per_gpu'],6080)
    def test_varlen_refusal(self):
        with self.assertRaises(ValueError):decode_metrics(20,64,5,1,varlen=True)

if __name__=='__main__':unittest.main()
