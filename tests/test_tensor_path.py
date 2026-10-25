import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from s10_tensor_path import attention,paged_view,route,combine,kv_bytes
class TensorReference(unittest.TestCase):
    def test_masked_key_does_not_leak(self):
        a=attention([1,0],[[1,0],[999,999]],[[2,3],[999,999]],[0]);self.assertEqual(a,[2,3])
    def test_paged_reconstruction(self):
        logical=[[1,0],[0,1],[2,2],[3,3]];self.assertEqual(paged_view([logical[2:],logical[:2]],[1,0],4),logical)
    def test_attention_weights_normalized(self):self.assertEqual(attention([0,0],[[1,0],[0,1]],[[1,0],[0,1]],[0,1]),[.5,.5])
    def test_empty_visible_rejected(self):
        with self.assertRaises(ValueError):attention([1],[[1]],[[1]],[])
    def test_logical_physical_separate(self):
        r=route([.6,.3,.1],[0,0,.5],2,{0:[3],1:[1],2:[0,2]});self.assertEqual([(l,p) for l,p,_ in r],[(0,3),(2,0)]);self.assertAlmostEqual(sum(w for _,_,w in r),1)
    def test_replica_choice_does_not_change_logical_function(self):
        a=route([1,0],[0,0],1,{0:[3],1:[1]});b=route([1,0],[0,0],1,{0:[8],1:[1]});self.assertEqual(combine([1,1],a,{0:[2,3]}),combine([1,1],b,{0:[2,3]}))
    def test_scale_bytes_counted(self):
        self.assertEqual(kv_bytes(512,8,32,1),528);self.assertEqual(kv_bytes(512,4,16,1),288);self.assertEqual(kv_bytes(128,4,32,1),68)
if __name__=='__main__':unittest.main()
