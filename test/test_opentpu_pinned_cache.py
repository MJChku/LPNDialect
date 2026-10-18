from types import SimpleNamespace
import unittest
from examples.opentpu.pinned_cache_study import study_kernel


class PinnedCacheTest(unittest.TestCase):
  def test_cold_warm_host_change_and_partial_write_invalidation(self):
    import numpy as np
    I=SimpleNamespace(MM=1,W8=0,WF_SHIFT=0);W=SimpleNamespace(SP_DRAM=0)
    cfg=SimpleNamespace(D=64)
    command=SimpleNamespace(instruction=SimpleNamespace(op=1,flags=0,w=[0,0,0,65537,64,0,0]),
                            footprint=SimpleNamespace(wr=[]))
    image=np.arange(128,dtype=np.uint8);states={128:np.zeros(2,dtype=bool)}
    cold,prior=study_kernel(cfg,[command,command],image,None,states,0,I,W)
    self.assertEqual(cold['128']['cache_hit_bytes'],0)
    self.assertEqual(cold['128']['hbm_weight_bytes'],128)
    warm,prior=study_kernel(cfg,[command,command],image,prior,states,0,I,W)
    self.assertEqual(warm['128']['cache_hit_bytes'],128)
    changed=image.copy();changed[3]^=1
    invalid,prior=study_kernel(cfg,[command],changed,prior,states,0,I,W)
    self.assertEqual(invalid['128']['invalidated_valid_lines'],1)
    self.assertEqual(invalid['128']['cache_hit_bytes'],0)
    writer=SimpleNamespace(instruction=SimpleNamespace(op=2),footprint=SimpleNamespace(wr=[(0,32,36)]))
    written,prior=study_kernel(cfg,[command,writer],changed,prior,states,0,I,W)
    self.assertEqual(written['128']['cache_hit_bytes'],0)
    self.assertFalse(states[128][0])
    self.assertEqual(states[128].sum(),0)


if __name__=='__main__':unittest.main()
