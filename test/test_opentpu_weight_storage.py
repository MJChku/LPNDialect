import unittest
from pathlib import Path
import tempfile
from examples.opentpu.weight_storage import immutable_blocks,storage_statistics,read_image


class WeightStorageTest(unittest.TestCase):
  def test_input_words_match_rtl_endianness_and_implicit_zero_tail(self):
    with tempfile.TemporaryDirectory() as tmp:
      path=Path(tmp)/'dram.bin';path.write_bytes(bytes([1,2,3,4]))
      self.assertEqual(read_image(path,8).tolist(),[4,3,2,1,0,0,0,0])
      path.write_bytes(bytes([1,2,3]))
      with self.assertRaises(ValueError):read_image(path,8)

  def test_partial_writes_remove_whole_affected_blocks(self):
    self.assertEqual(immutable_blocks([(0,128),(128,320)],[(80,84),(192,256)]),[(0,64),(128,192),(256,320)])
    self.assertEqual(immutable_blocks([(0,128)],[(0,256)]),[])
    self.assertEqual(immutable_blocks([(0,128),(256,320)],[(64,300)]),[(0,64)])

  def test_statistics_preserve_all_byte_values_and_escape_expensive_blocks(self):
    import numpy as np
    data=np.tile(np.arange(256,dtype=np.uint8),16)
    stats=storage_statistics(data)
    self.assertEqual(stats['raw_bytes'],4096)
    self.assertAlmostEqual(stats['byte_entropy_bits'],8)
    self.assertEqual(stats['global_huffman_payload_bytes'],4096)
    self.assertTrue(all(stats['zlib_lossless_round_trip_verified'].values()))
    zeros=storage_statistics(np.zeros(64,dtype=np.uint8))
    self.assertEqual(zeros['byte_entropy_bits'],0)
    self.assertEqual(zeros['rice64_payload_and_mode_bytes'],9)
    extremes=storage_statistics(np.full(64,128,dtype=np.uint8))
    self.assertEqual(extremes['rice64_payload_and_mode_bytes'],65)


if __name__=='__main__':unittest.main()
