"""Do not trust a modified table merely because its manifest was rehashed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from examples.opentpu.rtl_evolution.rom_inputs import reciprocal_inputs


class ReciprocalInputTest(unittest.TestCase):
    def test_self_consistent_unverified_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('coeff.hex','correction.hex'):(root/name).write_text('0\n')
            report=dict(status='complete',correction_bits=2,inputs=1<<23,coarse_segments=4096,
                        normalized_sequence_reconstructed_all_inputs=True,
                        files={n:dict(sha256=hashlib.sha256((root/n).read_bytes()).hexdigest())
                               for n in ('coeff.hex','correction.hex')})
            (root/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'differs from its calibration'):reciprocal_inputs(root)


if __name__=='__main__':unittest.main(verbosity=2)
