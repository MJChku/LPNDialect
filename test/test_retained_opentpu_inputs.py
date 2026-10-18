import json
from pathlib import Path
import tempfile
import unittest
from examples.opentpu.retained_inputs import verified_program


class RetainedProgramTest(unittest.TestCase):
    def test_omitted_zero_tail_still_preserves_full_instruction_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            words=[[2,1,0,0,0,0,0,0],[0]*8]
            (root/'program_0.json').write_text(json.dumps(words))
            (root/'prog_0.hex').write_text('00000002\n00000001\n')
            self.assertEqual(verified_program(root,0)[0],words)
            words[0][1]=3
            (root/'program_0.json').write_text(json.dumps(words))
            with self.assertRaisesRegex(ValueError,'differs from canonical'):verified_program(root,0)

    def test_incomplete_instruction_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'program_0.json').write_text('[[0]]')
            (root/'prog_0.hex').write_text('00000000\n')
            with self.assertRaisesRegex(ValueError,'eight-word'):verified_program(root,0)


if __name__=='__main__':unittest.main(verbosity=2)
