"""The produced executable runs both Python-authored stages without Python."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


class FunctionalBridgeTest(unittest.TestCase):
  tool = None

  def test_native_linked_functional_and_timing(self):
    with tempfile.TemporaryDirectory() as d:
      subprocess.run([sys.executable,"-m","examples.native_functional.build",
          "--lpn-compiler",self.tool,"--output",d],check=True,capture_output=True,text=True)
      # A minimal environment with no Python path/interpreter in PATH.
      r = subprocess.run([str(Path(d)/"native-functional-timing"),"--stats"],
          env={"PATH":"/nonexistent"},check=True,capture_output=True,text=True)
      self.assertEqual(float(re.search(r"finished at t=([0-9.]+)",r.stdout)[1]),28)
      self.assertIn("place program_completed: 1 token(s)",r.stdout)
      self.assertIn("rollbacks: 0",r.stdout)


if __name__=="__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--tool",required=True)
  a = p.parse_args()
  FunctionalBridgeTest.tool = a.tool
  unittest.main(argv=[__file__],verbosity=2)
