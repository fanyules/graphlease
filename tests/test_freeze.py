import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FreezeTests(unittest.TestCase):
    def test_freeze_blocks_controller_and_preserves_both_platforms(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "freeze.json"
            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "freeze_g0.py"),
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(value["a100_unblocked"])
            self.assertTrue(value["910b_unblocked"])
            self.assertFalse(value["controller_unblocked"])


if __name__ == "__main__":
    unittest.main()
