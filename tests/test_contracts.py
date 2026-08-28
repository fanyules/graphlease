from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graphlease.contracts import compilation_config, load_config


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(REPOSITORY_ROOT / "configs" / "g0.json")

    def test_specialized_graph_counts_are_matched_and_union_is_exact(self):
        portfolios = self.config["portfolios"]
        small = portfolios["small_dense"]["capture_sizes"]
        large = portfolios["large_dense"]["capture_sizes"]
        union = portfolios["coverage_union"]["capture_sizes"]
        self.assertLessEqual(abs(len(small) - len(large)), 1)
        self.assertEqual(union, sorted(set(small) | set(large)))

    def test_eager_and_default_are_not_custom_portfolios(self):
        eager = compilation_config(self.config, "eager")
        self.assertEqual(eager["cudagraph_mode"], "NONE")
        self.assertIsNone(compilation_config(self.config, "default"))


if __name__ == "__main__":
    unittest.main()
