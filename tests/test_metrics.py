from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from graphlease.metrics import output_digest, percentile, summarize_request_metrics


class MetricTests(unittest.TestCase):
    def test_percentile_interpolates_and_digest_is_order_sensitive(self):
        self.assertEqual(percentile([0.0, 10.0], 0.5), 5.0)
        self.assertNotEqual(
            output_digest([{"token_ids": [1]}, {"token_ids": [2]}]),
            output_digest([{"token_ids": [2]}, {"token_ids": [1]}]),
        )

    def test_request_summary_preserves_units(self):
        summary = summarize_request_metrics(
            [
                {"ttft_ms": 1.0, "tpot_ms": 2.0, "e2e_ms": 3.0},
                {"ttft_ms": 3.0, "tpot_ms": 4.0, "e2e_ms": 5.0},
            ]
        )
        self.assertEqual(summary["ttft_median_ms"], 2.0)
        self.assertGreater(summary["e2e_p99_ms"], 4.9)


if __name__ == "__main__":
    unittest.main()
