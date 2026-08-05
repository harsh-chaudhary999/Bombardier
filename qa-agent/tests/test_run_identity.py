import unittest
from datetime import datetime, timezone

from observability.run_identity import deterministic_analysis_run_id


class TestRunIdentity(unittest.TestCase):
    def test_deterministic_same_minute(self):
        t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        a = deterministic_analysis_run_id("confluence:1", ["B", "A"], minute_bucket=t)
        b = deterministic_analysis_run_id("confluence:1", ["A", "B"], minute_bucket=t)
        self.assertEqual(a, b)
        c = deterministic_analysis_run_id("confluence:2", ["A", "B"], minute_bucket=t)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
