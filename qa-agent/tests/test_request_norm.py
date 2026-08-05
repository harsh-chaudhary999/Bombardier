import unittest

from observability.request_norm import normalize_module_list


class TestRequestNorm(unittest.TestCase):
    def test_module_preserves_acronym_casing(self):
        self.assertEqual(
            normalize_module_list(["API", "api"]),
            ["API"],
        )
        self.assertEqual(
            normalize_module_list(["iOS", "ios"]),
            ["iOS"],
        )

    def test_module_sorted_unique(self):
        a = normalize_module_list(["B", "A"])
        b = normalize_module_list(["A", "B"])
        self.assertEqual(a, b)
        self.assertEqual(a, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
