import unittest

from observability.canonical_json import dumps_canonical, fingerprint_sha256, normalize_json_obj


class TestCanonicalJson(unittest.TestCase):
    def test_dumps_canonical_sorts_nested_keys(self):
        a = {"z": 1, "a": {"m": 2, "b": 3}}
        b = {"a": {"b": 3, "m": 2}, "z": 1}
        self.assertEqual(dumps_canonical(a), dumps_canonical(b))

    def test_fingerprint_stable(self):
        self.assertEqual(
            fingerprint_sha256({"b": 2, "a": 1}),
            fingerprint_sha256({"a": 1, "b": 2}),
        )

    def test_normalize_roundtrip(self):
        x = {"outer": {"z": True, "y": [3, 1, 2]}, "list": [{"c": 1, "a": 2}]}
        y = normalize_json_obj(x)
        self.assertEqual(dumps_canonical(y), dumps_canonical(x))


if __name__ == "__main__":
    unittest.main()
