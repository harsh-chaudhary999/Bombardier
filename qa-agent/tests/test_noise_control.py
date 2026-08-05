"""
Guards the two noise-control mechanisms, using measurements from the real corpus.

Context. A whole-space Confluence ingest produced 3,400 documents of which
only ~148 are PRDs. Retrieval for "what is provisional status for accounts"
returned ranks 1..15 spanning **5.5%** — the definitive answer at 0.8365 and an unrelated
UI release note at 0.8030. Two consequences:

  * Every absolute score threshold in this codebase is inert against that distribution.
  * Structural filtering (document type) removes far more noise than any score cut.

So: doc_classify narrows the corpus, and rank_filter reports whether score is usable at
all before trimming on it. Both are pinned here against the observed numbers.
"""
import unittest

from embeddings.rank_filter import relative_cut, separation
from ingestion.doc_classify import (
    IMPLEMENTATION_PLAN,
    MEETING_NOTE,
    OTHER,
    PRD,
    RELEASE_NOTE,
    TECH_DOC,
    TEST_PLAN,
    classify,
    title_filter,
)

# Title shapes taken from a real corpus, with product specifics replaced.
REAL_TITLES = [
    ("PRD: Account Re-Verification System", PRD),
    ("PRD- Nudging Users to Verify Work Emails", PRD),
    ("Product Requirement – Recommended Items_v6", PRD),
    ("Large and Branded Landing Pages- PRD", PRD),
    ("[PRD] [Overhaul] Email Strategy for End Users", PRD),
    ("Email Fill Rate PRD (Draft)", PRD),
    ("Implementation Plan for Adding 'PROVISIONAL' Status in Account Verification System",
     IMPLEMENTATION_PLAN),
    ("Implementation Plan — Report-anchored verification for trial accounts (PROJ-1234)",
     IMPLEMENTATION_PLAN),
    ("Tech Doc - New Pilot (Provisional Onboarding & Instant Publish)", TECH_DOC),
    ("TECHDOC - AI-Assisted Development", TECH_DOC),
    ("Imported Lead (lead_source_id = 13) — Activation-Triggered Creation | Tech Design", TECH_DOC),
    ("Test Plan: Experiment 1 — Work-Email Instant Publish (Backend)", TEST_PLAN),
    ("v1.6.2 : Added Analytics SDK", RELEASE_NOTE),
    ("v1.6.10 : Account Validations", RELEASE_NOTE),
    ("Sprint Grooming", MEETING_NOTE),
    ("Roadmap Planning:-June-End/July-21", MEETING_NOTE),
    ("Events Sheet - End User", OTHER),
    ("Account Profile", OTHER),
    ("Report-anchored verification for trial accounts", OTHER),
]


class ClassifyTests(unittest.TestCase):
    def test_real_corpus_titles(self):
        for title, expected in REAL_TITLES:
            with self.subTest(title=title[:50]):
                self.assertEqual(classify(title), expected)

    def test_compound_title_prefers_the_specific_type(self):
        """An implementation plan referencing a PRD is an implementation plan."""
        self.assertEqual(
            classify("Implementation Plan for the Account PRD"), IMPLEMENTATION_PLAN
        )

    def test_prd_marker_is_case_and_punctuation_insensitive(self):
        for t in ("prd - x", "PRD: x", "x - PRD", "[prd] x", "PRDs"):
            with self.subTest(t=t):
                self.assertEqual(classify(t), PRD)

    def test_unknown_and_empty(self):
        self.assertEqual(classify(""), OTHER)
        self.assertEqual(classify(None), OTHER)
        self.assertEqual(classify("Onboarding Funnal visibility phase 2"), OTHER)


class TitleFilterTests(unittest.TestCase):
    """
    The ES clauses are declared separately from the regexes. Deriving them once produced
    the phrase "prds" from `\\bprds?\\b`, which matches no document — every "PRD:" page
    would have been silently excluded from an include=[prd] filter.
    """

    def _phrases(self, flt, bucket):
        out = []
        for c in (flt or {}).get("bool", {}).get(bucket, []):
            if "match_phrase" in c:
                out.append(c["match_phrase"]["doc_title"])
        return out

    def test_prd_filter_contains_the_bare_token(self):
        phrases = self._phrases(title_filter(include=[PRD]), "should")
        self.assertIn("prd", phrases, "must match 'PRD:' titles, not only 'prds'")
        self.assertIn("product requirement", phrases)

    def test_include_uses_minimum_should_match(self):
        f = title_filter(include=[PRD])
        self.assertEqual(f["bool"]["minimum_should_match"], 1)

    def test_exclude_uses_must_not(self):
        f = title_filter(exclude=[TEST_PLAN])
        self.assertIn("test plan", self._phrases(f, "must_not"))
        self.assertNotIn("should", f["bool"])

    def test_release_note_uses_regexp_for_version_titles(self):
        """'v1.6.2 : Added Analytics SDK' cannot be matched as a phrase."""
        clauses = (title_filter(include=[RELEASE_NOTE]) or {})["bool"]["should"]
        self.assertTrue(any("regexp" in c for c in clauses))

    def test_other_is_expressed_as_matching_no_named_type(self):
        f = title_filter(include=[OTHER])
        self.assertTrue(any("must_not" in c.get("bool", {})
                            for c in f["bool"]["should"] if isinstance(c, dict)))

    def test_no_constraint_returns_none(self):
        self.assertIsNone(title_filter())
        self.assertIsNone(title_filter([], []))

    def test_unknown_type_names_are_ignored(self):
        self.assertIsNone(title_filter(include=["not_a_type"]))


class SeparationTests(unittest.TestCase):
    # The exact scores measured for "provisional status for accounts".
    REAL = [0.8365, 0.8107, 0.8065, 0.8043, 0.8030, 0.8008, 0.8008, 0.7952,
            0.7939, 0.7932, 0.7924, 0.7924, 0.7919, 0.7905, 0.7902]

    def _res(self, scores):
        return [{"score": s} for s in scores]

    def test_real_distribution_is_compressed(self):
        d = separation(self._res(self.REAL))
        self.assertLess(d["spread_pct"], 15.0,
                        "the whole point: absolute thresholds cannot work here")
        self.assertAlmostEqual(d["spread_pct"], 5.53, places=1)

    def test_real_distribution_still_has_a_statistically_distinct_leader(self):
        """5.5% spread looks hopeless; 6 sigma is a strong signal. That is the insight."""
        d = separation(self._res(self.REAL))
        self.assertGreater(d["top_z"], 5.0)
        self.assertTrue(d["usable"])
        self.assertIn("above tail", d["reason"])

    def test_flat_distribution_is_reported_unusable(self):
        d = separation(self._res([0.8] * 8))
        self.assertFalse(d["usable"])
        self.assertIn("cannot separate", d["reason"])

    def test_wide_distribution_is_usable_on_spread_alone(self):
        d = separation(self._res([0.91, 0.62, 0.58, 0.55]))
        self.assertGreater(d["spread_pct"], 15.0)
        self.assertTrue(d["usable"])
        self.assertEqual(d["reason"], "wide spread")

    def test_empty_and_single(self):
        self.assertFalse(separation([])["usable"])
        self.assertTrue(separation(self._res([0.5]))["usable"])


class RelativeCutTests(unittest.TestCase):
    REAL = SeparationTests.REAL

    def _res(self, scores):
        return [{"score": s, "i": i} for i, s in enumerate(scores)]

    def test_drops_the_noise_on_the_real_distribution(self):
        """Rank 5 was a UI release note; it must not survive."""
        kept, diag = relative_cut(self._res(self.REAL), min_keep=3)
        self.assertEqual(len(kept), 3)
        self.assertTrue(diag["cut_applied"])
        self.assertEqual(diag["dropped"], 12)
        self.assertNotIn(4, [r["i"] for r in kept])

    def test_min_keep_prevents_cutting_to_a_single_result(self):
        """Knee detection fires after rank 1 here, but rank 2 was genuinely relevant."""
        kept, _ = relative_cut(self._res(self.REAL), min_keep=1)
        self.assertEqual(len(kept), 1)
        kept3, _ = relative_cut(self._res(self.REAL), min_keep=3)
        self.assertEqual(len(kept3), 3)

    def test_smoothly_graded_list_is_not_trimmed(self):
        """No defensible cut point means no cut."""
        kept, diag = relative_cut(self._res([0.9, 0.85, 0.80, 0.75, 0.70]), min_keep=1)
        self.assertEqual(len(kept), 5)
        self.assertFalse(diag["cut_applied"])

    def test_results_are_reordered_by_score(self):
        kept, _ = relative_cut(self._res([0.5, 0.9, 0.7]), min_keep=3)
        self.assertEqual([r["score"] for r in kept], [0.9, 0.7, 0.5])

    def test_max_keep_caps_the_result(self):
        kept, _ = relative_cut(self._res([0.9, 0.88, 0.86, 0.84]), min_keep=1, max_keep=2)
        self.assertEqual(len(kept), 2)

    def test_rerank_score_key(self):
        res = [{"score": 0.5, "rerank_score": 9.0}, {"score": 0.9, "rerank_score": -2.0}]
        kept, _ = relative_cut(res, key="rerank_score", min_keep=1)
        self.assertEqual(kept[0]["rerank_score"], 9.0)

    def test_empty_input(self):
        kept, diag = relative_cut([], min_keep=3)
        self.assertEqual(kept, [])
        self.assertFalse(diag["usable"])

    def test_diagnostics_always_present(self):
        _, diag = relative_cut(self._res([0.9, 0.8]), min_keep=1)
        for k in ("spread_pct", "top_z", "usable", "reason", "kept", "dropped", "cut_applied"):
            self.assertIn(k, diag)


if __name__ == "__main__":
    unittest.main()
