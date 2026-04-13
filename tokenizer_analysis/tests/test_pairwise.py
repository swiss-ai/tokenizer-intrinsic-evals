"""Tests for tokenizer_analysis.metrics.pairwise (VocabularyOverlapMetrics)."""

import pytest

from tokenizer_analysis.metrics.pairwise import VocabularyOverlapMetrics
from tokenizer_analysis.core.input_types import TokenizedData

from .conftest import SimpleProvider


def _make_td(tok, lang, tokens):
    return TokenizedData(tokenizer_name=tok, language=lang, tokens=tokens, text="x")


def _make_metrics(tok="tok", languages=None):
    return VocabularyOverlapMetrics(SimpleProvider(tok), languages=languages)


def _pairs(results, tok):
    return results["vocabulary_overlap"]["per_tokenizer"][tok]["language_pairs"]


class TestStructure:

    def test_top_level_key(self):
        tok = "tok"
        td = {tok: [_make_td(tok, "eng_Latn", [0, 1, 2]), _make_td(tok, "fra_Latn", [3, 4, 5])]}
        results = _make_metrics(tok).compute(td)
        print(results)
        assert "vocabulary_overlap" in results

    def test_expected_subkeys(self):
        tok = "tok"
        td = {tok: [_make_td(tok, "eng_Latn", [0, 1]), _make_td(tok, "fra_Latn", [2, 3])]}
        results = _make_metrics(tok).compute(td)
        vo = results["vocabulary_overlap"]
        assert "per_tokenizer" in vo
        assert "metadata" in vo
        assert tok in vo["per_tokenizer"]
        assert "language_pairs" in vo["per_tokenizer"][tok]
        assert "languages_used" in vo["per_tokenizer"][tok]

    def test_pair_entry_keys(self):
        tok = "tok"
        td = {tok: [_make_td(tok, "eng_Latn", [0, 1]), _make_td(tok, "fra_Latn", [2, 3])]}
        pairs = _pairs(_make_metrics(tok).compute(td), tok)
        pair = next(iter(pairs.values()))
        for key in ("language_1", "language_2", "jsd", "overlap", "tokens_l1", "tokens_l2", "shared_types"):
            assert key in pair

    def test_number_of_pairs(self):
        """n languages → n*(n-1)/2 pairs."""
        tok = "tok"
        langs = ["eng_Latn", "fra_Latn", "deu_Latn", "zho_Hans"]
        td = {tok: [_make_td(tok, l, [i, i + 1]) for i, l in enumerate(langs)]}
        pairs = _pairs(_make_metrics(tok).compute(td), tok)
        assert len(pairs) == 6  # 4*(4-1)/2

    def test_missing_tokenizer_skipped(self):
        tok = "tok"
        td = {}  # no data for this tokenizer
        results = _make_metrics(tok).compute(td)
        assert tok not in results["vocabulary_overlap"]["per_tokenizer"]


class TestJSDCorrectness:
    """Hand-verified JSD values."""

    def _jsd(self, tok, td, **kw):
        results = _make_metrics(tok, **kw).compute(td)
        pairs = _pairs(results, tok)
        return pairs

    def test_identical_distributions_jsd_zero(self):
        """Same token distribution in both languages → JSD = 0."""
        tok = "tok"
        # Both languages see token 0 twice and token 1 once → identical p
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 0, 1]),
            _make_td(tok, "fra_Latn", [0, 0, 1]),
        ]}
        pairs = self._jsd(tok, td)
        assert pairs["eng_Latn_vs_fra_Latn"]["jsd"] == pytest.approx(0.0, abs=1e-10)
        assert pairs["eng_Latn_vs_fra_Latn"]["overlap"] == pytest.approx(1.0, abs=1e-10)

    def test_disjoint_vocabularies_jsd_one(self):
        """Completely disjoint vocabularies → JSD = 1 (maximum divergence).

        p1 = {0: 1.0},  p2 = {1: 1.0}
        m  = {0: 0.5, 1: 0.5}
        JSD = 0.5*log2(1/0.5) + 0.5*log2(1/0.5) = 1.0
        """
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 0, 0]),
            _make_td(tok, "zho_Hans", [1, 1, 1]),
        ]}
        pairs = self._jsd(tok, td)
        assert pairs["eng_Latn_vs_zho_Hans"]["jsd"] == pytest.approx(1.0, abs=1e-10)
        assert pairs["eng_Latn_vs_zho_Hans"]["shared_types"] == 0

    def test_half_overlap_known_value(self):
        """Partially overlapping distributions with hand-computed JSD = 0.5.

        p1 = {0: 0.5, 1: 0.5, 2: 0.0}
        p2 = {0: 0.0, 1: 0.5, 2: 0.5}
        m  = {0: 0.25, 1: 0.5, 2: 0.25}
        KL(p1||m) = 0.5*log2(0.5/0.25) + 0.5*log2(0.5/0.5) = 0.5*1 + 0 = 0.5
        KL(p2||m) = 0.5*log2(0.5/0.5) + 0.5*log2(0.5/0.25) = 0 + 0.5*1 = 0.5
        JSD = 0.5*0.5 + 0.5*0.5 = 0.5
        """
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 0, 1, 1]),
            _make_td(tok, "fra_Latn", [1, 1, 2, 2]),
        ]}
        pairs = self._jsd(tok, td)
        assert pairs["eng_Latn_vs_fra_Latn"]["jsd"] == pytest.approx(0.5, abs=1e-10)

    def test_jsd_bounded_in_0_1(self):
        """JSD is always in [0, 1] regardless of token distribution."""
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1, 1, 2, 3]),
            _make_td(tok, "fra_Latn", [1, 2, 4, 4, 5]),
            _make_td(tok, "deu_Latn", [0, 0, 0, 6, 7]),
        ]}
        pairs = _make_metrics(tok).compute(td)["vocabulary_overlap"]["per_tokenizer"][tok]["language_pairs"]
        for pair in pairs.values():
            assert 0.0 <= pair["jsd"] <= 1.0
            assert 0.0 <= pair["overlap"] <= 1.0

    def test_overlap_is_one_minus_jsd(self):
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1, 2]),
            _make_td(tok, "fra_Latn", [1, 2, 3]),
        ]}
        pairs = _pairs(_make_metrics(tok).compute(td), tok)
        pair = next(iter(pairs.values()))
        assert pair["overlap"] == pytest.approx(1.0 - pair["jsd"])


class TestSymmetry:

    def test_jsd_is_symmetric(self):
        """JSD(l1, l2) should equal JSD(l2, l1)."""
        tok = "tok"
        td_ab = {tok: [
            _make_td(tok, "eng_Latn", [0, 0, 1, 2]),
            _make_td(tok, "fra_Latn", [1, 1, 3, 4]),
        ]}
        td_ba = {tok: [
            _make_td(tok, "fra_Latn", [1, 1, 3, 4]),
            _make_td(tok, "eng_Latn", [0, 0, 1, 2]),
        ]}
        pairs_ab = _pairs(_make_metrics(tok).compute(td_ab), tok)
        pairs_ba = _pairs(_make_metrics(tok).compute(td_ba), tok)

        jsd_ab = pairs_ab.get("eng_Latn_vs_fra_Latn", pairs_ab.get("fra_Latn_vs_eng_Latn"))["jsd"]
        jsd_ba = pairs_ba.get("eng_Latn_vs_fra_Latn", pairs_ba.get("fra_Latn_vs_eng_Latn"))["jsd"]
        assert jsd_ab == pytest.approx(jsd_ba, abs=1e-12)


class TestLanguagesConfig:

    def test_languages_filter_restricts_pairs(self):
        """Passing languages=['eng_Latn','fra_Latn'] excludes 'deu_Latn' from all pairs."""
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1]),
            _make_td(tok, "fra_Latn", [1, 2]),
            _make_td(tok, "deu_Latn", [3, 4]),
        ]}
        results = _make_metrics(tok, languages=["eng_Latn", "fra_Latn"]).compute(td)
        pairs = _pairs(results, tok)
        assert len(pairs) == 1
        assert "eng_Latn_vs_fra_Latn" in pairs
        langs_used = results["vocabulary_overlap"]["per_tokenizer"][tok]["languages_used"]
        assert "deu_Latn" not in langs_used

    def test_languages_none_uses_all(self):
        """When languages=None all languages in data are used."""
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1]),
            _make_td(tok, "fra_Latn", [1, 2]),
            _make_td(tok, "deu_Latn", [3, 4]),
        ]}
        pairs = _pairs(_make_metrics(tok, languages=None).compute(td), tok)
        assert len(pairs) == 3

    def test_missing_language_in_filter_is_skipped(self):
        """Languages not present in data are silently skipped."""
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1]),
            _make_td(tok, "fra_Latn", [2, 3]),
        ]}
        # "zho_Hans" is requested but absent — should not raise
        results = _make_metrics(tok, languages=["eng_Latn", "fra_Latn", "zho_Hans"]).compute(td)
        pairs = _pairs(results, tok)
        assert "eng_Latn_vs_fra_Latn" in pairs
        assert len(pairs) == 1

    def test_single_language_returns_empty_pairs(self):
        """Only one language present → no pairs, no crash."""
        tok = "tok"
        td = {tok: [_make_td(tok, "eng_Latn", [0, 1, 2])]}
        results = _make_metrics(tok).compute(td)
        pairs = _pairs(results, tok)
        assert pairs == {}

    def test_languages_order_does_not_affect_jsd(self):
        """Passing languages in different orders yields the same JSD."""
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1, 1]),
            _make_td(tok, "fra_Latn", [1, 2, 2]),
        ]}
        jsd_a = _pairs(_make_metrics(tok, languages=["eng_Latn", "fra_Latn"]).compute(td), tok)
        jsd_b = _pairs(_make_metrics(tok, languages=["fra_Latn", "eng_Latn"]).compute(td), tok)

        val_a = next(iter(jsd_a.values()))["jsd"]
        val_b = next(iter(jsd_b.values()))["jsd"]
        assert val_a == pytest.approx(val_b, abs=1e-12)


class TestExplicitPairs:

    def test_only_specified_pairs_computed(self):
        """Passing a list of tuples computes exactly those pairs."""
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1]),
            _make_td(tok, "fra_Latn", [1, 2]),
            _make_td(tok, "deu_Latn", [3, 4]),
        ]}
        metrics = _make_metrics(tok, languages=[("fra_Latn", "eng_Latn"), ("fra_Latn", "deu_Latn")])
        pairs = _pairs(metrics.compute(td), tok)
        assert set(pairs.keys()) == {"fra_Latn_vs_eng_Latn", "fra_Latn_vs_deu_Latn"}
        assert "eng_Latn_vs_deu_Latn" not in pairs

    def test_explicit_pairs_jsd_matches_flat_list(self):
        """JSD from explicit pair equals JSD from flat list for same pair."""
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 0, 1, 2]),
            _make_td(tok, "fra_Latn", [1, 1, 3, 4]),
        ]}
        jsd_flat = _pairs(_make_metrics(tok, languages=["eng_Latn", "fra_Latn"]).compute(td), tok)
        jsd_explicit = _pairs(_make_metrics(tok, languages=[("eng_Latn", "fra_Latn")]).compute(td), tok)

        flat_val = jsd_flat["eng_Latn_vs_fra_Latn"]["jsd"]
        explicit_val = jsd_explicit["eng_Latn_vs_fra_Latn"]["jsd"]
        assert flat_val == pytest.approx(explicit_val, abs=1e-12)

    def test_explicit_pairs_missing_language_skipped(self):
        """A pair where one language is absent is skipped without crashing."""
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1]),
            _make_td(tok, "fra_Latn", [2, 3]),
        ]}
        metrics = _make_metrics(tok, languages=[("eng_Latn", "fra_Latn"), ("eng_Latn", "zho_Hans")])
        pairs = _pairs(metrics.compute(td), tok)
        assert "eng_Latn_vs_fra_Latn" in pairs
        assert "eng_Latn_vs_zho_Hans" not in pairs

    def test_explicit_pairs_languages_used(self):
        """languages_used reflects only languages that appear in evaluated pairs."""
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1]),
            _make_td(tok, "fra_Latn", [1, 2]),
            _make_td(tok, "deu_Latn", [3, 4]),
        ]}
        results = _make_metrics(tok, languages=[("fra_Latn", "eng_Latn")]).compute(td)
        langs_used = results["vocabulary_overlap"]["per_tokenizer"][tok]["languages_used"]
        assert set(langs_used) == {"eng_Latn", "fra_Latn"}
        assert "deu_Latn" not in langs_used


class TestSharedTypes:

    def test_shared_types_count(self):
        """shared_types counts vocabulary types present in both languages."""
        tok = "tok"
        # eng_Latn uses {0,1,2}, fra_Latn uses {1,2,3} → shared = {1,2}
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1, 2]),
            _make_td(tok, "fra_Latn", [1, 2, 3]),
        ]}
        pairs = _pairs(_make_metrics(tok).compute(td), tok)
        assert pairs["eng_Latn_vs_fra_Latn"]["shared_types"] == 2

    def test_no_shared_types(self):
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1]),
            _make_td(tok, "zho_Hans", [2, 3]),
        ]}
        pairs = _pairs(_make_metrics(tok).compute(td), tok)
        assert pairs["eng_Latn_vs_zho_Hans"]["shared_types"] == 0


class TestMultipleTokenizers:

    def test_two_tokenizers_independent(self):
        """Results for two tokenizers are computed independently."""
        td = {
            "tok1": [_make_td("tok1", "eng_Latn", [0, 0, 1]), _make_td("tok1", "fra_Latn", [1, 1, 2])],
            "tok2": [_make_td("tok2", "eng_Latn", [10, 10]), _make_td("tok2", "fra_Latn", [20, 20])],
        }

        class TwoProvider(SimpleProvider):
            def get_tokenizer_names(self):
                return ["tok1", "tok2"]

        metrics = VocabularyOverlapMetrics(TwoProvider("tok1"))
        metrics.tokenizer_names = ["tok1", "tok2"]
        results = metrics.compute(td)

        per_tok = results["vocabulary_overlap"]["per_tokenizer"]
        assert "tok1" in per_tok
        assert "tok2" in per_tok
        # tok2 is fully disjoint → JSD = 1
        assert per_tok["tok2"]["language_pairs"]["eng_Latn_vs_fra_Latn"]["jsd"] == pytest.approx(1.0)


class TestPrintResults:

    def test_print_results_runs_without_error(self, capsys):
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 1, 1]),
            _make_td(tok, "fra_Latn", [1, 2, 2]),
            _make_td(tok, "deu_Latn", [0, 3, 3]),
        ]}
        metrics = _make_metrics(tok)
        results = metrics.compute(td)
        metrics.print_results(results)
        captured = capsys.readouterr().out
        assert "VOCABULARY OVERLAP" in captured
        assert "eng_Latn" in captured
        assert "fra_Latn" in captured

    def test_print_results_empty_does_not_crash(self, capsys):
        metrics = _make_metrics()
        metrics.print_results({})  # no vocabulary_overlap key

    def test_print_results_shows_jsd_values(self, capsys):
        tok = "tok"
        td = {tok: [
            _make_td(tok, "eng_Latn", [0, 0, 0]),
            _make_td(tok, "zho_Hans", [1, 1, 1]),
        ]}
        metrics = _make_metrics(tok)
        results = metrics.compute(td)
        metrics.print_results(results)
        captured = capsys.readouterr().out
        assert "1.0000" in captured  # disjoint → JSD = 1
