"""
Pairwise tokenizer metrics: vocabulary overlap between language pairs.

Implements the Jensen-Shannon Divergence (JSD) based vocabulary overlap from:
    Limisiewicz et al. (2023). "Tokenization Impacts Multilingual Language Modeling:
    Assessing Vocabulary Allocation and Overlap Across Languages."
    ACL Findings 2023. https://aclanthology.org/2023.findings-acl.350
"""

from typing import Dict, List, Any, Optional, Sequence, Tuple, Union
from collections import Counter
import math
import logging

from .base import BaseMetrics, TokenizedDataProcessor
from ..core.input_types import TokenizedData
from ..core.input_providers import InputProvider

logger = logging.getLogger(__name__)


def _adjacency_to_pairs(graph: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Convert an adjacency dict to a deduplicated list of (l1, l2) pairs.

    ``{"A": ["B", "C"], "B": ["C"]}`` → ``[("A","B"), ("A","C"), ("B","C")]``

    Pairs where both orderings appear (A→B and B→A) are kept as the first
    occurrence only.
    """
    seen: set = set()
    pairs: List[Tuple[str, str]] = []
    for src, neighbours in graph.items():
        for dst in neighbours:
            key = (min(src, dst), max(src, dst))
            if key not in seen:
                seen.add(key)
                pairs.append((src, dst))
    return pairs


class VocabularyOverlapMetrics(BaseMetrics):
    """Vocabulary overlap between language pairs measured via Jensen-Shannon Divergence.

    For each tokenizer and each pair of languages (l1, l2):
    1. Build the empirical token frequency distribution d_{l,V} for each language
       by counting token occurrences and normalizing (Eq. 3 in the paper).
    2. Compute JSD(d_{l1} || d_{l2}) as:
           JSD = 0.5 * KL(d_l1 || m) + 0.5 * KL(d_l2 || m)
       where m = 0.5 * (d_l1 + d_l2)  (Eqs. 6–7).

    JSD is bounded in [0, 1] (using log base 2). Lower values indicate more
    vocabulary overlap; JSD = 0 means identical distributions.
    """

    def __init__(
        self,
        input_provider: InputProvider,
        languages: Optional[Union[Sequence[str], Sequence[Tuple[str, str]], Dict[str, List[str]]]] = None,
    ):
        """
        Initialize vocabulary overlap metrics.

        Args:
            input_provider: InputProvider instance.
            languages: Controls which language pairs are evaluated. Four forms:

                * ``None`` — all pairs from all languages in the data.
                * ``["eng_Latn", "fra_Latn", "deu_Latn"]`` — all pairs among
                  the listed languages (n*(n-1)/2 pairs).
                * ``[("fra_Latn", "eng_Latn"), ("fra_Latn", "deu_Latn")]`` —
                  only the explicitly listed pairs.
                * ``{"eng_Latn": ["fra_Latn", "deu_Latn"], "jpn_Jpan": ["kor_Hang"]}``
                  — adjacency dict (graph); each key is connected to its listed
                  neighbours. Duplicate edges (A→B and B→A) are deduplicated.
                  This format can be loaded directly from a JSON file.
        """
        super().__init__(input_provider)
        if languages is None:
            self._explicit_pairs: Optional[List[Tuple[str, str]]] = None
            self.languages: Optional[List[str]] = None
        elif isinstance(languages, dict):
            self._explicit_pairs = _adjacency_to_pairs(languages)
            self.languages = None
        elif languages and isinstance(languages[0], tuple):
            self._explicit_pairs = [tuple(p) for p in languages]  # type: ignore[misc]
            self.languages = None
        else:
            self._explicit_pairs = None
            self.languages = list(languages)  # type: ignore[arg-type]

    def compute(
        self,
        tokenized_data: Optional[Dict[str, List[TokenizedData]]] = None,
    ) -> Dict[str, Any]:
        """Compute pairwise vocabulary overlap for all tokenizers.

        Args:
            tokenized_data: Optional pre-tokenized data dict. If None the
                input_provider is queried.

        Returns:
            ``{"vocabulary_overlap": {...}}`` dict structured as::

                {
                  "vocabulary_overlap": {
                    "per_tokenizer": {
                      "<tok_name>": {
                        "language_pairs": {
                          "<l1>_vs_<l2>": {
                            "language_1": str,
                            "language_2": str,
                            "jsd": float,          # Jensen-Shannon divergence [0,1]
                            "overlap": float,      # 1 - jsd (convenience)
                            "tokens_l1": int,
                            "tokens_l2": int,
                            "shared_types": int,   # vocab types appearing in both langs
                          },
                          ...
                        },
                        "languages_used": [str, ...],
                      }
                    },
                    "metadata": {...}
                  }
                }
        """
        if tokenized_data is None:
            tokenized_data = self.get_tokenized_data()

        results: Dict[str, Any] = {
            "vocabulary_overlap": {
                "per_tokenizer": {},
                "metadata": {
                    "description": (
                        "Pairwise vocabulary overlap between languages measured via "
                        "Jensen-Shannon Divergence (JSD). JSD ∈ [0,1]; lower = more overlap."
                    ),
                    "metric_range": "[0.0, 1.0]",
                    "reference": "Limisiewicz et al., ACL Findings 2023",
                },
            }
        }

        for tok_name in self.tokenizer_names:
            if tok_name not in tokenized_data:
                continue

            tok_data = tokenized_data[tok_name]
            tok_result = self._compute_for_tokenizer(tok_data)
            results["vocabulary_overlap"]["per_tokenizer"][tok_name] = tok_result

        return results


    def _compute_for_tokenizer(
        self, tok_data: List[TokenizedData]
    ) -> Dict[str, Any]:
        """Compute pairwise JSD for one tokenizer."""
        lang_groups = TokenizedDataProcessor.group_by_language(tok_data)

        if self._explicit_pairs is not None:
            return self._compute_explicit_pairs(lang_groups)

        # Determine which languages to include
        if self.languages is not None:
            languages_to_use = [l for l in self.languages if l in lang_groups]
            missing = [l for l in self.languages if l not in lang_groups]
            if missing:
                logger.warning(
                    "VocabularyOverlapMetrics: requested languages not found in data: %s",
                    missing,
                )
        else:
            languages_to_use = sorted(lang_groups.keys())

        if len(languages_to_use) < 2:
            logger.warning(
                "VocabularyOverlapMetrics: need at least 2 languages, got %d (%s). "
                "Returning empty result.",
                len(languages_to_use),
                languages_to_use,
            )
            return {"language_pairs": {}, "languages_used": languages_to_use}

        # Build per-language token frequency Counter (token_id → count)
        lang_counters: Dict[str, Counter] = {}
        for lang in languages_to_use:
            all_tokens = TokenizedDataProcessor.flatten_all_tokens(lang_groups[lang])
            lang_counters[lang] = Counter(all_tokens)

        # Compute pairwise JSD
        language_pairs: Dict[str, Any] = {}
        for i, l1 in enumerate(languages_to_use):
            for l2 in languages_to_use[i + 1 :]:
                pair_key = f"{l1}_vs_{l2}"
                language_pairs[pair_key] = self._compute_pair(
                    l1, l2, lang_counters[l1], lang_counters[l2]
                )

        return {
            "language_pairs": language_pairs,
            "languages_used": languages_to_use,
        }

    def _compute_explicit_pairs(
        self, lang_groups: Dict[str, List[TokenizedData]]
    ) -> Dict[str, Any]:
        """Compute JSD only for the caller-specified (l1, l2) tuples."""
        assert self._explicit_pairs is not None

        needed = {lang for pair in self._explicit_pairs for lang in pair}
        missing = [l for l in needed if l not in lang_groups]
        if missing:
            logger.warning(
                "VocabularyOverlapMetrics: requested languages not found in data: %s",
                missing,
            )

        lang_counters: Dict[str, Counter] = {
            lang: Counter(TokenizedDataProcessor.flatten_all_tokens(lang_groups[lang]))
            for lang in needed
            if lang in lang_groups
        }

        language_pairs: Dict[str, Any] = {}
        languages_seen: set = set()
        for l1, l2 in self._explicit_pairs:
            if l1 not in lang_counters or l2 not in lang_counters:
                logger.warning(
                    "VocabularyOverlapMetrics: skipping pair (%s, %s) — one or both languages missing.",
                    l1, l2,
                )
                continue
            pair_key = f"{l1}_vs_{l2}"
            language_pairs[pair_key] = self._compute_pair(
                l1, l2, lang_counters[l1], lang_counters[l2]
            )
            languages_seen.update([l1, l2])

        return {
            "language_pairs": language_pairs,
            "languages_used": sorted(languages_seen),
        }

    @staticmethod
    def _compute_pair(
        l1: str,
        l2: str,
        counts1: Counter,
        counts2: Counter,
    ) -> Dict[str, Any]:
        """Compute JSD between two language token distributions.

        Implements Eqs. 3, 6, 7 from Limisiewicz et al. (2023).
        Uses log base 2 so JSD ∈ [0, 1].
        """
        total1 = sum(counts1.values())
        total2 = sum(counts2.values())

        if total1 == 0 or total2 == 0:
            # Degenerate: one language has no tokens
            return {
                "language_1": l1,
                "language_2": l2,
                "jsd": 1.0,
                "overlap": 0.0,
                "tokens_l1": total1,
                "tokens_l2": total2,
                "shared_types": 0,
            }

        # Union of vocabulary types present in either language
        all_types = set(counts1.keys()) | set(counts2.keys())
        shared_types = len(set(counts1.keys()) & set(counts2.keys()))

        # Empirical probability distributions (Eq. 3)
        p1 = {v: counts1.get(v, 0) / total1 for v in all_types}
        p2 = {v: counts2.get(v, 0) / total2 for v in all_types}

        # Mixture distribution m = 0.5 * (p1 + p2)  (Eq. 7)
        # JSD = 0.5 * KL(p1 || m) + 0.5 * KL(p2 || m)  (Eq. 6)
        jsd = 0.0
        for v in all_types:
            q1 = p1[v]
            q2 = p2[v]
            m = 0.5 * (q1 + q2)
            if m == 0.0:
                continue
            if q1 > 0.0:
                jsd += 0.5 * q1 * math.log2(q1 / m)
            if q2 > 0.0:
                jsd += 0.5 * q2 * math.log2(q2 / m)

        # Numerical safety: clamp to [0, 1]
        jsd = max(0.0, min(1.0, jsd))

        return {
            "language_1": l1,
            "language_2": l2,
            "jsd": jsd,
            "overlap": 1.0 - jsd,
            "tokens_l1": total1,
            "tokens_l2": total2,
            "shared_types": shared_types,
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print vocabulary overlap results as a JSD matrix per tokenizer.

        For each tokenizer, renders a symmetric language-pair matrix showing
        the Jensen-Shannon Divergence (lower = more overlap), followed by a
        sorted ranking of the most/least overlapping pairs.
        """
        data = results.get("vocabulary_overlap")
        if not data:
            return

        per_tokenizer = data.get("per_tokenizer", {})
        if not per_tokenizer:
            print("\nVOCABULARY OVERLAP: no results available.")
            return

        print("\n" + "=" * 60)
        print("VOCABULARY OVERLAP (Jensen-Shannon Divergence)")
        print("Lower JSD = more overlap  |  JSD in [0, 1]")
        print("=" * 60)

        for tok_name in self.tokenizer_names:
            if tok_name not in per_tokenizer:
                continue

            tok_data = per_tokenizer[tok_name]
            languages = tok_data.get("languages_used", [])
            pairs = tok_data.get("language_pairs", {})

            print(f"\n{tok_name}")
            print("-" * 50)

            if not pairs:
                print("  (no language pairs available)")
                continue

            # Build full symmetric JSD matrix for display
            col_w = max(6, max(len(l) for l in languages) + 1)
            row_label_w = col_w

            # Header row
            header = f"{'':>{row_label_w}}"
            for lang in languages:
                header += f"  {lang:>{col_w}}"
            print(header)

            for l1 in languages:
                row = f"{l1:>{row_label_w}}"
                for l2 in languages:
                    if l1 == l2:
                        row += f"  {'—':>{col_w}}"
                    else:
                        key = f"{l1}_vs_{l2}" if f"{l1}_vs_{l2}" in pairs else f"{l2}_vs_{l1}"
                        jsd = pairs[key]["jsd"] if key in pairs else float("nan")
                        row += f"  {jsd:>{col_w}.4f}"
                print(row)

            # Ranked summary: sort pairs by JSD ascending
            ranked = sorted(pairs.values(), key=lambda p: p["jsd"])
            print(f"\n  Most overlapping pairs (lowest JSD):")
            for p in ranked[:5]:
                print(
                    f"    {p['language_1']:>6} ↔ {p['language_2']:<6}  "
                    f"JSD={p['jsd']:.4f}  overlap={p['overlap']:.4f}  "
                    f"shared_types={p['shared_types']:,}"
                )
            if len(ranked) > 5:
                print(f"  Least overlapping pairs (highest JSD):")
                for p in ranked[-min(5, len(ranked)):]:
                    print(
                        f"    {p['language_1']:>6} ↔ {p['language_2']:<6}  "
                        f"JSD={p['jsd']:.4f}  overlap={p['overlap']:.4f}  "
                        f"shared_types={p['shared_types']:,}"
                    )
