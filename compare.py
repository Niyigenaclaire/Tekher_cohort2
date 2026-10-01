"""
Before/After comparison: TF-IDF chatbot vs Word2Vec embedding chatbot.

Runs both chatbots against the same set of test questions and prints
a side-by-side metric table showing which approach performs better.

Usage:
    python compare.py
    python compare.py --faq data/faq.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from chatbot import FAQChatbot
from chatbot_embeddings import EmbeddingFAQChatbot


# ---------------------------------------------------------------------------
# Test suite
# Each entry: (user_question, expected_answer_substring, should_match)
# should_match=True  → a real answer is expected
# should_match=False → fallback "I'm sorry" is expected
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str, bool]] = [
    # --- exact / near-exact phrasing ---
    ("What is your return policy?",             "30 days",              True),
    ("How do I track my order?",                "tracking number",      True),
    ("What payment methods do you accept?",     "Visa",                 True),
    ("How long does shipping take?",            "business days",        True),
    ("How do I reset my password?",             "Forgot Password",      True),

    # --- paraphrased / synonym phrasing (tests semantic understanding) ---
    ("Can I get a refund?",                     "30 days",              True),
    ("Where is my package?",                    "tracking number",      True),
    ("Which cards can I pay with?",             "Visa",                 True),
    ("How fast is delivery?",                   "business days",        True),
    ("I forgot my password",                    "Forgot Password",      True),
    ("Do you deliver abroad?",                  "50 countries",         True),
    ("Is my data private?",                     "SSL",                  True),
    ("Any promo codes available?",              "newsletter",           True),
    ("What if my item arrives broken?",         "photo",                True),
    ("How do I make an account?",               "Sign Up",              True),

    # --- out-of-scope (should trigger fallback) ---
    ("What is the capital of France?",          "sorry",                False),
    ("xyzzy random gibberish",                  "sorry",                False),
    ("Tell me a joke",                          "sorry",                False),
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(
    chatbot: FAQChatbot | EmbeddingFAQChatbot,
    tests: list[tuple[str, str, bool]],
) -> dict[str, object]:
    """
    Run all test cases and return a metrics dict.

    Metrics
    -------
    accuracy      : fraction of cases where the correct behaviour occurred
    precision     : TP / (TP + FP)  among cases predicted as "matched"
    recall        : TP / (TP + FN)  among cases that should match
    f1            : harmonic mean of precision and recall
    avg_score     : mean similarity score across all queries
    per_case      : list of per-question result dicts (for the detail table)
    """
    results = []
    for question, keyword, should_match in tests:
        answer, score = chatbot.get_best_match(question)
        answered = "sorry" not in answer.lower()
        correct_keyword = keyword.lower() in answer.lower()

        # True Positive  : should match AND did match AND keyword found
        # True Negative  : should NOT match AND fallback returned
        # False Positive : should NOT match BUT an answer returned
        # False Negative : should match BUT fallback returned
        if should_match:
            tp = int(answered and correct_keyword)
            fn = int(not (answered and correct_keyword))
            fp = 0
            tn = 0
            correct = bool(answered and correct_keyword)
        else:
            tn = int(not answered)
            fp = int(answered)
            tp = 0
            fn = 0
            correct = not answered

        results.append({
            "question":      question,
            "expected":      keyword,
            "answer":        answer[:70] + ("..." if len(answer) > 70 else ""),
            "score":         score,
            "should_match":  should_match,
            "correct":       correct,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    total_correct = sum(r["correct"] for r in results)

    accuracy  = total_correct / len(results)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    avg_score = float(np.mean([r["score"] for r in results]))

    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "avg_score": avg_score,
        "per_case":  results,
    }


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

W = 36  # column width for questions


def print_detail_table(
    tfidf_results: list[dict],
    embed_results: list[dict],
) -> None:
    header = (
        f"{'#':<3} {'Question':<{W}} "
        f"{'TF-IDF':^14} {'W2V':^14} {'Expected keyword'}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for i, (tr, er) in enumerate(zip(tfidf_results, embed_results), 1):
        t_mark = "✓" if tr["correct"] else "✗"
        e_mark = "✓" if er["correct"] else "✗"
        q = tr["question"]
        if len(q) > W:
            q = q[:W - 3] + "..."
        print(
            f"{i:<3} {q:<{W}} "
            f"{t_mark} {tr['score']:>6.3f}      "
            f"{e_mark} {er['score']:>6.3f}      "
            f"{tr['expected']}"
        )

    print("=" * len(header))


def print_summary(tfidf: dict, embed: dict) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "avg_score"]
    labels  = ["Accuracy ", "Precision", "Recall   ", "F1 Score ", "Avg Sim  "]

    print("\n┌─────────────────────────────────────────────────────┐")
    print("│          Before / After Comparison Summary          │")
    print("├─────────────┬──────────────┬──────────────┬─────────┤")
    print("│ Metric      │   TF-IDF     │  Word2Vec    │  Delta  │")
    print("├─────────────┼──────────────┼──────────────┼─────────┤")
    for label, key in zip(labels, metrics):
        t_val = float(tfidf[key])
        e_val = float(embed[key])
        delta = e_val - t_val
        sign  = "+" if delta >= 0 else ""
        print(
            f"│ {label} │   {t_val:.4f}     │   {e_val:.4f}     │ {sign}{delta:.4f} │"
        )
    print("└─────────────┴──────────────┴──────────────┴─────────┘")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF vs Word2Vec FAQ chatbot on the same test set."
    )
    parser.add_argument("--faq", type=Path, default=Path("data/faq.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n── Loading TF-IDF chatbot ──────────────────────────────")
    tfidf_bot = FAQChatbot(faq_path=args.faq, threshold=0.15)

    print("\n── Loading Word2Vec chatbot ────────────────────────────")
    embed_bot = EmbeddingFAQChatbot(faq_path=args.faq, threshold=0.60)

    print(f"\n── Running {len(TEST_CASES)} test cases on both chatbots ──────────────")
    tfidf_metrics = evaluate(tfidf_bot,  TEST_CASES)
    embed_metrics = evaluate(embed_bot,  TEST_CASES)

    print_detail_table(tfidf_metrics["per_case"], embed_metrics["per_case"])
    print_summary(tfidf_metrics, embed_metrics)

    # save results to CSV for the README
    import csv
    out = Path("comparison_results.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "tfidf", "word2vec", "delta"])
        for key in ["accuracy", "precision", "recall", "f1", "avg_score"]:
            t = float(tfidf_metrics[key])
            e = float(embed_metrics[key])
            writer.writerow([key, f"{t:.4f}", f"{e:.4f}", f"{e - t:+.4f}"])
    print(f"\n  Saved numeric results to {out}")


if __name__ == "__main__":
    main()
