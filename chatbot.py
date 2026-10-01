"""
Simple FAQ Chatbot using TF-IDF Similarity.

Matches a user's question to the closest stored FAQ entry
using cosine similarity on TF-IDF vectors.

Usage:
    python chatbot.py                          # uses default data/faq.json
    python chatbot.py --faq data/faq.json      # explicit path
    python chatbot.py --threshold 0.2          # adjust confidence cutoff
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def preprocess(text: str) -> str:
    """Return a cleaned, lowercased string (keeps punctuation stripped)."""
    return " ".join(tokenize(text))


# ---------------------------------------------------------------------------
# FAQ store
# ---------------------------------------------------------------------------

class FAQChatbot:
    """
    Loads a JSON FAQ file and answers questions by finding the FAQ entry
    whose question has the highest TF-IDF cosine similarity to the user input.
    """

    def __init__(self, faq_path: Path, threshold: float = 0.15) -> None:
        self.threshold = threshold
        self.questions: list[str] = []
        self.answers: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.faq_matrix = None

        self._load(faq_path)
        self._build_index()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        """Parse the JSON FAQ file into parallel question/answer lists."""
        if not path.exists():
            raise FileNotFoundError(f"FAQ file not found: {path}")

        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list) or not data:
            raise ValueError("FAQ file must be a non-empty JSON array.")

        for i, entry in enumerate(data):
            if "question" not in entry or "answer" not in entry:
                raise ValueError(
                    f"Entry {i} is missing 'question' or 'answer' key."
                )
            self.questions.append(str(entry["question"]))
            self.answers.append(str(entry["answer"]))

        print(f"  Loaded {len(self.questions)} FAQ entries from '{path}'.")

    def _build_index(self) -> None:
        """Fit TF-IDF on the FAQ questions and store the matrix."""
        cleaned = [preprocess(q) for q in self.questions]
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),   # unigrams + bigrams for better matching
        )
        self.faq_matrix = self.vectorizer.fit_transform(cleaned)

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def get_best_match(self, user_input: str) -> tuple[str, float]:
        """
        Return (answer, similarity_score) for the closest FAQ entry.
        Returns a fallback message when no entry clears the threshold.
        """
        cleaned_input = preprocess(user_input)
        if not cleaned_input.strip():
            return "Please type a question so I can help you.", 0.0

        query_vector = self.vectorizer.transform([cleaned_input])
        similarities = cosine_similarity(query_vector, self.faq_matrix).flatten()

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < self.threshold:
            return (
                "I'm sorry, I don't have an answer for that. "
                "Please try rephrasing, or contact support directly.",
                best_score,
            )

        return self.answers[best_idx], best_score

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def top_matches(self, user_input: str, n: int = 3) -> list[tuple[str, float]]:
        """Return the top-n (question, score) pairs — useful for debugging."""
        cleaned_input = preprocess(user_input)
        query_vector = self.vectorizer.transform([cleaned_input])
        similarities = cosine_similarity(query_vector, self.faq_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:n]
        return [(self.questions[i], float(similarities[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════════╗
║           Simple FAQ Chatbot  (TF-IDF)               ║
║  Type your question and press Enter to get an answer ║
║  Commands:  'quit' or 'exit' to stop                 ║
║             'help' to list all FAQ topics            ║
║             'debug' to see similarity scores         ║
╚══════════════════════════════════════════════════════╝
"""


def run_cli(chatbot: FAQChatbot, debug: bool = False) -> None:
    """Interactive question-answering loop."""
    print(BANNER)

    debug_mode = debug  # can be toggled at runtime with 'debug'

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        # ---- built-in commands ----------------------------------------
        if lower in ("quit", "exit", "bye"):
            print("Bot: Goodbye! Have a great day.")
            break

        if lower == "help":
            print("\nBot: Here are the topics I can help with:\n")
            for i, q in enumerate(chatbot.questions, 1):
                print(f"  {i:>2}. {q}")
            print()
            continue

        if lower == "debug":
            debug_mode = not debug_mode
            state = "ON" if debug_mode else "OFF"
            print(f"Bot: Debug mode {state}.\n")
            continue

        # ---- normal answer --------------------------------------------
        answer, score = chatbot.get_best_match(user_input)
        print(f"\nBot: {answer}\n")

        if debug_mode:
            print("  [debug] Top matches:")
            for q, s in chatbot.top_matches(user_input):
                print(f"         {s:.4f}  {q}")
            print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple FAQ Chatbot using TF-IDF cosine similarity."
    )
    parser.add_argument(
        "--faq",
        type=Path,
        default=Path("data/faq.json"),
        help="Path to the FAQ JSON file (default: data/faq.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Minimum cosine similarity score to return an answer (default: 0.15)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Start with debug mode on (shows similarity scores for each query)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be between 0.0 and 1.0")

    print("\nInitialising FAQ Chatbot...")
    try:
        chatbot = FAQChatbot(faq_path=args.faq, threshold=args.threshold)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"Error loading FAQ data: {err}") from err

    run_cli(chatbot, debug=args.debug)


if __name__ == "__main__":
    main()
