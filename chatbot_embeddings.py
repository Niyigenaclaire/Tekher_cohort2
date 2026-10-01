"""
FAQ Chatbot using Word Embeddings (improved version).

Approach: trains a lightweight Word2Vec-style co-occurrence embedding
using only numpy and scikit-learn (no gensim required), so it works on
any Python version.

Each FAQ question is represented as the mean of its word vectors.
Retrieval uses cosine similarity — the same metric used by the TF-IDF
chatbot, so the two approaches are directly comparable.

Why embeddings over TF-IDF?
  TF-IDF treats every word as independent and unrelated.  Embeddings
  place semantically similar words (e.g. "refund" and "return") close
  together in vector space, so paraphrased questions can still match
  the right FAQ entry even when they share no exact words.

Usage:
    python chatbot_embeddings.py                       # default data/faq.json
    python chatbot_embeddings.py --faq data/faq.json
    python chatbot_embeddings.py --threshold 0.55
    python chatbot_embeddings.py --debug
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Tokeniser  (identical to chatbot.py for fair comparison)
# ---------------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def preprocess(text: str) -> str:
    return " ".join(tokenize(text))


# ---------------------------------------------------------------------------
# Small general-purpose corpus to enrich embedding coverage.
# These sentences cover vocabulary that FAQ questions touch but may not
# repeat enough times on their own (15 FAQ entries is a small corpus).
# ---------------------------------------------------------------------------

BACKGROUND_SENTENCES: list[str] = [
    "how do i find my order status and delivery date",
    "i want to return or exchange a product i bought",
    "what are the accepted payment options for online shopping",
    "how can i contact customer service or support team",
    "i forgot my account password and need to reset it",
    "does the store ship to international locations worldwide",
    "how long does standard or express shipping take",
    "is my credit card and personal data safe and secure",
    "do you offer any discount codes or promotional coupons",
    "what is the warranty and guarantee on your products",
    "how do i create a new user account on the website",
    "can i modify or cancel an order after placing it",
    "where is the company headquarters and warehouse located",
    "i received a damaged or broken item what should i do",
    "sign up for newsletter to get exclusive deals and offers",
    "track package shipment using the tracking number sent by email",
    "return policy allows refunds within thirty days of purchase",
    "ssl encryption protects your personal information and payment details",
    "manufacturer warranty covers defects in materials and workmanship",
    "express delivery takes one to two business days",
    "refund money back guarantee satisfied customers",
    "shipping courier parcel delivery international domestic",
    "account login register signup email verification",
    "damaged broken defective replacement exchange compensation",
    "discount coupon promo sale seasonal offer newsletter",
    "secure privacy policy gdpr data protection",
    "visa mastercard paypal bank transfer credit debit",
    "track trace monitor status update shipment",
    "cancel modify change edit update order request",
    "warranty guarantee repair replace defect quality",
]


# ---------------------------------------------------------------------------
# Embedding model: PMI-SVD (a.k.a. GloVe-style count-based embeddings)
#
# This is the classical NLP approach before neural Word2Vec became popular.
# It builds a word-context co-occurrence matrix and factorises it with SVD
# to produce dense word vectors.  The result is semantically meaningful:
# words that appear in similar contexts get similar vectors.
# ---------------------------------------------------------------------------

class PMISVDEmbeddings:
    """
    Pointwise Mutual Information + Truncated SVD word embeddings.

    Steps:
      1. Build a word × context co-occurrence matrix from the corpus.
      2. Apply positive PMI weighting (PPMI).
      3. Reduce to `n_components` dimensions via truncated SVD.
      4. L2-normalise all word vectors.
    """

    def __init__(self, n_components: int = 50, window: int = 3) -> None:
        self.n_components = n_components
        self.window = window
        self.word2idx: dict[str, int] = {}
        self.vectors: np.ndarray | None = None   # shape (vocab, n_components)

    def fit(self, sentences: list[str]) -> "PMISVDEmbeddings":
        """Build embeddings from a list of raw text sentences."""
        # --- step 1: co-occurrence matrix via CountVectorizer + window trick
        # We use a bigram/trigram context approximation: each sentence is
        # split into overlapping windows and treated as a "document".
        windows = self._extract_windows(sentences)

        vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
        X = vectorizer.fit_transform(windows)       # shape: (windows, vocab)
        vocab = vectorizer.get_feature_names_out()
        self.word2idx = {w: i for i, w in enumerate(vocab)}

        # co-occurrence: word × context  (word appears in same window as context)
        # X.T @ X gives us that matrix
        cooc = (X.T @ X).toarray().astype(np.float32)
        np.fill_diagonal(cooc, 0)                   # remove self-co-occurrence

        # --- step 2: PPMI
        total = cooc.sum()
        row_sums = cooc.sum(axis=1, keepdims=True)
        col_sums = cooc.sum(axis=0, keepdims=True)
        # avoid division by zero
        expected = (row_sums @ col_sums) / (total + 1e-9)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log2((cooc + 1e-9) / (expected + 1e-9))
        ppmi = np.maximum(pmi, 0)                   # keep only positive PMI

        # --- step 3: truncated SVD
        n = min(self.n_components, ppmi.shape[1] - 1)
        svd = TruncatedSVD(n_components=n, random_state=42)
        raw_vectors = svd.fit_transform(ppmi)        # shape: (vocab, n)

        # --- step 4: L2 normalise
        self.vectors = normalize(raw_vectors, norm="l2")
        return self

    def _extract_windows(self, sentences: list[str]) -> list[str]:
        """
        Slide a context window over each sentence and return each window
        as a short string.  This is how we approximate word co-occurrence.
        """
        result = []
        for sent in sentences:
            tokens = tokenize(sent)
            for i in range(len(tokens)):
                start = max(0, i - self.window)
                end   = min(len(tokens), i + self.window + 1)
                window_tokens = tokens[start:end]
                result.append(" ".join(window_tokens))
        return result if result else [""]

    def get_vector(self, word: str) -> np.ndarray | None:
        idx = self.word2idx.get(word)
        if idx is None:
            return None
        return self.vectors[idx]

    def mean_vector(self, tokens: list[str]) -> np.ndarray:
        """Return the mean embedding for a list of tokens."""
        vecs = [self.get_vector(t) for t in tokens if self.get_vector(t) is not None]
        if not vecs:
            return np.zeros(self.n_components, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding FAQ chatbot
# ---------------------------------------------------------------------------

class EmbeddingFAQChatbot:
    """
    FAQ chatbot that represents questions as mean PMI-SVD word vectors
    and retrieves answers via cosine similarity.
    """

    def __init__(
        self,
        faq_path: Path,
        threshold: float = 0.55,
        n_components: int = 50,
        window: int = 3,
    ) -> None:
        self.threshold = threshold
        self.questions: list[str] = []
        self.answers: list[str] = []
        self.embeddings: PMISVDEmbeddings | None = None
        self.faq_vectors: np.ndarray | None = None

        self._load(faq_path)
        self._train_embeddings(n_components, window)
        self._build_index()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"FAQ file not found: {path}")
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list) or not data:
            raise ValueError("FAQ file must be a non-empty JSON array.")
        for i, entry in enumerate(data):
            if "question" not in entry or "answer" not in entry:
                raise ValueError(f"Entry {i} missing 'question' or 'answer'.")
            self.questions.append(str(entry["question"]))
            self.answers.append(str(entry["answer"]))
        print(f"  Loaded {len(self.questions)} FAQ entries from '{path}'.")

    def _train_embeddings(self, n_components: int, window: int) -> None:
        """
        Train PMI-SVD embeddings on:
          - FAQ questions
          - FAQ answers   (so answer vocabulary is covered)
          - Background sentences (general e-commerce vocabulary)
        """
        corpus = self.questions + self.answers + BACKGROUND_SENTENCES
        self.embeddings = PMISVDEmbeddings(n_components=n_components, window=window)
        self.embeddings.fit(corpus)
        print(
            f"  Trained PMI-SVD embeddings: "
            f"vocab={len(self.embeddings.word2idx)}, "
            f"dims={n_components}, window={window}."
        )

    def _build_index(self) -> None:
        """Pre-compute mean embedding vectors for all FAQ questions."""
        self.faq_vectors = np.vstack([
            self.embeddings.mean_vector(tokenize(q))
            for q in self.questions
        ])

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def get_best_match(self, user_input: str) -> tuple[str, float]:
        """Return (answer, similarity_score) for the closest FAQ entry."""
        tokens = tokenize(user_input)
        if not tokens:
            return "Please type a question so I can help you.", 0.0

        query_vec = self.embeddings.mean_vector(tokens).reshape(1, -1)

        if not np.any(query_vec):
            return (
                "I'm sorry, I don't have an answer for that. "
                "Please try rephrasing, or contact support directly.",
                0.0,
            )

        sims = cosine_similarity(query_vec, self.faq_vectors).flatten()
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < self.threshold:
            return (
                "I'm sorry, I don't have an answer for that. "
                "Please try rephrasing, or contact support directly.",
                best_score,
            )

        return self.answers[best_idx], best_score

    def top_matches(self, user_input: str, n: int = 3) -> list[tuple[str, float]]:
        """Return top-n (question, score) pairs — useful for debugging."""
        tokens = tokenize(user_input)
        query_vec = self.embeddings.mean_vector(tokens).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.faq_vectors).flatten()
        top_idx = np.argsort(sims)[::-1][:n]
        return [(self.questions[i], float(sims[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------

BANNER = """
╔══════════════════════════════════════════════════════╗
║     FAQ Chatbot  — Word Embeddings (PMI-SVD)         ║
║  Type your question and press Enter to get an answer ║
║  Commands:  'quit' / 'exit' to stop                  ║
║             'help'  to list all FAQ topics           ║
║             'debug' to toggle similarity scores      ║
╚══════════════════════════════════════════════════════╝
"""


def run_cli(chatbot: EmbeddingFAQChatbot, debug: bool = False) -> None:
    print(BANNER)
    debug_mode = debug

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lower = user_input.lower()

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
            print(f"Bot: Debug mode {'ON' if debug_mode else 'OFF'}.\n")
            continue

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
        description="FAQ Chatbot using PMI-SVD word embeddings (no external dependencies)."
    )
    parser.add_argument("--faq",         type=Path,  default=Path("data/faq.json"))
    parser.add_argument("--threshold",   type=float, default=0.55)
    parser.add_argument("--n-components",type=int,   default=50)
    parser.add_argument("--window",      type=int,   default=3)
    parser.add_argument("--debug",       action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("\nInitialising Embedding FAQ Chatbot...")
    try:
        chatbot = EmbeddingFAQChatbot(
            faq_path=args.faq,
            threshold=args.threshold,
            n_components=args.n_components,
            window=args.window,
        )
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"Error: {err}") from err
    run_cli(chatbot, debug=args.debug)


if __name__ == "__main__":
    main()
