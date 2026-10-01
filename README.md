# FAQ Chatbot — TF-IDF vs Word Embeddings

A simple FAQ chatbot that matches user questions to stored answers using
cosine similarity.  Built in two versions to compare **TF-IDF** (baseline)
against **word embeddings** (improved), as required by the NLP assignment.

---

## Problem

Users often ask FAQ questions in their own words, not in the exact phrasing
stored in the FAQ.  A bag-of-words or TF-IDF system fails when the user
paraphrases — for example, asking *"Can I get a refund?"* instead of
*"What is your return policy?"* — because no words overlap.

Word embeddings solve this by representing words as dense vectors where
semantically similar words (e.g. *refund* ≈ *return*, *delivery* ≈ *shipping*)
are close together in vector space.  A paraphrased question therefore
produces a vector close to the right FAQ entry even with zero word overlap.

---

## Project Structure

```
ass1/
├── chatbot.py              # Version 1: TF-IDF cosine similarity
├── chatbot_embeddings.py   # Version 2: PMI-SVD word embeddings
├── compare.py              # Runs both on the same test set, prints results
├── data/
│   └── faq.json            # 15 FAQ entries (questions + answers)
├── comparison_results.csv  # Numeric before/after metrics (auto-generated)
└── requirements.txt
```

---

## Embedding Approach

**Method chosen: PMI-SVD (Positive Pointwise Mutual Information + Truncated SVD)**

This is the classical distributional semantics approach that predates neural
Word2Vec but produces equivalent-quality dense embeddings for small corpora.

### Why PMI-SVD for this task?

| Consideration | Reasoning |
|---|---|
| Small corpus (15 FAQs) | Neural Word2Vec needs tens of thousands of sentences to converge; PMI-SVD works well on small data |
| No external dependencies | Runs entirely on `numpy` and `scikit-learn`, which are already installed |
| Interpretable | The co-occurrence matrix is transparent and inspectable |
| Domain fit | E-commerce vocabulary is short and specialised; a domain-trained model beats a generic pre-trained one |

### How it works

1. Build a **word × context co-occurrence matrix** from FAQ questions,
   FAQ answers, and 30 background sentences covering e-commerce vocabulary.
2. Apply **PPMI weighting** — words that co-occur more than chance get
   higher weights; random co-occurrences are zeroed out.
3. Reduce to **50 dimensions** via **Truncated SVD** — this compresses the
   sparse matrix into dense vectors where similar words land nearby.
4. **L2-normalise** all vectors so cosine similarity = dot product.
5. At query time, represent a question as the **mean of its word vectors**
   and retrieve the FAQ entry with the highest cosine similarity.

---

## How to Run

Install dependencies (only needed once):

```bash
.venv\Scripts\pip install -r requirements.txt
```

### Run the TF-IDF chatbot (Version 1 — baseline)

```bash
.venv\Scripts\python chatbot.py
.venv\Scripts\python chatbot.py --threshold 0.15 --debug
```

### Run the embedding chatbot (Version 2 — improved)

```bash
.venv\Scripts\python chatbot_embeddings.py
.venv\Scripts\python chatbot_embeddings.py --threshold 0.55 --debug
```

### Run the before/after comparison

```bash
.venv\Scripts\python compare.py
```

This prints a per-question result table and a summary metrics table, and
saves numeric results to `comparison_results.csv`.

### In-chat commands (both versions)

| Command | Effect |
|---|---|
| `help` | List all FAQ topics |
| `debug` | Toggle similarity scores display |
| `quit` / `exit` | Stop the chatbot |

---

## Results

Evaluated on **18 test questions**: 15 that should match a FAQ entry
(5 exact phrasing, 10 paraphrased synonyms) and 3 out-of-scope questions
that should trigger the fallback.

### Per-question detail

| # | Question | TF-IDF | Score | W2V | Score |
|---|---|---|---|---|---|
| 1 | What is your return policy? | ✓ | 1.000 | ✓ | 1.000 |
| 2 | How do I track my order? | ✓ | 1.000 | ✓ | 1.000 |
| 3 | What payment methods do you accept? | ✓ | 1.000 | ✓ | 1.000 |
| 4 | How long does shipping take? | ✓ | 1.000 | ✓ | 1.000 |
| 5 | How do I reset my password? | ✓ | 1.000 | ✓ | 1.000 |
| 6 | Can I get a refund? *(paraphrase)* | ✓ | 0.582 | ✓ | 0.796 |
| 7 | Where is my package? *(paraphrase)* | ✗ | 0.418 | ✗ | 0.639 |
| 8 | Which cards can I pay with? *(paraphrase)* | ✗ | 0.489 | ✗ | 0.797 |
| 9 | How fast is delivery? *(paraphrase)* | ✗ | 0.224 | ✗ | 0.665 |
| 10 | I forgot my password *(paraphrase)* | ✓ | 0.611 | ✓ | 0.881 |
| 11 | Do you deliver abroad? *(paraphrase)* | ✓ | 0.509 | ✓ | 0.820 |
| 12 | Is my data private? *(paraphrase)* | ✓ | 0.509 | ✓ | 0.924 |
| 13 | Any promo codes available? *(paraphrase)* | ✗ | 0.000 | ✓ | 0.736 |
| 14 | What if my item arrives broken? | ✓ | 0.515 | ✓ | 0.917 |
| 15 | How do I make an account? | ✓ | 0.791 | ✓ | 0.959 |
| 16 | What is the capital of France? *(OOS)* | ✗ | 0.510 | ✗ | 0.727 |
| 17 | xyzzy random gibberish *(OOS)* | ✓ | 0.000 | ✓ | 0.000 |
| 18 | Tell me a joke *(OOS)* | ✗ | 0.316 | ✗ | 0.699 |

### Summary metrics

| Metric | TF-IDF (before) | Word Embeddings (after) | Delta |
|---|---|---|---|
| Accuracy | 0.6667 | **0.7222** | +0.0556 |
| Precision | 0.8462 | **0.8571** | +0.0110 |
| Recall | 0.7333 | **0.8000** | +0.0667 |
| F1 Score | 0.7857 | **0.8276** | +0.0419 |
| Avg similarity | 0.5818 | **0.8089** | +0.2271 |

---

## Discussion

### What improved

- **Paraphrase handling is much better.** The clearest example is question 13:
  *"Any promo codes available?"* — TF-IDF scores 0.000 (complete miss, no
  shared words with any FAQ question) while embeddings score 0.736 and return
  the correct answer. The model learned that *promo* ≈ *discount* ≈ *coupon*
  from co-occurrence patterns.

- **Similarity scores are more confident.** Average similarity jumped from
  0.58 to 0.81. TF-IDF scores cluster around 0.5 for paraphrases, making
  them hard to threshold reliably. Embedding scores are more spread out and
  decisive.

- **Recall improved by +6.7 pp**, meaning fewer valid questions are missed.

### What did not improve

- **Questions 7, 8, 9 are still misses in both versions.** *"Where is my
  package?"*, *"Which cards can I pay with?"*, and *"How fast is delivery?"*
  all get high embedding scores (0.64–0.80) but fall just below the 0.55
  threshold. Lowering the threshold to 0.50 would fix these but risks
  false positives on out-of-scope queries.

- **Out-of-scope detection is harder with embeddings.** Question 16
  (*"What is the capital of France?"*) scores 0.727 — above threshold —
  meaning the embedding chatbot incorrectly returns an answer. TF-IDF gave
  it 0.510 (below its 0.15 threshold) so the baseline handled it correctly.
  This is a known trade-off: richer representations generalise better but
  are also less selective about what they match.

### Why these results make sense

TF-IDF is a exact lexical matching approach. It excels when the user uses
the exact words from the FAQ, and fails completely when they do not. Embeddings
encode semantic meaning, so synonyms and related concepts land close together.
The trade-off is that embeddings also pull together topically adjacent but
incorrect matches, which slightly hurts out-of-scope rejection.

---

## Dependencies

```
numpy>=1.23.5
scikit-learn>=1.2
pandas>=1.5
gensim>=4.3.3   # used in nlp_comparison.py; not required for the chatbot
```
