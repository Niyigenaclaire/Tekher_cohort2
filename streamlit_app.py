"""
FAQ Chatbot — Streamlit Web App
TF-IDF vs Word Embeddings side by side.

Deploy on Streamlit Cloud:
    Main file: streamlit_app.py
"""

from pathlib import Path
import streamlit as st
from chatbot import FAQChatbot
from chatbot_embeddings import EmbeddingFAQChatbot

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="💬",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS — white + navy blue (projection-friendly)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #ffffff; }

  /* Title */
  h1 { color: #0f1f3d !important; text-align: center; }

  /* Subtitle */
  .subtitle {
    text-align: center;
    color: #4a5e7a;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
  }

  /* Cards */
  .card {
    background: #f4f7fb;
    border: 2px solid #dce6f0;
    border-radius: 14px;
    padding: 20px 22px;
    min-height: 160px;
  }

  .card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #1a3f7a;
    margin-bottom: 10px;
  }

  .badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 99px;
    background: #1a3f7a;
    color: #ffffff;
    margin-right: 6px;
  }

  .answer-text {
    font-size: 1rem;
    color: #0f1f3d;
    line-height: 1.6;
    font-weight: 500;
  }

  .score-text {
    font-size: 0.78rem;
    color: #4a5e7a;
    margin-top: 10px;
  }

  .score-text b { color: #1a3f7a; }

  /* History item */
  .hist-item {
    background: #f4f7fb;
    border: 1px solid #dce6f0;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.87rem;
    color: #4a5e7a;
  }
  .hist-item strong { color: #0f1f3d; }

  /* Button override */
  .stButton > button {
    background-color: #1a3f7a !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100%;
  }
  .stButton > button:hover {
    background-color: #0f2a57 !important;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load chatbots (cached so they only load once)
# ---------------------------------------------------------------------------
FAQ_PATH = Path("data/faq.json")

@st.cache_resource
def load_bots():
    tfidf = FAQChatbot(faq_path=FAQ_PATH, threshold=0.15)
    embed = EmbeddingFAQChatbot(faq_path=FAQ_PATH, threshold=0.40)
    return tfidf, embed

tfidf_bot, embed_bot = load_bots()

# ---------------------------------------------------------------------------
# Session state for history
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("💬 FAQ Chatbot")
st.markdown('<p class="subtitle">Type a question — see TF-IDF vs Word Embeddings side by side</p>',
            unsafe_allow_html=True)

# Input row
col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        label="question",
        placeholder="e.g. Can I get a refund?",
        label_visibility="collapsed",
    )
with col_btn:
    ask_clicked = st.button("Ask")

# ---------------------------------------------------------------------------
# Answer cards
# ---------------------------------------------------------------------------
if ask_clicked and question.strip():
    q = question.strip()

    tfidf_answer, tfidf_score = tfidf_bot.get_best_match(q)
    embed_answer, embed_score = embed_bot.get_best_match(q)

    # save to history
    st.session_state.history.insert(0, {
        "q": q,
        "tfidf_answer": tfidf_answer,
        "tfidf_score": tfidf_score,
        "embed_answer": embed_answer,
        "embed_score": embed_score,
    })

    left, right = st.columns(2)

    with left:
        st.markdown(f"""
        <div class="card">
          <div class="card-title"><span class="badge">V1</span> TF-IDF</div>
          <div class="answer-text">{tfidf_answer}</div>
          <div class="score-text">Similarity score: <b>{tfidf_score:.3f}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="card">
          <div class="card-title"><span class="badge">V2</span> Word Embeddings</div>
          <div class="answer-text">{embed_answer}</div>
          <div class="score-text">Similarity score: <b>{embed_score:.3f}</b></div>
        </div>
        """, unsafe_allow_html=True)

elif ask_clicked and not question.strip():
    st.warning("Please type a question first.")

else:
    # placeholder cards before first question
    left, right = st.columns(2)
    with left:
        st.markdown("""
        <div class="card">
          <div class="card-title"><span class="badge">V1</span> TF-IDF</div>
          <div class="answer-text" style="color:#8fa3c0;font-style:italic">Answer will appear here...</div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="card">
          <div class="card-title"><span class="badge">V2</span> Word Embeddings</div>
          <div class="answer-text" style="color:#8fa3c0;font-style:italic">Answer will appear here...</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("---")
    st.markdown("#### Previous questions")
    for item in st.session_state.history:
        short_t = item["tfidf_answer"][:80] + ("…" if len(item["tfidf_answer"]) > 80 else "")
        short_e = item["embed_answer"][:80] + ("…" if len(item["embed_answer"]) > 80 else "")
        st.markdown(f"""
        <div class="hist-item">
          <strong>{item['q']}</strong><br>
          TF-IDF ({item['tfidf_score']:.3f}): {short_t}<br>
          Embeddings ({item['embed_score']:.3f}): {short_e}
        </div>
        """, unsafe_allow_html=True)
