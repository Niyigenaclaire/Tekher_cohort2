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
    page_title="FAQ Chatbot — NLP Assignment",
    page_icon="🤖",
    layout="centered",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

  html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #f0f4ff;
  }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, #1a3f7a 0%, #2563eb 60%, #3b82f6 100%);
    border-radius: 20px;
    padding: 36px 28px 28px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(26,63,122,0.18);
  }
  .hero-icon { font-size: 3.2rem; margin-bottom: 10px; }
  .hero h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.5px;
  }
  .hero p {
    color: #bfdbfe;
    font-size: 0.97rem;
    margin: 0;
  }

  /* ── Search bar area ── */
  .search-wrap {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 22px;
    box-shadow: 0 2px 12px rgba(26,63,122,0.08);
    border: 1.5px solid #dbeafe;
  }
  .search-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: #1a3f7a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
  }

  /* ── Cards ── */
  .card {
    background: #ffffff;
    border-radius: 16px;
    padding: 22px 22px 18px;
    min-height: 200px;
    box-shadow: 0 4px 18px rgba(26,63,122,0.10);
    border: 1.5px solid #dbeafe;
    height: 100%;
  }
  .card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 8px;
    font-size: 0.7rem;
    font-weight: 800;
    color: #fff;
  }
  .badge-v1 { background: #1a3f7a; }
  .badge-v2 { background: #2563eb; }
  .card-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #1e3a5f;
  }
  .answer-text {
    font-size: 0.97rem;
    color: #1e293b;
    line-height: 1.65;
    font-weight: 500;
    margin-bottom: 16px;
  }
  .answer-text.placeholder { color: #94a3b8; font-style: italic; }

  /* score bar */
  .score-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 4px;
  }
  .score-bar-bg {
    flex: 1;
    height: 6px;
    background: #e2e8f0;
    border-radius: 99px;
    overflow: hidden;
  }
  .score-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    transition: width 0.4s ease;
  }
  .score-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #1a3f7a;
    white-space: nowrap;
  }

  /* ── Divider tag ── */
  .section-tag {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 26px 0 14px;
  }
  .section-tag span {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    white-space: nowrap;
  }
  .section-tag hr {
    flex: 1;
    border: none;
    border-top: 1.5px solid #e2e8f0;
  }

  /* ── History item ── */
  .hist-item {
    background: #ffffff;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border: 1.5px solid #dbeafe;
    font-size: 0.85rem;
    color: #475569;
    box-shadow: 0 1px 6px rgba(26,63,122,0.05);
  }
  .hist-item strong { color: #0f172a; font-size: 0.9rem; }
  .hist-chip {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 1px 7px;
    border-radius: 99px;
    margin-right: 4px;
  }
  .chip-v1 { background: #dbeafe; color: #1a3f7a; }
  .chip-v2 { background: #eff6ff; color: #2563eb; }

  /* ── Tip box ── */
  .tip-box {
    background: #eff6ff;
    border: 1.5px solid #bfdbfe;
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.83rem;
    color: #1e3a5f;
    line-height: 1.6;
  }
  .tip-box b { color: #1a3f7a; }

  /* ── Streamlit overrides ── */
  .stButton > button {
    background: linear-gradient(135deg, #1a3f7a, #2563eb) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    width: 100%;
    border: none !important;
    padding: 0.6rem 1rem !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
    transition: opacity 0.2s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  div[data-testid="stTextInput"] input {
    border-radius: 10px !important;
    border: 2px solid #bfdbfe !important;
    background: #f8faff !important;
    font-size: 1rem !important;
    color: #0f172a !important;
    padding: 10px 14px !important;
  }
  div[data-testid="stTextInput"] input:focus {
    border-color: #2563eb !important;
    background: #ffffff !important;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load chatbots (cached — only loads once)
# ---------------------------------------------------------------------------
FAQ_PATH = Path("data/faq.json")

@st.cache_resource
def load_bots():
    tfidf = FAQChatbot(faq_path=FAQ_PATH, threshold=0.15)
    embed = EmbeddingFAQChatbot(faq_path=FAQ_PATH, threshold=0.40)
    return tfidf, embed

tfidf_bot, embed_bot = load_bots()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="hero-icon">🤖</div>
  <h1>FAQ Chatbot</h1>
  <p>NLP Assignment &nbsp;·&nbsp; TF-IDF vs Word Embeddings &nbsp;·&nbsp; Side-by-side comparison</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tip box
# ---------------------------------------------------------------------------
st.markdown("""
<div class="tip-box">
  💡 <b>Try these questions:</b>
  &nbsp; "Can I get a refund?" &nbsp;·&nbsp; "Where is my package?" &nbsp;·&nbsp;
  "Any promo codes?" &nbsp;·&nbsp; "Is my data safe?" &nbsp;·&nbsp; "How fast is delivery?"
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        label="question",
        placeholder="Type your question here and press Ask →",
        label_visibility="collapsed",
    )
with col_btn:
    ask_clicked = st.button("Ask →")

# ---------------------------------------------------------------------------
# Answer cards
# ---------------------------------------------------------------------------
def score_bar(score: float) -> str:
    pct = int(score * 100)
    return f"""
    <div class="score-row">
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct}%"></div>
      </div>
      <span class="score-label">{score:.3f}</span>
    </div>
    """

if ask_clicked and question.strip():
    q = question.strip()

    tfidf_answer, tfidf_score = tfidf_bot.get_best_match(q)
    embed_answer, embed_score = embed_bot.get_best_match(q)

    st.session_state.history.insert(0, {
        "q": q,
        "tfidf_answer": tfidf_answer,
        "tfidf_score": tfidf_score,
        "embed_answer": embed_answer,
        "embed_score": embed_score,
    })

    left, right = st.columns(2, gap="medium")

    with left:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">
            <span class="badge badge-v1">V1</span>
            <span class="card-label">TF-IDF (Baseline)</span>
          </div>
          <div class="answer-text">{tfidf_answer}</div>
          <div style="font-size:0.72rem;color:#64748b;margin-bottom:4px">Confidence</div>
          {score_bar(tfidf_score)}
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">
            <span class="badge badge-v2">V2</span>
            <span class="card-label">Word Embeddings</span>
          </div>
          <div class="answer-text">{embed_answer}</div>
          <div style="font-size:0.72rem;color:#64748b;margin-bottom:4px">Confidence</div>
          {score_bar(embed_score)}
        </div>
        """, unsafe_allow_html=True)

elif ask_clicked and not question.strip():
    st.warning("Please type a question first.")

else:
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("""
        <div class="card">
          <div class="card-header">
            <span class="badge badge-v1">V1</span>
            <span class="card-label">TF-IDF (Baseline)</span>
          </div>
          <div class="answer-text placeholder">Answer will appear here after you ask a question...</div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="card">
          <div class="card-header">
            <span class="badge badge-v2">V2</span>
            <span class="card-label">Word Embeddings</span>
          </div>
          <div class="answer-text placeholder">Answer will appear here after you ask a question...</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Metrics strip (static — from compare.py results)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="section-tag">
  <hr/><span>📊 Before / After Metrics</span><hr/>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
metrics = [
    ("Accuracy",  "66.7%", "72.2%", "+5.6%"),
    ("Recall",    "73.3%", "80.0%", "+6.7%"),
    ("F1 Score",  "78.6%", "82.8%", "+4.2%"),
    ("Avg Sim",   "0.582", "0.809", "+22.7%"),
]
for col, (label, before, after, delta) in zip([m1, m2, m3, m4], metrics):
    with col:
        st.markdown(f"""
        <div style="background:#ffffff;border:1.5px solid #dbeafe;border-radius:12px;
                    padding:14px 12px;text-align:center;box-shadow:0 2px 8px rgba(26,63,122,0.07)">
          <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.08em;color:#64748b;margin-bottom:6px">{label}</div>
          <div style="font-size:0.82rem;color:#94a3b8;text-decoration:line-through">{before}</div>
          <div style="font-size:1.3rem;font-weight:800;color:#1a3f7a">{after}</div>
          <div style="font-size:0.78rem;font-weight:700;color:#16a34a">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("""
    <div class="section-tag">
      <hr/><span>🕑 Previous Questions</span><hr/>
    </div>
    """, unsafe_allow_html=True)

    for item in st.session_state.history:
        short_t = item["tfidf_answer"][:90] + ("…" if len(item["tfidf_answer"]) > 90 else "")
        short_e = item["embed_answer"][:90] + ("…" if len(item["embed_answer"]) > 90 else "")
        st.markdown(f"""
        <div class="hist-item">
          <strong>❓ {item['q']}</strong><br><br>
          <span class="hist-chip chip-v1">V1 · {item['tfidf_score']:.3f}</span> {short_t}<br><br>
          <span class="hist-chip chip-v2">V2 · {item['embed_score']:.3f}</span> {short_e}
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center;margin-top:40px;padding-top:20px;
            border-top:1.5px solid #e2e8f0;font-size:0.78rem;color:#94a3b8">
  NLP Assignment &nbsp;·&nbsp; FAQ Chatbot &nbsp;·&nbsp; TF-IDF vs PMI-SVD Word Embeddings
</div>
""", unsafe_allow_html=True)
