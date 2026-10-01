"""
FAQ Chatbot — Streamlit Web App
TF-IDF vs Word Embeddings side by side.
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
    page_icon="https://cdn-icons-png.flaticon.com/512/4712/4712035.png",
    layout="centered",
)

# ---------------------------------------------------------------------------
# CSS  — teal / charcoal / white palette, no blue, no emojis
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #f7f8fa;
  }

  /* ── Hero ── */
  .hero {
    background: linear-gradient(135deg, #0f4c4c 0%, #1a7a6e 60%, #22a899 100%);
    border-radius: 20px;
    padding: 36px 28px 30px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(15,76,76,0.18);
  }
  .hero img.hero-icon {
    width: 72px; height: 72px;
    border-radius: 16px;
    margin-bottom: 14px;
    background: rgba(255,255,255,0.15);
    padding: 8px;
  }
  .hero h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.5px;
  }
  .hero p { color: #a7f3d0; font-size: 0.95rem; margin: 0; }

  /* ── Tip box ── */
  .tip-box {
    background: #ecfdf5;
    border: 1.5px solid #6ee7b7;
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 0.84rem;
    color: #065f46;
    line-height: 1.6;
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }
  .tip-box img { width:20px; height:20px; margin-top:1px; flex-shrink:0; }

  /* ── Cards ── */
  .card {
    background: #ffffff;
    border-radius: 16px;
    padding: 22px 22px 18px;
    min-height: 210px;
    box-shadow: 0 4px 18px rgba(15,76,76,0.09);
    border: 1.5px solid #d1fae5;
  }
  .card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
  }
  .card-header img { width:28px; height:28px; }
  .card-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #0f4c4c;
  }
  .badge {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 99px;
    background: #0f4c4c;
    color: #fff;
    margin-left: 4px;
  }
  .answer-text {
    font-size: 0.97rem;
    color: #1a2e2e;
    line-height: 1.65;
    font-weight: 500;
    margin-bottom: 16px;
  }
  .answer-text.placeholder { color: #94a3b8; font-style: italic; }

  /* score bar */
  .score-row { display:flex; align-items:center; gap:10px; margin-top:4px; }
  .score-bar-bg {
    flex:1; height:6px; background:#e2e8f0;
    border-radius:99px; overflow:hidden;
  }
  .score-bar-fill {
    height:100%; border-radius:99px;
    background: linear-gradient(90deg, #0f4c4c, #22a899);
  }
  .score-label { font-size:0.75rem; font-weight:700; color:#0f4c4c; white-space:nowrap; }

  /* ── Metric cards ── */
  .metric-card {
    background: #ffffff;
    border: 1.5px solid #d1fae5;
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(15,76,76,0.07);
  }
  .metric-label {
    font-size: 0.67rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.08em;
    color:#64748b; margin-bottom:6px;
    display:flex; align-items:center; justify-content:center; gap:6px;
  }
  .metric-label img { width:14px; height:14px; }
  .metric-value {
    font-size:1.5rem; font-weight:800; color:#0f4c4c; line-height:1;
  }
  .metric-delta {
    font-size:0.78rem; font-weight:700; color:#059669; margin-top:4px;
  }

  /* ── Section divider ── */
  .section-tag {
    display:flex; align-items:center; gap:10px; margin:28px 0 14px;
  }
  .section-tag span {
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#64748b; white-space:nowrap;
    display:flex; align-items:center; gap:6px;
  }
  .section-tag span img { width:14px; height:14px; }
  .section-tag hr { flex:1; border:none; border-top:1.5px solid #e2e8f0; }

  /* ── History ── */
  .hist-item {
    background:#ffffff; border-radius:12px; padding:14px 18px;
    margin-bottom:10px; border:1.5px solid #d1fae5;
    font-size:0.85rem; color:#475569;
    box-shadow:0 1px 6px rgba(15,76,76,0.05);
  }
  .hist-item strong { color:#0f172a; font-size:0.9rem; }
  .hist-chip {
    display:inline-block; font-size:0.67rem; font-weight:700;
    padding:1px 7px; border-radius:99px; margin-right:4px;
  }
  .chip-v1 { background:#ccfbf1; color:#0f4c4c; }
  .chip-v2 { background:#d1fae5; color:#065f46; }

  /* ── Streamlit overrides ── */
  .stButton > button {
    background: linear-gradient(135deg,#0f4c4c,#1a7a6e) !important;
    color:white !important; border-radius:10px !important;
    font-weight:700 !important; font-size:0.95rem !important;
    width:100%; border:none !important;
    padding:0.6rem 1rem !important;
    box-shadow:0 4px 12px rgba(15,76,76,0.25) !important;
  }
  .stButton > button:hover { opacity:0.88 !important; }

  div[data-testid="stTextInput"] input {
    border-radius:10px !important;
    border:2px solid #6ee7b7 !important;
    background:#f0fdf9 !important;
    font-size:1rem !important; color:#0f172a !important;
    padding:10px 14px !important;
  }
  div[data-testid="stTextInput"] input:focus {
    border-color:#0f4c4c !important; background:#ffffff !important;
  }

  footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load chatbots
# ---------------------------------------------------------------------------
FAQ_PATH = Path("data/faq.json")

@st.cache_resource
def load_bots():
    tfidf = FAQChatbot(faq_path=FAQ_PATH, threshold=0.15)
    embed = EmbeddingFAQChatbot(faq_path=FAQ_PATH, threshold=0.40)
    return tfidf, embed

tfidf_bot, embed_bot = load_bots()

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <img class="hero-icon"
       src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png"
       alt="chatbot icon"/>
  <h1>FAQ Chatbot</h1>
  <p>NLP Assignment &nbsp;&middot;&nbsp; TF-IDF vs Word Embeddings &nbsp;&middot;&nbsp; Side-by-side comparison</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tip box
# ---------------------------------------------------------------------------
st.markdown("""
<div class="tip-box">
  <img src="https://cdn-icons-png.flaticon.com/512/1828/1828884.png" alt="tip"/>
  <span>
    <strong>Try these questions:</strong>&nbsp;
    "Can I get a refund?" &nbsp;&middot;&nbsp;
    "Where is my package?" &nbsp;&middot;&nbsp;
    "Any promo codes?" &nbsp;&middot;&nbsp;
    "Is my data safe?" &nbsp;&middot;&nbsp;
    "How fast is delivery?"
  </span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Input row
# ---------------------------------------------------------------------------
col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        label="question",
        placeholder="Type your question here...",
        label_visibility="collapsed",
    )
with col_btn:
    ask_clicked = st.button("Ask")

# ---------------------------------------------------------------------------
# Score bar helper
# ---------------------------------------------------------------------------
def score_bar(score: float) -> str:
    pct = int(score * 100)
    return f"""
    <div class="score-row">
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct}%"></div>
      </div>
      <span class="score-label">{score:.3f}</span>
    </div>"""

# card icons from Flaticon (free / open CDN)
ICON_V1 = "https://cdn-icons-png.flaticon.com/512/2920/2920349.png"   # document/text
ICON_V2 = "https://cdn-icons-png.flaticon.com/512/8637/8637101.png"   # neural network

# ---------------------------------------------------------------------------
# Answer cards
# ---------------------------------------------------------------------------
if ask_clicked and question.strip():
    q = question.strip()
    tfidf_answer, tfidf_score = tfidf_bot.get_best_match(q)
    embed_answer, embed_score = embed_bot.get_best_match(q)

    st.session_state.history.insert(0, {
        "q": q,
        "tfidf_answer": tfidf_answer, "tfidf_score": tfidf_score,
        "embed_answer": embed_answer, "embed_score": embed_score,
    })

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">
            <img src="{ICON_V1}" alt="tfidf"/>
            <span class="card-label">TF-IDF <span class="badge">V1</span></span>
          </div>
          <div class="answer-text">{tfidf_answer}</div>
          <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:4px">Confidence score</div>
          {score_bar(tfidf_score)}
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">
            <img src="{ICON_V2}" alt="embeddings"/>
            <span class="card-label">Word Embeddings <span class="badge">V2</span></span>
          </div>
          <div class="answer-text">{embed_answer}</div>
          <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:4px">Confidence score</div>
          {score_bar(embed_score)}
        </div>""", unsafe_allow_html=True)

elif ask_clicked:
    st.warning("Please type a question first.")

else:
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">
            <img src="{ICON_V1}" alt="tfidf"/>
            <span class="card-label">TF-IDF <span class="badge">V1</span></span>
          </div>
          <div class="answer-text placeholder">Answer will appear here after you ask a question...</div>
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">
            <img src="{ICON_V2}" alt="embeddings"/>
            <span class="card-label">Word Embeddings <span class="badge">V2</span></span>
          </div>
          <div class="answer-text placeholder">Answer will appear here after you ask a question...</div>
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Metrics strip — NO crossed-out numbers, just new value + delta
# ---------------------------------------------------------------------------
st.markdown("""
<div class="section-tag">
  <hr/>
  <span>
    <img src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png" alt="chart"/>
    Before / After Metrics
  </span>
  <hr/>
</div>
""", unsafe_allow_html=True)

metrics = [
    ("https://cdn-icons-png.flaticon.com/512/190/190411.png", "Accuracy",  "72.2%", "+5.6% vs TF-IDF"),
    ("https://cdn-icons-png.flaticon.com/512/3163/3163478.png","Recall",    "80.0%", "+6.7% vs TF-IDF"),
    ("https://cdn-icons-png.flaticon.com/512/992/992651.png",  "F1 Score",  "82.8%", "+4.2% vs TF-IDF"),
    ("https://cdn-icons-png.flaticon.com/512/1041/1041916.png","Avg Sim",   "0.809", "+22.7% vs TF-IDF"),
]

cols = st.columns(4)
for col, (icon, label, value, delta) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">
            <img src="{icon}" alt="{label}"/>
            {label}
          </div>
          <div class="metric-value">{value}</div>
          <div class="metric-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("""
    <div class="section-tag">
      <hr/>
      <span>
        <img src="https://cdn-icons-png.flaticon.com/512/2956/2956785.png" alt="history"/>
        Previous Questions
      </span>
      <hr/>
    </div>
    """, unsafe_allow_html=True)

    for item in st.session_state.history:
        short_t = item["tfidf_answer"][:90] + ("…" if len(item["tfidf_answer"]) > 90 else "")
        short_e = item["embed_answer"][:90] + ("…" if len(item["embed_answer"]) > 90 else "")
        st.markdown(f"""
        <div class="hist-item">
          <strong>{item['q']}</strong><br><br>
          <span class="hist-chip chip-v1">V1 &middot; {item['tfidf_score']:.3f}</span> {short_t}<br><br>
          <span class="hist-chip chip-v2">V2 &middot; {item['embed_score']:.3f}</span> {short_e}
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="text-align:center;margin-top:40px;padding-top:20px;
            border-top:1.5px solid #e2e8f0;font-size:0.78rem;color:#94a3b8">
  NLP Assignment &nbsp;&middot;&nbsp; FAQ Chatbot &nbsp;&middot;&nbsp;
  TF-IDF vs PMI-SVD Word Embeddings
</div>
""", unsafe_allow_html=True)
