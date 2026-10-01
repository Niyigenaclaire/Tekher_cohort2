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
    page_icon="🤖",
    layout="centered",
)

# ---------------------------------------------------------------------------
# SVG icons — defined FIRST so they can be used anywhere below
# ---------------------------------------------------------------------------
SVG_ROBOT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
  <rect width="64" height="64" rx="16" fill="rgba(255,255,255,0.15)"/>
  <rect x="16" y="20" width="32" height="24" rx="6" fill="white"/>
  <circle cx="24" cy="30" r="4" fill="#0f4c4c"/>
  <circle cx="40" cy="30" r="4" fill="#0f4c4c"/>
  <rect x="24" y="37" width="16" height="3" rx="1.5" fill="#0f4c4c"/>
  <rect x="30" y="12" width="4" height="8" rx="2" fill="white"/>
  <circle cx="32" cy="11" r="3" fill="white"/>
  <rect x="8" y="26" width="6" height="10" rx="3" fill="white"/>
  <rect x="50" y="26" width="6" height="10" rx="3" fill="white"/>
</svg>"""

SVG_TIP = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" width="20" height="20" fill="#059669">
  <path d="M10 2a8 8 0 100 16A8 8 0 0010 2zm1 11H9v-4h2v4zm0-6H9V5h2v2z"/>
</svg>"""

SVG_DOC = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28">
  <rect width="28" height="28" rx="8" fill="#e6f7f5"/>
  <rect x="7" y="7" width="14" height="3" rx="1.5" fill="#0f4c4c"/>
  <rect x="7" y="13" width="14" height="2" rx="1" fill="#0f4c4c" opacity=".6"/>
  <rect x="7" y="18" width="9" height="2" rx="1" fill="#0f4c4c" opacity=".4"/>
</svg>"""

SVG_NEURAL = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="28" height="28">
  <rect width="28" height="28" rx="8" fill="#e6f7f5"/>
  <circle cx="7" cy="10" r="2.5" fill="#0f4c4c"/>
  <circle cx="7" cy="18" r="2.5" fill="#0f4c4c"/>
  <circle cx="14" cy="7"  r="2.5" fill="#22a899"/>
  <circle cx="14" cy="14" r="2.5" fill="#22a899"/>
  <circle cx="14" cy="21" r="2.5" fill="#22a899"/>
  <circle cx="21" cy="10" r="2.5" fill="#0f4c4c"/>
  <circle cx="21" cy="18" r="2.5" fill="#0f4c4c"/>
  <line x1="9.5" y1="10" x2="11.5" y2="9"  stroke="#0f4c4c" stroke-width="1"/>
  <line x1="9.5" y1="10" x2="11.5" y2="14" stroke="#0f4c4c" stroke-width="1"/>
  <line x1="9.5" y1="18" x2="11.5" y2="14" stroke="#0f4c4c" stroke-width="1"/>
  <line x1="9.5" y1="18" x2="11.5" y2="21" stroke="#0f4c4c" stroke-width="1"/>
  <line x1="16.5" y1="7"  x2="18.5" y2="10" stroke="#0f4c4c" stroke-width="1"/>
  <line x1="16.5" y1="14" x2="18.5" y2="10" stroke="#0f4c4c" stroke-width="1"/>
  <line x1="16.5" y1="14" x2="18.5" y2="18" stroke="#0f4c4c" stroke-width="1"/>
  <line x1="16.5" y1="21" x2="18.5" y2="18" stroke="#0f4c4c" stroke-width="1"/>
</svg>"""

SVG_CHART = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14" fill="#64748b">
  <rect x="1" y="7" width="3" height="6" rx="1"/>
  <rect x="5.5" y="4" width="3" height="9" rx="1"/>
  <rect x="10" y="1" width="3" height="12" rx="1"/>
</svg>"""

SVG_HISTORY = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <circle cx="7" cy="7" r="6" stroke="#64748b" stroke-width="1.5" fill="none"/>
  <polyline points="7,4 7,7 9.5,9.5" stroke="#64748b" stroke-width="1.5" fill="none" stroke-linecap="round"/>
</svg>"""

SVG_ACCURACY = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <circle cx="7" cy="7" r="6" stroke="#059669" stroke-width="1.5" fill="none"/>
  <polyline points="4,7 6,9.5 10,4.5" stroke="#059669" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

SVG_RECALL = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14">
  <circle cx="7" cy="7" r="5" stroke="#0f4c4c" stroke-width="1.3" fill="none" opacity=".4"/>
  <polyline points="7,4.5 7,7 9,8.5" stroke="#0f4c4c" stroke-width="1.3" fill="none" stroke-linecap="round"/>
</svg>"""

SVG_F1 = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14" fill="#0f4c4c">
  <rect x="2" y="5" width="4" height="7" rx="1"/>
  <rect x="8" y="2" width="4" height="10" rx="1"/>
</svg>"""

SVG_SIM = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14" width="14" height="14" stroke="#0f4c4c" fill="none">
  <circle cx="4" cy="7" r="2.5" stroke-width="1.3"/>
  <circle cx="10" cy="7" r="2.5" stroke-width="1.3"/>
  <line x1="6.5" y1="7" x2="7.5" y2="7" stroke-width="1.5" stroke-linecap="round"/>
</svg>"""

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #f7f8fa;
  }

  .hero {
    background: linear-gradient(135deg, #0f4c4c 0%, #1a7a6e 60%, #22a899 100%);
    border-radius: 20px;
    padding: 36px 28px 30px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(15,76,76,0.18);
  }
  .hero svg { margin-bottom: 14px; display: block; margin-left: auto; margin-right: auto; }
  .hero h1 {
    color: #ffffff !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.5px;
  }
  .hero p { color: #a7f3d0; font-size: 0.95rem; margin: 0; }

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
  .tip-box svg { flex-shrink: 0; margin-top: 2px; }

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

  .score-row { display:flex; align-items:center; gap:10px; margin-top:4px; }
  .score-bar-bg { flex:1; height:6px; background:#e2e8f0; border-radius:99px; overflow:hidden; }
  .score-bar-fill { height:100%; border-radius:99px; background: linear-gradient(90deg, #0f4c4c, #22a899); }
  .score-label { font-size:0.75rem; font-weight:700; color:#0f4c4c; white-space:nowrap; }

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
  .metric-value { font-size:1.5rem; font-weight:800; color:#0f4c4c; line-height:1; }
  .metric-delta { font-size:0.78rem; font-weight:700; color:#059669; margin-top:4px; }

  .section-tag { display:flex; align-items:center; gap:10px; margin:28px 0 14px; }
  .section-tag span {
    font-size:0.72rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#64748b; white-space:nowrap;
    display:flex; align-items:center; gap:6px;
  }
  .section-tag hr { flex:1; border:none; border-top:1.5px solid #e2e8f0; }

  .hist-item {
    background:#ffffff; border-radius:12px; padding:14px 18px;
    margin-bottom:10px; border:1.5px solid #d1fae5;
    font-size:0.85rem; color:#475569;
    box-shadow:0 1px 6px rgba(15,76,76,0.05);
  }
  .hist-item strong { color:#0f172a; font-size:0.9rem; }
  .hist-chip { display:inline-block; font-size:0.67rem; font-weight:700; padding:1px 7px; border-radius:99px; margin-right:4px; }
  .chip-v1 { background:#ccfbf1; color:#0f4c4c; }
  .chip-v2 { background:#d1fae5; color:#065f46; }

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
st.markdown(f"""
<div class="hero">
  {SVG_ROBOT}
  <h1>FAQ Chatbot</h1>
  <p>NLP Assignment &nbsp;&middot;&nbsp; TF-IDF vs Word Embeddings &nbsp;&middot;&nbsp; Side-by-side comparison</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tip box
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="tip-box">
  {SVG_TIP}
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
# Input
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
    return f"""<div class="score-row">
      <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>
      <span class="score-label">{score:.3f}</span>
    </div>"""

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
          <div class="card-header">{SVG_DOC}
            <span class="card-label">TF-IDF <span class="badge">V1</span></span>
          </div>
          <div class="answer-text">{tfidf_answer}</div>
          <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:4px">Confidence score</div>
          {score_bar(tfidf_score)}
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">{SVG_NEURAL}
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
          <div class="card-header">{SVG_DOC}
            <span class="card-label">TF-IDF <span class="badge">V1</span></span>
          </div>
          <div class="answer-text placeholder">Answer will appear here after you ask a question...</div>
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">{SVG_NEURAL}
            <span class="card-label">Word Embeddings <span class="badge">V2</span></span>
          </div>
          <div class="answer-text placeholder">Answer will appear here after you ask a question...</div>
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="section-tag"><hr/>
  <span>{SVG_CHART} Before / After Metrics</span>
<hr/></div>
""", unsafe_allow_html=True)

metrics = [
    (SVG_ACCURACY, "Accuracy",  "72.2%", "+5.6% vs TF-IDF"),
    (SVG_RECALL,   "Recall",    "80.0%", "+6.7% vs TF-IDF"),
    (SVG_F1,       "F1 Score",  "82.8%", "+4.2% vs TF-IDF"),
    (SVG_SIM,      "Avg Sim",   "0.809", "+22.7% vs TF-IDF"),
]
cols = st.columns(4)
for col, (icon_svg, label, value, delta) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{icon_svg} {label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown(f"""
    <div class="section-tag"><hr/>
      <span>{SVG_HISTORY} Previous Questions</span>
    <hr/></div>""", unsafe_allow_html=True)

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
  NLP Assignment &nbsp;&middot;&nbsp; FAQ Chatbot &nbsp;&middot;&nbsp; TF-IDF vs PMI-SVD Word Embeddings
</div>
""", unsafe_allow_html=True)
