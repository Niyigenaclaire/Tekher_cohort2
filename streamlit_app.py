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
    page_icon="https://img.icons8.com/fluency/48/chatbot.png",
    layout="centered",
)

# ---------------------------------------------------------------------------
# SVG icons — all inline, no external dependencies
# ---------------------------------------------------------------------------
SVG_BOT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80" width="80" height="80">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>
  </defs>
  <rect width="80" height="80" rx="20" fill="url(#bg)" opacity="0.9"/>
  <rect x="20" y="25" width="40" height="30" rx="8" fill="white" opacity="0.95"/>
  <circle cx="31" cy="38" r="5" fill="#7c3aed"/>
  <circle cx="49" cy="38" r="5" fill="#7c3aed"/>
  <circle cx="31" cy="38" r="2" fill="white"/>
  <circle cx="49" cy="38" r="2" fill="white"/>
  <rect x="29" y="46" width="22" height="4" rx="2" fill="#7c3aed" opacity="0.7"/>
  <rect x="37" y="14" width="6" height="11" rx="3" fill="white" opacity="0.8"/>
  <circle cx="40" cy="13" r="4" fill="white" opacity="0.8"/>
  <rect x="10" y="32" width="8" height="14" rx="4" fill="white" opacity="0.7"/>
  <rect x="62" y="32" width="8" height="14" rx="4" fill="white" opacity="0.7"/>
</svg>"""

SVG_DOC = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
  <defs><linearGradient id="d1" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#6366f1"/><stop offset="100%" stop-color="#8b5cf6"/>
  </linearGradient></defs>
  <rect width="32" height="32" rx="10" fill="url(#d1)"/>
  <rect x="8" y="9" width="16" height="3" rx="1.5" fill="white"/>
  <rect x="8" y="15" width="16" height="2.5" rx="1.2" fill="white" opacity=".7"/>
  <rect x="8" y="21" width="10" height="2.5" rx="1.2" fill="white" opacity=".5"/>
</svg>"""

SVG_NEURAL = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" height="32">
  <defs><linearGradient id="n1" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#a855f7"/><stop offset="100%" stop-color="#ec4899"/>
  </linearGradient></defs>
  <rect width="32" height="32" rx="10" fill="url(#n1)"/>
  <circle cx="8" cy="11" r="3" fill="white"/>
  <circle cx="8" cy="21" r="3" fill="white"/>
  <circle cx="16" cy="8"  r="3" fill="white" opacity=".8"/>
  <circle cx="16" cy="16" r="3" fill="white" opacity=".8"/>
  <circle cx="16" cy="24" r="3" fill="white" opacity=".8"/>
  <circle cx="24" cy="11" r="3" fill="white"/>
  <circle cx="24" cy="21" r="3" fill="white"/>
  <line x1="11" y1="11" x2="13" y2="10" stroke="white" stroke-width="1.2" opacity=".6"/>
  <line x1="11" y1="11" x2="13" y2="16" stroke="white" stroke-width="1.2" opacity=".6"/>
  <line x1="11" y1="21" x2="13" y2="16" stroke="white" stroke-width="1.2" opacity=".6"/>
  <line x1="11" y1="21" x2="13" y2="24" stroke="white" stroke-width="1.2" opacity=".6"/>
  <line x1="19" y1="9"  x2="21" y2="11" stroke="white" stroke-width="1.2" opacity=".6"/>
  <line x1="19" y1="16" x2="21" y2="11" stroke="white" stroke-width="1.2" opacity=".6"/>
  <line x1="19" y1="16" x2="21" y2="21" stroke="white" stroke-width="1.2" opacity=".6"/>
  <line x1="19" y1="24" x2="21" y2="21" stroke="white" stroke-width="1.2" opacity=".6"/>
</svg>"""

SVG_CHART = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
  <rect x="1" y="8" width="3.5" height="7" rx="1" fill="#a855f7"/>
  <rect x="6" y="5" width="3.5" height="10" rx="1" fill="#8b5cf6"/>
  <rect x="11" y="2" width="3.5" height="13" rx="1" fill="#7c3aed"/>
</svg>"""

SVG_CLOCK = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
  <circle cx="8" cy="8" r="7" stroke="#a855f7" stroke-width="1.5" fill="none"/>
  <polyline points="8,4.5 8,8 10.5,10" stroke="#a855f7" stroke-width="1.5" fill="none" stroke-linecap="round"/>
</svg>"""

SVG_CHECK = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
  <circle cx="8" cy="8" r="7" stroke="#22d3ee" stroke-width="1.5" fill="none"/>
  <polyline points="4.5,8 7,10.5 11.5,5" stroke="#22d3ee" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

SVG_BAR = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16">
  <rect x="2" y="6" width="4.5" height="9" rx="1" fill="#22d3ee"/>
  <rect x="9" y="2" width="4.5" height="13" rx="1" fill="#a855f7"/>
</svg>"""

SVG_LINK = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="16" height="16" stroke="#22d3ee" fill="none">
  <circle cx="4.5" cy="8" r="3" stroke-width="1.4"/>
  <circle cx="11.5" cy="8" r="3" stroke-width="1.4"/>
  <line x1="7.5" y1="8" x2="8.5" y2="8" stroke-width="2" stroke-linecap="round"/>
</svg>"""

# ---------------------------------------------------------------------------
# CSS — dark purple/violet futuristic theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #0d0d1a;
  }

  /* ── Hero ── */
  .hero {
    background: linear-gradient(135deg, #1a0533 0%, #2d1060 50%, #1a0533 100%);
    border: 1px solid rgba(168,85,247,0.3);
    border-radius: 24px;
    padding: 40px 28px 32px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 0 60px rgba(168,85,247,0.15), 0 8px 32px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(168,85,247,0.08) 0%, transparent 60%);
    pointer-events: none;
  }
  .hero svg { display: block; margin: 0 auto 16px; }
  .hero h1 {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.5px;
    text-shadow: 0 0 30px rgba(168,85,247,0.5);
  }
  .hero p { color: #c4b5fd; font-size: 0.95rem; margin: 0; }

  /* ── Tip box ── */
  .tip-box {
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 14px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #c4b5fd;
    line-height: 1.6;
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }

  /* ── Cards ── */
  .card {
    background: linear-gradient(145deg, #1a1030 0%, #130d25 100%);
    border-radius: 18px;
    padding: 22px 22px 18px;
    min-height: 220px;
    border: 1px solid rgba(168,85,247,0.25);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), 0 0 20px rgba(168,85,247,0.05);
  }
  .card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }
  .card-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #e2d9f3;
  }
  .badge {
    font-size: 0.62rem;
    font-weight: 800;
    padding: 2px 8px;
    border-radius: 99px;
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: #fff;
    margin-left: 6px;
    letter-spacing: 0.05em;
  }
  .answer-text {
    font-size: 0.96rem;
    color: #e2d9f3;
    line-height: 1.7;
    font-weight: 400;
    margin-bottom: 16px;
  }
  .answer-text.placeholder { color: #4a3f6b; font-style: italic; }

  /* score bar */
  .score-row { display:flex; align-items:center; gap:10px; margin-top:4px; }
  .score-bar-bg { flex:1; height:5px; background:rgba(255,255,255,0.08); border-radius:99px; overflow:hidden; }
  .score-bar-fill {
    height:100%; border-radius:99px;
    background: linear-gradient(90deg, #7c3aed, #a855f7, #ec4899);
    box-shadow: 0 0 8px rgba(168,85,247,0.5);
  }
  .score-label { font-size:0.75rem; font-weight:700; color:#a78bfa; white-space:nowrap; }

  /* ── Metric cards ── */
  .metric-card {
    background: linear-gradient(145deg, #1a1030, #130d25);
    border: 1px solid rgba(168,85,247,0.2);
    border-radius: 16px;
    padding: 18px 12px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
  }
  .metric-label {
    font-size: 0.65rem; font-weight:700;
    text-transform:uppercase; letter-spacing:0.1em;
    color:#6d5e9e; margin-bottom:8px;
    display:flex; align-items:center; justify-content:center; gap:6px;
  }
  .metric-value { font-size:1.6rem; font-weight:800; color:#e2d9f3; line-height:1; }
  .metric-delta { font-size:0.75rem; font-weight:700; color:#34d399; margin-top:6px; }

  /* ── Section divider ── */
  .section-tag { display:flex; align-items:center; gap:10px; margin:32px 0 16px; }
  .section-tag span {
    font-size:0.7rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.12em; color:#6d5e9e; white-space:nowrap;
    display:flex; align-items:center; gap:6px;
  }
  .section-tag hr { flex:1; border:none; border-top:1px solid rgba(168,85,247,0.15); }

  /* ── History ── */
  .hist-item {
    background: linear-gradient(145deg, #1a1030, #130d25);
    border-radius: 14px; padding:14px 18px;
    margin-bottom:10px; border:1px solid rgba(168,85,247,0.2);
    font-size:0.85rem; color:#9880c8;
  }
  .hist-item strong { color:#e2d9f3; font-size:0.92rem; }
  .hist-chip {
    display:inline-block; font-size:0.66rem; font-weight:700;
    padding:2px 8px; border-radius:99px; margin-right:5px;
  }
  .chip-v1 { background:rgba(99,102,241,0.2); color:#a5b4fc; border:1px solid rgba(99,102,241,0.3); }
  .chip-v2 { background:rgba(168,85,247,0.2); color:#d8b4fe; border:1px solid rgba(168,85,247,0.3); }

  /* ── Streamlit overrides ── */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed, #a855f7) !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    width: 100%; border: none !important;
    padding: 0.65rem 1rem !important;
    box-shadow: 0 4px 16px rgba(124,58,237,0.4) !important;
    letter-spacing: 0.02em !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  div[data-testid="stTextInput"] input {
    border-radius: 12px !important;
    border: 1.5px solid rgba(124,58,237,0.4) !important;
    background: rgba(255,255,255,0.04) !important;
    font-size: 1rem !important;
    color: #e2d9f3 !important;
    padding: 11px 16px !important;
  }
  div[data-testid="stTextInput"] input:focus {
    border-color: #a855f7 !important;
    background: rgba(168,85,247,0.06) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.15) !important;
  }
  div[data-testid="stTextInput"] input::placeholder { color: #4a3f6b !important; }

  footer { visibility: hidden; }
  .stDeployButton { display: none; }
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
  {SVG_BOT}
  <h1>FAQ Chatbot</h1>
  <p>NLP Assignment &nbsp;&middot;&nbsp; TF-IDF vs Word Embeddings &nbsp;&middot;&nbsp; Side-by-side comparison</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tip box  — using native st.info so it always renders correctly
# ---------------------------------------------------------------------------
st.info(
    '**Try these questions:** '
    '"Can I get a refund?" · '
    '"Where is my package?" · '
    '"Any promo codes?" · '
    '"Is my data safe?" · '
    '"How fast is delivery?"'
)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        label="question",
        placeholder="Ask a question...",
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
          <div style="font-size:0.7rem;color:#4a3f6b;margin-bottom:4px">Confidence</div>
          {score_bar(tfidf_score)}
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown(f"""
        <div class="card">
          <div class="card-header">{SVG_NEURAL}
            <span class="card-label">Word Embeddings <span class="badge">V2</span></span>
          </div>
          <div class="answer-text">{embed_answer}</div>
          <div style="font-size:0.7rem;color:#4a3f6b;margin-bottom:4px">Confidence</div>
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
  <span>{SVG_CHART}&nbsp; Before / After Metrics</span>
<hr/></div>
""", unsafe_allow_html=True)

metrics = [
    (SVG_CHECK, "Accuracy",  "72.2%", "+5.6% vs TF-IDF"),
    (SVG_CLOCK, "Recall",    "80.0%", "+6.7% vs TF-IDF"),
    (SVG_BAR,   "F1 Score",  "82.8%", "+4.2% vs TF-IDF"),
    (SVG_LINK,  "Avg Sim",   "0.809", "+22.7% vs TF-IDF"),
]
cols = st.columns(4)
for col, (icon_svg, label, value, delta) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{icon_svg}&nbsp;{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown(f"""
    <div class="section-tag"><hr/>
      <span>{SVG_CLOCK}&nbsp; Previous Questions</span>
    <hr/></div>""", unsafe_allow_html=True)

    for item in st.session_state.history:
        short_t = item["tfidf_answer"][:90] + ("..." if len(item["tfidf_answer"]) > 90 else "")
        short_e = item["embed_answer"][:90] + ("..." if len(item["embed_answer"]) > 90 else "")
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
<div style="text-align:center;margin-top:48px;padding-top:20px;
            border-top:1px solid rgba(168,85,247,0.15);
            font-size:0.76rem;color:#3d3060;letter-spacing:0.05em">
  NLP ASSIGNMENT &nbsp;&middot;&nbsp; FAQ CHATBOT &nbsp;&middot;&nbsp; TF-IDF VS PMI-SVD WORD EMBEDDINGS
</div>
""", unsafe_allow_html=True)
