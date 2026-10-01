"""
Flask web UI for the FAQ Chatbot.
Runs both TF-IDF and Embedding chatbots side by side.

Start with:
    .venv\Scripts\python app.py
Then open: http://127.0.0.1:5000
"""

from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

from chatbot import FAQChatbot
from chatbot_embeddings import EmbeddingFAQChatbot

app = Flask(__name__)

FAQ_PATH = Path("data/faq.json")

print("Loading TF-IDF chatbot...")
tfidf_bot = FAQChatbot(faq_path=FAQ_PATH, threshold=0.15)

print("Loading Embedding chatbot...")
embed_bot = EmbeddingFAQChatbot(faq_path=FAQ_PATH, threshold=0.40)

print("Both chatbots ready.\n")

# ---------------------------------------------------------------------------
# HTML page (single file, no templates folder needed)
# ---------------------------------------------------------------------------

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>FAQ Chatbot — TF-IDF vs Embeddings</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', sans-serif;
      background: #ffffff;
      color: #0f1f3d;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 40px 16px;
    }

    h1 {
      font-size: 2rem;
      font-weight: 800;
      color: #0f1f3d;
      margin-bottom: 6px;
      text-align: center;
      letter-spacing: -0.5px;
    }

    .subtitle {
      font-size: 0.92rem;
      color: #4a5e7a;
      margin-bottom: 32px;
      text-align: center;
    }

    .input-row {
      display: flex;
      gap: 10px;
      width: 100%;
      max-width: 740px;
      margin-bottom: 28px;
    }

    input[type="text"] {
      flex: 1;
      padding: 13px 18px;
      border-radius: 10px;
      border: 2px solid #c5d0e0;
      background: #f4f7fb;
      color: #0f1f3d;
      font-size: 1rem;
      outline: none;
      transition: border 0.2s;
    }

    input[type="text"]:focus { border-color: #1a3f7a; background: #ffffff; }

    button {
      padding: 13px 28px;
      border-radius: 10px;
      border: none;
      background: #1a3f7a;
      color: #ffffff;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }

    button:hover { background: #0f2a57; }
    button:disabled { background: #8fa3c0; cursor: default; }

    .cards {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      width: 100%;
      max-width: 740px;
    }

    @media (max-width: 580px) {
      .cards { grid-template-columns: 1fr; }
    }

    .card {
      background: #f4f7fb;
      border-radius: 14px;
      padding: 22px;
      border: 2px solid #dce6f0;
      min-height: 170px;
      box-shadow: 0 2px 10px rgba(15,31,61,0.07);
    }

    .card-title {
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      gap: 8px;
      color: #1a3f7a;
    }

    .badge {
      font-size: 0.7rem;
      padding: 3px 9px;
      border-radius: 99px;
      font-weight: 700;
    }

    .badge-tfidf  { background: #1a3f7a; color: #ffffff; }
    .badge-embed  { background: #0f2a57; color: #ffffff; }

    .answer {
      font-size: 1rem;
      line-height: 1.6;
      color: #0f1f3d;
      font-weight: 500;
    }

    .score {
      margin-top: 14px;
      font-size: 0.8rem;
      color: #4a5e7a;
    }

    .score span {
      font-weight: 700;
      color: #1a3f7a;
    }

    .history {
      width: 100%;
      max-width: 740px;
      margin-top: 36px;
    }

    .history h2 {
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #4a5e7a;
      margin-bottom: 12px;
      font-weight: 700;
    }

    .history-item {
      background: #f4f7fb;
      border-radius: 10px;
      padding: 13px 18px;
      margin-bottom: 10px;
      border: 1px solid #dce6f0;
      font-size: 0.87rem;
      color: #4a5e7a;
    }

    .history-item strong { color: #0f1f3d; font-size: 0.95rem; }

    .placeholder {
      color: #8fa3c0;
      font-style: italic;
      font-size: 0.9rem;
    }

    .spinner { display: none; margin-left: 8px; }
  </style>
</head>
<body>

  <h1>FAQ Chatbot</h1>
  <p class="subtitle">Type a question — see TF-IDF vs Word Embeddings side by side</p>

  <div class="input-row">
    <input type="text" id="question" placeholder="e.g. Can I get a refund?" autofocus />
    <button id="askBtn" onclick="ask()">Ask</button>
  </div>

  <div class="cards">
    <div class="card">
      <div class="card-title">
        <span class="badge badge-tfidf">v1</span> TF-IDF
      </div>
      <div class="answer" id="tfidf-answer">
        <span class="placeholder">Answer will appear here...</span>
      </div>
      <div class="score" id="tfidf-score"></div>
    </div>

    <div class="card">
      <div class="card-title">
        <span class="badge badge-embed">v2</span> Word Embeddings
      </div>
      <div class="answer" id="embed-answer">
        <span class="placeholder">Answer will appear here...</span>
      </div>
      <div class="score" id="embed-score"></div>
    </div>
  </div>

  <div class="history">
    <h2>Previous questions</h2>
    <div id="history-list"><p class="placeholder" style="font-size:0.82rem">None yet.</p></div>
  </div>

  <script>
    const input   = document.getElementById('question');
    const btn     = document.getElementById('askBtn');
    const histDiv = document.getElementById('history-list');

    input.addEventListener('keydown', e => { if (e.key === 'Enter') ask(); });

    async function ask() {
      const q = input.value.trim();
      if (!q) return;

      btn.disabled = true;
      btn.textContent = 'Asking…';

      document.getElementById('tfidf-answer').innerHTML = '<span class="placeholder">Loading…</span>';
      document.getElementById('embed-answer').innerHTML  = '<span class="placeholder">Loading…</span>';
      document.getElementById('tfidf-score').textContent = '';
      document.getElementById('embed-score').textContent  = '';

      try {
        const res  = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q })
        });
        const data = await res.json();

        document.getElementById('tfidf-answer').textContent = data.tfidf.answer;
        document.getElementById('tfidf-score').innerHTML =
          `Similarity score: <span>${data.tfidf.score.toFixed(3)}</span>`;

        document.getElementById('embed-answer').textContent = data.embed.answer;
        document.getElementById('embed-score').innerHTML =
          `Similarity score: <span>${data.embed.score.toFixed(3)}</span>`;

        // add to history
        if (histDiv.querySelector('.placeholder')) histDiv.innerHTML = '';
        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `<strong>${escHtml(q)}</strong><br>
          TF-IDF (${data.tfidf.score.toFixed(3)}): ${escHtml(data.tfidf.answer.slice(0,80))}${data.tfidf.answer.length>80?'…':''}<br>
          Embeddings (${data.embed.score.toFixed(3)}): ${escHtml(data.embed.answer.slice(0,80))}${data.embed.answer.length>80?'…':''}`;
        histDiv.prepend(item);

        input.value = '';
        input.focus();
      } catch (err) {
        document.getElementById('tfidf-answer').textContent = 'Error contacting server.';
        document.getElementById('embed-answer').textContent  = 'Error contacting server.';
      }

      btn.disabled = false;
      btn.textContent = 'Ask';
    }

    function escHtml(s) {
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = str(data.get("question", "")).strip()

    tfidf_answer, tfidf_score = tfidf_bot.get_best_match(question)
    embed_answer, embed_score = embed_bot.get_best_match(question)

    return jsonify({
        "tfidf": {"answer": tfidf_answer, "score": tfidf_score},
        "embed": {"answer": embed_answer, "score": embed_score},
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
