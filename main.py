from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import nest_asyncio
from pyngrok import ngrok
from model import generate_answer
from pdf_processor import search_relevant_text


class ChatRequest(BaseModel):
    input_text: str


app = FastAPI()


INDEX_HTML = """<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ImamGPT</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Tahoma, Arial, sans-serif;
         max-width: 760px; margin: 2rem auto; padding: 0 1rem; line-height: 1.7; }
  h1 { margin: 0 0 0.25rem; font-size: 1.6rem; }
  .sub { color: #888; font-size: 0.9rem; margin-bottom: 1.25rem; }
  textarea { width: 100%; min-height: 90px; padding: 0.75rem; font-size: 1rem;
             box-sizing: border-box; border: 1px solid #ccc; border-radius: 6px;
             font-family: inherit; }
  button { margin-top: 0.5rem; padding: 0.6rem 1.4rem; font-size: 1rem; cursor: pointer;
           border: 0; border-radius: 6px; background: #2a6df4; color: white; }
  button:disabled { background: #888; cursor: not-allowed; }
  #status { margin-top: 0.5rem; color: #888; font-size: 0.9rem; min-height: 1.2em; }
  #response { margin-top: 1.5rem; padding: 1rem; background: #f5f5f5; color: #111;
              border-radius: 6px; white-space: pre-wrap; min-height: 4rem; }
  #sources { margin-top: 0.75rem; font-size: 0.92rem; }
  #sources .label { color: #888; margin-bottom: 0.25rem; }
  #sources ul { margin: 0; padding-inline-start: 1.4rem; }
  #sources li { color: #444; }
  @media (prefers-color-scheme: dark) {
    #response { background: #1e1e1e; color: #eee; }
    textarea { background: #1e1e1e; color: #eee; border-color: #333; }
    #sources li { color: #bbb; }
  }
</style>
</head>
<body>
  <h1>ImamGPT</h1>
  <div class="sub">اكتب سؤالك بالعربية. النموذج يبحث في مراجع سنية ثم يُجيب.</div>
  <form id="f">
    <textarea id="q" placeholder="مثال: ما هي أركان الإسلام الخمسة؟" autofocus></textarea>
    <button id="btn" type="submit">إرسال</button>
    <div id="status"></div>
  </form>
  <div id="response"></div>
  <div id="sources"></div>
<script>
const f = document.getElementById('f');
const q = document.getElementById('q');
const r = document.getElementById('response');
const srcBox = document.getElementById('sources');
const s = document.getElementById('status');
const b = document.getElementById('btn');

function renderSources(sources) {
  srcBox.innerHTML = '';
  if (!sources || !sources.length) return;
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = 'المصادر:';
  srcBox.appendChild(label);
  const ul = document.createElement('ul');
  for (const src of sources) {
    const li = document.createElement('li');
    li.textContent = src;
    ul.appendChild(li);
  }
  srcBox.appendChild(ul);
}

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const txt = q.value.trim();
  if (!txt) return;
  b.disabled = true;
  s.textContent = '...جاري التحميل';
  r.textContent = '';
  srcBox.innerHTML = '';
  const t0 = performance.now();
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({input_text: txt}),
    });
    const data = await res.json();
    r.textContent = data.response || JSON.stringify(data, null, 2);
    renderSources(data.sources);
    s.textContent = '(' + Math.round((performance.now() - t0) / 1000) + 's)';
  } catch (err) {
    r.textContent = 'خطأ: ' + err.message;
    s.textContent = '';
  } finally {
    b.disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML


@app.get("/health")
def health():
    return {"message": "ImamGPT API is running!"}


@app.post("/chat")
def chat(request: ChatRequest):
    context, sources = search_relevant_text(request.input_text)
    response = generate_answer(request.input_text, context)
    return {"response": response, "sources": sources}


subdomain = "imamgpt"

# Allow FastAPI to run inside Google Colab
nest_asyncio.apply()
ngrok_tunnel = ngrok.connect(8000)

print(f"Public API URL: {ngrok_tunnel.public_url}")

# Start FastAPI Server
uvicorn.run(app, host="0.0.0.0", port=8000)
