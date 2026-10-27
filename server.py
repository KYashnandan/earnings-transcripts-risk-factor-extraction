"""Serve the website and connect it to the Python code.

    python server.py   ->   http://127.0.0.1:8000

POST /api/analyse  {"text": "..."}   sentence-by-sentence keyword match + classifier guess
POST /api/extract  <csv body>        run the spaCy extraction on an uploaded transcripts CSV
"""
import io, json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import pandas as pd
from fault_lines.classify import analyse_text, load_model
from fault_lines.cli import guess_columns, to_quarter
from fault_lines.extract import build_nlp, extract

NLP, MATCHER = build_nlp()
MODEL = load_model()
MAX_BYTES = 8_000_000

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory="docs", **k)

    def reply(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/health"):
            return self.reply({"ok": True})
        super().do_GET()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BYTES:
            return self.reply({"error": "file too large"}, 413)
        body = self.rfile.read(n)
        try:
            if self.path == "/api/analyse":
                text = json.loads(body)["text"][:20000]
                return self.reply(analyse_text(text, NLP, MATCHER, MODEL))
            if self.path == "/api/extract":
                df = pd.read_csv(io.BytesIO(body))
                c, q, t = guess_columns(df)
                df[q] = to_quarter(df[q])
                return self.reply(extract(df, c, q, t))
            self.reply({"error": "not found"}, 404)
        except Exception as e:
            self.reply({"error": str(e)[:200]}, 400)

if __name__ == "__main__":
    print("Risk factor extraction running at http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
