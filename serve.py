"""Minimal web demo: serves index.html and POST /predict -> label +
probabilities + L1-vs-L2 breakdown.

Run: CUDA_VISIBLE_DEVICES=2 .venv/bin/python serve.py
Then open http://127.0.0.1:8377
"""

import argparse
import json
import os
import threading
import time

import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from classifier import Classifier

HERE = os.path.dirname(os.path.abspath(__file__))
_lock = threading.Lock()  # CUDA is not thread-safe


def load_model(device):
    state = torch.load(os.path.join(HERE, "classifier.pt"), map_location=device)
    n = state["head.weight"].shape[0]
    n_slots = state["l2.memory"].shape[0]
    model = Classifier(n_classes=n, n_slots=n_slots).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def make_handler(model, device):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            else:
                self._send(404, '{"error":"not found"}')

        def do_POST(self):
            if self.path != "/predict":
                self._send(404, '{"error":"not found"}')
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                text = json.loads(self.rfile.read(n)).get("text", "")
            except Exception:
                self._send(400, '{"error":"bad request"}')
                return
            if not text.strip():
                self._send(400, '{"error":"empty text"}')
                return
            with _lock:
                t0 = time.perf_counter()
                result = model.predict_rich(text)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) * 1000
            result["ms"] = round(ms, 2)
            self._send(200, json.dumps(result))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8377)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    print(f"model loaded on {device}, serving on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), make_handler(model, device)).serve_forever()


if __name__ == "__main__":
    main()