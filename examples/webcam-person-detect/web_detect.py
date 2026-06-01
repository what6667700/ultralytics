# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Web UI for real-time person detection (browser view, no local GUI required)."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2

from ultralytics import YOLO

from main import (
    DEFAULT_STREAM_URL,
    PERSON_CLASS_ID,
    REMOTE_LAPTOP_HOST,
    REMOTE_STREAM_PATH,
    REMOTE_STREAM_PORT,
    open_capture,
    resolve_source,
)

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>人体识别 - 实时预览</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; background: #1a1a2e; color: #eee; }
    header { padding: 12px 20px; background: #16213e; border-bottom: 1px solid #0f3460; }
    h1 { margin: 0; font-size: 1.25rem; }
    main { padding: 16px; max-width: 1280px; margin: 0 auto; }
    .stats { margin: 12px 0; font-size: 1.1rem; }
    .stats span { color: #4ecca3; font-weight: bold; font-size: 1.5rem; }
    img { width: 100%; max-width: 1280px; border-radius: 8px; background: #000; }
    .meta { color: #888; font-size: 0.85rem; margin-top: 8px; }
  </style>
</head>
<body>
  <header><h1>人体识别 · 实时预览</h1></header>
  <main>
    <p class="stats">当前人数：<span id="count">-</span></p>
    <img src="/stream" alt="detection stream"/>
    <p class="meta">视频源：__SOURCE__ · 按 F5 刷新页面</p>
  </main>
  <script>
    async function refreshCount() {
      try {
        const r = await fetch('/api/stats');
        const d = await r.json();
        document.getElementById('count').textContent = d.persons;
      } catch (e) {}
    }
    setInterval(refreshCount, 500);
    refreshCount();
  </script>
</body>
</html>
"""


class SharedState:
    """Thread-safe latest annotated frame and stats."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: cv2.typing.MatLike | None = None
        self.persons = 0
        self.fps = 0.0
        self.ready = threading.Event()

    def update(self, frame: cv2.typing.MatLike, persons: int, fps: float) -> None:
        with self.lock:
            self.frame = frame
            self.persons = persons
            self.fps = fps
        self.ready.set()

    def get_frame(self) -> cv2.typing.MatLike | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def get_stats(self) -> dict:
        with self.lock:
            return {"persons": self.persons, "fps": round(self.fps, 1)}


state = SharedState()


def detection_loop(
    source: str | int,
    model: YOLO,
    conf: float,
    device: str,
) -> None:
    """Read camera stream, run YOLO, push annotated frames to shared state."""
    cap = open_capture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source={source}")

    predict_kwargs: dict = {"conf": conf, "classes": [PERSON_CLASS_ID], "verbose": False}
    if device:
        predict_kwargs["device"] = device

    frames = 0
    t0 = time.perf_counter()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        results = model(frame, **predict_kwargs)
        annotated = results[0].plot()
        persons = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(
            annotated,
            f"persons: {persons}",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        frames += 1
        elapsed = time.perf_counter() - t0
        fps = frames / elapsed if elapsed > 0 else 0.0
        state.update(annotated, persons, fps)

    cap.release()


class WebHandler(BaseHTTPRequestHandler):
    source_label: str = DEFAULT_STREAM_URL

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_html()
        elif self.path == "/stream":
            self._send_mjpeg()
        elif self.path == "/api/stats":
            self._send_json(state.get_stats())
        else:
            self.send_error(404)

    def _send_html(self) -> None:
        body = INDEX_HTML.replace("__SOURCE__", self.source_label).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while True:
            state.ready.wait(timeout=30)
            frame = state.get_frame()
            if frame is None:
                continue
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            try:
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(buf.tobytes())
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(0.03)

    def log_message(self, format: str, *args) -> None:
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web UI for person detection on a video stream.")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="YOLO weights")
    parser.add_argument("--source", type=str, default="", help="Input video stream URL")
    parser.add_argument("--stream-host", type=str, default=REMOTE_LAPTOP_HOST, help="Remote laptop IP")
    parser.add_argument("--stream-port", type=int, default=REMOTE_STREAM_PORT, help="Remote stream port")
    parser.add_argument("--stream-path", type=str, default=REMOTE_STREAM_PATH, help="Remote stream path")
    parser.add_argument("--host", type=str, default="", help="Web server bind address ('' = all interfaces)")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    parser.add_argument("--local", action="store_true", help="Use local camera index 0")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="", help="cuda device id or 'cpu'")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source = args.source or ""

    bind_host = args.host
    bind_port = args.port
    stream_cfg = argparse.Namespace(
        local=args.local,
        source=args.source,
        host=args.stream_host,
        port=args.stream_port,
        path=args.stream_path,
    )
    stream_source = resolve_source(stream_cfg)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"Input stream: {stream_source}")
    worker = threading.Thread(
        target=detection_loop,
        args=(stream_source, model, args.conf, args.device),
        daemon=True,
    )
    worker.start()
    time.sleep(1.0)

    WebHandler.source_label = str(stream_source)
    server = ThreadedHTTPServer((bind_host, bind_port), WebHandler)
    print(f"Web UI: http://<server-ip>:{bind_port}/")
    print(f"  Annotated stream: http://<server-ip>:{bind_port}/stream")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
