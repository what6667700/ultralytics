# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Publish the laptop integrated webcam as an MJPEG stream (run on 192.168.51.65)."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_CAMERA = 0


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse 'WIDTHxHEIGHT' (e.g. 640x480)."""
    parts = value.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{value}'. Use WIDTHxHEIGHT, e.g. 640x480.")
    try:
        w, h = int(parts[0]), int(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid resolution '{value}'. Use WIDTHxHEIGHT, e.g. 640x480.") from e
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"Resolution must be positive, got {value}.")
    return w, h


def apply_resolution(cap: cv2.VideoCapture, width: int, height: int) -> tuple[int, int]:
    """Request capture size; return actual width/height reported by the device."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return actual_w, actual_h


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish integrated webcam over HTTP (MJPEG).")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (0.0.0.0 = LAN accessible)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA, help="Local camera index (0 = integrated)")
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        metavar="WxH",
        help="Capture size WIDTHxHEIGHT, e.g. 640x480 (camera may use nearest supported mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open local camera index={args.camera}.")

    if args.resolution:
        req_w, req_h = parse_resolution(args.resolution)
        act_w, act_h = apply_resolution(cap, req_w, req_h)
        print(f"Resolution: requested {req_w}x{req_h}, camera reports {act_w}x{act_h}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/video"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                try:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(buf.tobytes())
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadedHTTPServer((args.host, args.port), Handler)
    print(f"Integrated webcam streaming at http://<this-laptop-ip>:{args.port}/video")
    print("On another PC, run: python main.py")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        server.server_close()


if __name__ == "__main__":
    main()
