# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Real-time person detection from a local or remote laptop integrated webcam."""

from __future__ import annotations

import argparse
import os

import cv2

from ultralytics import YOLO

# COCO dataset: class 0 is "person"
PERSON_CLASS_ID = 0

# Remote laptop with integrated webcam (run publish_webcam.py on that machine first)
REMOTE_LAPTOP_HOST = "192.168.51.65"
REMOTE_STREAM_PORT = 8080
REMOTE_STREAM_PATH = "/video"
DEFAULT_STREAM_URL = f"http://{REMOTE_LAPTOP_HOST}:{REMOTE_STREAM_PORT}{REMOTE_STREAM_PATH}"


def build_remote_url(host: str, port: int, path: str) -> str:
    path = path if path.startswith("/") else f"/{path}"
    return f"http://{host}:{port}{path}"


def parse_source(source: str, local: bool) -> str | int:
    """Resolve camera source for local integrated cam or remote laptop stream."""
    if local:
        return int(source) if source.isdigit() else 0
    if source.isdigit():
        return build_remote_url(REMOTE_LAPTOP_HOST, REMOTE_STREAM_PORT, REMOTE_STREAM_PATH)
    if "://" in source:
        return source
    if ":" in source and not source.startswith("["):
        return f"http://{source}{REMOTE_STREAM_PATH}"
    return build_remote_url(source, REMOTE_STREAM_PORT, REMOTE_STREAM_PATH)


def open_capture(source: str | int) -> cv2.VideoCapture:
    """Open local webcam or remote laptop MJPEG/RTSP stream."""
    if isinstance(source, str):
        if source.lower().startswith("rtsp"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        elif source.lower().startswith("http"):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print(f"Warning: FFMPEG backend failed for {source}, trying default backend")
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    else:
        cap = cv2.VideoCapture(source)
    return cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect people from a laptop integrated webcam (local or over LAN)."
    )
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="YOLO weights")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use this machine's integrated webcam (source=0). Run on 192.168.51.65 laptop.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Override: camera index (with --local), full URL, or remote host IP",
    )
    parser.add_argument("--host", type=str, default=REMOTE_LAPTOP_HOST, help="Remote laptop IP")
    parser.add_argument("--port", type=int, default=REMOTE_STREAM_PORT, help="Remote MJPEG port")
    parser.add_argument("--path", type=str, default=REMOTE_STREAM_PATH, help="Remote stream path")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="", help="cuda device id or 'cpu'")
    return parser.parse_args()


def resolve_source(args: argparse.Namespace) -> str | int:
    if args.local:
        return int(args.source) if args.source.isdigit() else 0
    if args.source:
        if args.source.isdigit():
            return build_remote_url(args.host, args.port, args.path)
        return parse_source(args.source, local=False)
    return DEFAULT_STREAM_URL


def main() -> None:
    args = parse_args()
    source = resolve_source(args)

    model = YOLO(args.model)
    cap = open_capture(source)
    if not cap.isOpened():
        msg = f"Cannot open source={source}."
        if not args.local:
            msg += (
                f"\nOn laptop {args.host}, start the integrated camera stream first:\n"
                f"  python publish_webcam.py --port {args.port}\n"
                f"Then run this script again from your current machine."
            )
        raise SystemExit(msg)

    window = "Person Detection (press q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    mode = "local integrated webcam" if args.local else f"remote laptop stream ({args.host})"
    print(f"Model: {args.model}")
    print(f"Mode: {mode}")
    print(f"Source: {source}")
    print("Press 'q' in the video window to exit.")

    predict_kwargs = {"conf": args.conf, "classes": [PERSON_CLASS_ID], "verbose": False}
    if args.device:
        predict_kwargs["device"] = args.device

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(f"Failed to read frame from {source}.")
            break

        results = model(frame, **predict_kwargs)
        annotated = results[0].plot()
        count = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(
            annotated,
            f"persons: {count}",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
