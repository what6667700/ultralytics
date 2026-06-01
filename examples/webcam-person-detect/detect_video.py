# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Person detection on local video files (mp4, avi, mov, ...)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

PERSON_CLASS_ID = 0
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts", ".mpeg", ".mpg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people in a local video file.")
    parser.add_argument("video", nargs="?", default="", help="Path to video, e.g. demo.mp4")
    parser.add_argument("--source", type=str, default="", help="Same as video path")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="YOLO weights")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="", help="cuda device id or 'cpu'")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated video (default)")
    parser.add_argument("--no-save", action="store_true", help="Do not save output video")
    parser.add_argument("--show", action="store_true", help="Show preview window while processing")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project directory")
    parser.add_argument("--name", type=str, default="video", help="Run name under project")
    return parser.parse_args()


def resolve_video_path(args: argparse.Namespace) -> Path:
    path_str = args.source or args.video
    if not path_str:
        raise SystemExit("Provide a video path: python detect_video.py your.mp4")
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Video not found: {path}")
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        print(f"Warning: uncommon extension '{path.suffix}', trying anyway.")
    return path


def main() -> None:
    args = parse_args()
    video_path = resolve_video_path(args)

    print(f"Model: {args.model}")
    print(f"Video: {video_path}")

    model = YOLO(args.model)
    predict_kwargs: dict = {
        "source": str(video_path),
        "classes": [PERSON_CLASS_ID],
        "conf": args.conf,
        "stream": True,
        "verbose": True,
    }
    if args.device:
        predict_kwargs["device"] = args.device
    if args.save and not args.no_save:
        predict_kwargs["save"] = True
        predict_kwargs["project"] = args.project
        predict_kwargs["name"] = args.name
    if args.show:
        predict_kwargs["show"] = True

    for _ in model.predict(**predict_kwargs):
        pass

    if args.save and not args.no_save:
        out_dir = Path(args.project) / args.name
        print(f"Done. Annotated video saved under: {out_dir.resolve()}")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
