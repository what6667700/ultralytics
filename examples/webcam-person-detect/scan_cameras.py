# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Scan the local subnet for hosts with common camera / streaming ports."""

from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import socket
from dataclasses import dataclass, field

# Common ports: RTSP, HTTP, MJPEG apps, ONVIF HTTP, Hikvision, DroidCam
CAMERA_PORTS = (80, 554, 8000, 8080, 8554, 4747, 37777)

# HTTP paths often used by webcam / IP camera MJPEG
PROBE_PATHS = ("/", "/video", "/stream", "/mjpeg", "/?action=stream", "/live")


@dataclass
class CameraHit:
    ip: str
    port: int
    service: str = ""
    hint: str = ""
    paths: list[str] = field(default_factory=list)


def get_local_subnet() -> str:
    """Guess local /24 subnet from default route interface."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        net = ipaddress.ip_network(f"{local_ip}/24", strict=False)
        return str(net)
    except OSError:
        return "192.168.1.0/24"


def port_open(ip: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def probe_http(ip: str, port: int, timeout: float) -> tuple[str, list[str]]:
    """Return server banner hint and paths that look like video streams."""
    hints: list[str] = []
    ok_paths: list[str] = []
    for path in PROBE_PATHS:
        try:
            req = (
                f"GET {path} HTTP/1.0\r\n"
                f"Host: {ip}\r\n"
                "Connection: close\r\n\r\n"
            ).encode()
            with socket.create_connection((ip, port), timeout=timeout) as sock:
                sock.sendall(req)
                data = sock.recv(2048)
            head = data.split(b"\r\n\r\n", 1)[0].decode(errors="ignore")
            if "200" not in head and "401" not in head:
                continue
            lower = head.lower()
            if path not in ok_paths:
                ok_paths.append(path)
            if "multipart/x-mixed-replace" in lower or "image/jpeg" in lower:
                hints.append(f"MJPEG@{path}")
            if "server:" in lower:
                for line in head.splitlines():
                    if line.lower().startswith("server:"):
                        hints.append(line.strip())
                        break
        except OSError:
            continue
    return ("; ".join(dict.fromkeys(hints)) if hints else ""), ok_paths


def probe_rtsp(ip: str, port: int, timeout: float) -> str:
    try:
        payload = "OPTIONS rtsp://{}/ RTSP/1.0\r\nCSeq: 1\r\n\r\n".format(ip).encode()
        with socket.create_connection((ip, port), timeout=timeout) as sock:
            sock.sendall(payload)
            data = sock.recv(1024).decode(errors="ignore")
        if "RTSP" in data:
            return "RTSP"
    except OSError:
        pass
    return ""


def scan_host(ip: str, ports: tuple[int, ...], timeout: float, probe: bool) -> list[CameraHit]:
    hits: list[CameraHit] = []
    for port in ports:
        if not port_open(ip, port, timeout):
            continue
        hit = CameraHit(ip=ip, port=port)
        if port == 554 or port == 8554:
            hit.service = probe_rtsp(ip, port, timeout) or "RTSP?"
        elif probe and port in (80, 8000, 8080, 4747):
            hint, paths = probe_http(ip, port, timeout)
            hit.hint = hint
            hit.paths = paths
            hit.service = "HTTP"
            if hint:
                hit.service = "HTTP/MJPEG?"
        else:
            hit.service = "open"
        hits.append(hit)
    return hits


def scan_subnet(
    cidr: str,
    ports: tuple[int, ...],
    timeout: float,
    workers: int,
    probe: bool,
) -> list[CameraHit]:
    network = ipaddress.ip_network(cidr, strict=False)
    hosts = [str(h) for h in network.hosts()]
    all_hits: list[CameraHit] = []

    def task(ip: str) -> list[CameraHit]:
        return scan_host(ip, ports, timeout, probe)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for hits in pool.map(task, hosts):
            all_hits.extend(hits)
    return sorted(all_hits, key=lambda h: (tuple(int(x) for x in h.ip.split(".")), h.port))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan subnet for camera-like services.")
    parser.add_argument(
        "--subnet",
        type=str,
        default="",
        help="CIDR to scan (default: auto-detect local /24)",
    )
    parser.add_argument(
        "--ports",
        type=str,
        default=",".join(map(str, CAMERA_PORTS)),
        help=f"Comma-separated ports (default: {','.join(map(str, CAMERA_PORTS))})",
    )
    parser.add_argument("--timeout", type=float, default=0.35, help="Socket timeout seconds")
    parser.add_argument("--workers", type=int, default=64, help="Parallel workers")
    parser.add_argument("--no-probe", action="store_true", help="Only check open ports, skip HTTP/RTSP probe")
    parser.add_argument("--nmap", action="store_true", help="Also print suggested nmap command")
    return parser.parse_args()


def print_hits(hits: list[CameraHit], cidr: str) -> None:
    if not hits:
        print(f"No camera-like ports found on {cidr}.")
        print("Try: --ports 554,8080,80  or scan 192.168.51.0/24 if camera is on another subnet.")
        return

    print(f"Found {len(hits)} candidate(s) on {cidr}:\n")
    print(f"{'IP':<16} {'Port':<6} {'Service':<14} {'Hint / stream URL'}")
    print("-" * 72)
    for h in hits:
        extra = h.hint
        if h.paths and h.port in (80, 8080, 8000, 4747):
            stream = f"http://{h.ip}:{h.port}{h.paths[0]}"
            extra = f"{extra} | {stream}" if extra else stream
        elif h.port in (554, 8554):
            stream = f"rtsp://{h.ip}:{h.port}/"
            extra = f"{extra} | {stream}" if extra else stream
        print(f"{h.ip:<16} {h.port:<6} {h.service:<14} {extra}")


def main() -> None:
    args = parse_args()
    cidr = args.subnet or get_local_subnet()
    ports = tuple(int(p.strip()) for p in args.ports.split(",") if p.strip())

    print(f"Scanning {cidr} ports {ports} (timeout={args.timeout}s) ...")
    hits = scan_subnet(cidr, ports, args.timeout, args.workers, probe=not args.no_probe)
    print_hits(hits, cidr)

    if args.nmap:
        port_arg = ",".join(map(str, ports))
        print(f"\n# nmap (if installed):\n"
              f"nmap -p {port_arg} --open -T4 {cidr}\n"
              f"nmap -p 554,8080 --script rtsp-url-brute,http-title {cidr}")


if __name__ == "__main__":
    main()
