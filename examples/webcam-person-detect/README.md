# Laptop Webcam Person Detection

Detect people using a **laptop integrated webcam**, either locally or over the LAN.

Default setup: integrated webcam on laptop **192.168.51.65**, viewed from another PC on the same network.

## Local MP4 / video file

Detect **person** only in a local video and save the annotated result:

```bash
cd examples/webcam-person-detect
python detect_video.py /path/to/your.mp4 --model ../../weights/yolo26n.pt
```

Preview while processing:

```bash
python detect_video.py your.mp4 --model ../../weights/yolo26n.pt --show
```

Output directory: `runs/detect/video/` (same name as input, with boxes drawn).

One-line CLI:

```bash
yolo predict model=weights/yolo26n.pt source=your.mp4 classes=0 save=True
```

## Step 1 — On laptop `192.168.51.65` (camera machine)

Publish the integrated webcam as an HTTP MJPEG stream:

```bash
python publish_webcam.py
python publish_webcam.py --port 8080 --resolution 640x480
```

Stream URL: `http://192.168.51.65:8080/video`

`--resolution` sets capture width/height (e.g. `640x480`). The camera may fall back to the nearest supported mode; check the startup log for the actual size.

## Step 2 — On your current PC (detection machine)

```bash
python main.py --model weights/yolo26n.pt
```

Default connects to `http://192.168.51.65:8080/video`.

## Run detection on the same laptop (local integrated cam)

If you run `main.py` **on** `192.168.51.65` and want to use the built-in camera directly:

```bash
python main.py --model weights/yolo26n.pt --local
```

## Options

```bash
python main.py --host 192.168.51.65 --port 8080 --path /video
python main.py --source "rtsp://192.168.51.65:8554/live"   # custom stream URL
python main.py --local --source 0
python main.py --device cpu
```

Press **q** in the video window to quit.

## Web UI（服务端浏览器查看，无需本地窗口）

在检测服务器上启动 Web 服务（默认读取 `http://192.168.51.65:8080/video`）：

```bash
cd examples/webcam-person-detect
python web_detect.py --model ../../weights/yolo26n.pt --port 5000
```

浏览器打开（将 `<server-ip>` 换成运行脚本的机器 IP）：

```
http://<server-ip>:5000/
```

- 页面实时显示带检测框的画面
- 显示当前检测到的人数
- 标注视频流地址：`http://<server-ip>:5000/stream`

指定输入流：

```bash
python web_detect.py --model ../../weights/yolo26n.pt \
  --source "http://192.168.51.65:8080/video" --port 5000
```

## Scan subnet for cameras

```bash
python scan_cameras.py                        # auto local /24
python scan_cameras.py --subnet 192.168.51.0/24
nmap -p 80,554,8080,8554 --open 192.168.51.0/24
```

## CLI (Ultralytics)

```bash
yolo predict model=weights/yolo26n.pt source=http://192.168.51.65:8080/video classes=0 show=True
```
