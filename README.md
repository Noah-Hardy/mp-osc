# Live Pose, Hand, and Face Landmark Analysis with OSC Streaming

## Overview

`live_analysis.py` is a Python script that uses [MediaPipe](https://google.github.io/mediapipe/) to perform real-time detection and tracking of human pose, hand, and face landmarks from a webcam video stream. The script visualizes these landmarks on the video feed and streams the landmark data over [OSC (Open Sound Control)](https://opensoundcontrol.stanford.edu/) to a specified network address, enabling integration with multimedia and creative coding environments.

Additionally, the script calculates the pose landmark indices corresponding to the maximum and minimum x and y coordinates (the "bounds" of the pose) and sends this information, including each bound's id, x, y, z, and visibility, on a dedicated OSC channel.

## Features

- **Real-time pose, hand, and face landmark detection** using MediaPipe Holistic.
- **Landmark visualization**: Draws landmarks and their indices on the video feed for pose, left hand, and right hand.
- **OSC streaming**: Sends each landmark's data as a JSON message on a unique OSC channel (e.g., `/pose/0`, `/right_hand/5`).
- **Pose bounds calculation**: Computes and streams the indices and values of the landmarks with the maximum and minimum x and y coordinates on the `/pose/bounds` OSC channel.
- **Configurable OSC target**: Easily change the OSC destination IP and port.

## Prerequisites

- **Python 3.7+**
- **A working webcam** (or Apple Continuity Camera, or other video device)
- **Network access** to the OSC target (if not running locally)
- **[uv](https://github.com/astral-sh/uv) package manager** (required)

### Required Python Packages

Install the following packages using [uv](https://github.com/astral-sh/uv):

```sh
uv pip install opencv-python mediapipe python-osc
```

## Usage
1. Clone or download this repository and navigate to the project directory.

2. Edit the OSC target address (optional):
    By default, the script sends OSC messages to 192.168.1.28:1234.
    To change this, edit the following line in live_analysis.py:

    ```python
    osc_client = udp_client.SimpleUDPClient("192.168.1.28", 1234)
    ```
    Replace the IP and port with your OSC receiver's address.

3. Run the script using uv:

```sh
uv venv
source .venv/bin/activate
uv pip install opencv-python mediapipe python-osc
uv run python live_analysis.py
```

4. View the output:
    - A window will open showing the webcam feed with pose, hand, and (optionally) face landmarks and their indices.
    - OSC messages will be sent in real time for each detected landmark and for the pose bounds.

5. Quit:


## OSC Message Structure

- Per-landmark messages
Sent on `/pose/{idx}`, `/right_hand/{idx}`, `/left_hand{idx}` as JSON Strings
```json
{
  "timestamp": 1720000000.123,
  "id": 15,
  "x": 0.5,
  "y": 0.6,
  "z": -0.1,
  "visibility": 0.98
}
```
(Hand landmarks do not include visibility)
- Pose bounds message
Sent on `/pose/bounds` as a JSON String
```json
{
  "max_x": {"id": 23, "x": 0.9, "y": 0.5, "z": -0.1, "visibility": 0.99},
  "min_x": {"id": 11, "x": 0.1, "y": 0.6, "z": -0.2, "visibility": 0.98},
  "max_y": {"id": 27, "x": 0.5, "y": 0.95, "z": -0.3, "visibility": 0.97},
  "min_y": {"id": 0, "x": 0.4, "y": 0.05, "z": -0.4, "visibility": 0.96}
}
```

## Customization

- Change video source:
By default, the script uses cv2.VideoCapture(0).
To use a different camera (e.g., Apple Continuity Camera), change the index to 1 or another appropriate value.

- Add or remove features:
You can comment out or modify sections for face, hand, or pose detection as needed.

## Troubleshooting
- If the webcam does not open, ensure no other application is using it and try changing the camera index.
- If OSC messages are not received, check your firewall, network, and OSC receiver address/port.
- For best performance, use a recent computer and a good lighting environment.

## License
This project is based on MediaPipe and is licensed under the Apache License 2.0.

---
#### Author:
Noah Hardy
