#!/usr/bin/env python3
"""Camera driver (host) — first real-world perception source into the system.

Reads the laptop camera, computes **motion** (target visible/near/dangerous) + a simplified
8-dimensional embedding vector, and prints one line per frame. **Privacy: no image is stored
or transmitted — numeric features only.**

Output is piped to `sil-realworld` (Rust) which builds the world memory + FSM + guard.
Usage: python3 drivers/camera_sensor.py [frames] | <sil-realworld>
"""
import sys
import time

import cv2


def main():
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERR no camera", flush=True)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
    prev = None
    sent = 0
    while sent < frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 24))
        if prev is None:
            prev = gray
            continue
        diff = cv2.absdiff(gray, prev)
        prev = gray
        area = float((diff > 25).mean())  # fraction of moving pixels = "motion/target"
        visible = 1 if area > 0.02 else 0
        near = 1 if area > 0.10 else 0
        unsafe = 1 if area > 0.35 else 0  # sudden coverage = dangerous approach
        emb = (cv2.resize(gray, (4, 2)).astype("float32") / 255.0).flatten()  # 8-d
        line = f"{visible} {near} {unsafe} " + " ".join(f"{v:.3f}" for v in emb)
        print(line, flush=True)
        sent += 1
        time.sleep(0.03)
    cap.release()


if __name__ == "__main__":
    main()
