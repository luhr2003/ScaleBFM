import json
import threading
import time

import numpy as np
from loguru import logger

from scalebridge.utils.legged_estimator.depth_ground import CameraIntrinsics

TOPIC = b"cam1"


def unpack_depth(raw):
    """Decode one frame of MagicLoco's depth wire (`sim2real/perception/tools/rs_probe.py --serve`, ZeroMQ, port 5609).

    Frame = [topic cam1][4-byte header length][JSON header][depth float16 (h, w), metres][optional rgb uint8]; the header
    carries the intrinsics (fx, fy, ppx, ppy) and the depth scale. Returns (seq, depth float32, CameraIntrinsics) or None.
    """
    if not raw.startswith(TOPIC):
        return None
    length = int.from_bytes(raw[4:8], "little")
    header = json.loads(raw[8:8 + length].decode())
    height, width = header["d"]
    depth = np.frombuffer(raw, dtype=np.float16, count=height * width, offset=8 + length).reshape(height, width).astype(np.float32)
    meta = header["meta"]
    return int(header["seq"]), depth, CameraIntrinsics(fx=meta["fx"], fy=meta["fy"], cx=meta["ppx"], cy=meta["ppy"])


class DepthWireSubscriber:
    """Receives the D435i depth frames that `rs_probe.py --serve` publishes on PC2 (where the camera is plugged in) and calls
    `callback(depth, intrinsics, receive_time)` on a background thread, at most `hz` times per second."""

    def __init__(self, endpoint, callback, hz=10.0):
        self.endpoint = endpoint
        self.callback = callback
        self.period = 1.0 / hz
        self._running = threading.Event()
        self._thread = None
        self._last_seq = None
        self._last_time = 0.0

    def start(self):
        import zmq

        self._socket = zmq.Context.instance().socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, TOPIC)
        self._socket.setsockopt(zmq.CONFLATE, 1)  # latest frame only
        self._socket.connect(self.endpoint)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="DepthWireSubscriber")
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._socket.close(0)

    def _loop(self):
        import zmq

        while self._running.is_set():
            if not self._socket.poll(100, zmq.POLLIN):
                continue
            decoded = unpack_depth(self._socket.recv())
            now = time.monotonic()
            if decoded is None or decoded[0] == self._last_seq or now - self._last_time < self.period:
                continue
            self._last_seq, self._last_time = decoded[0], now
            self.callback(decoded[1], decoded[2], now)


class RealSenseDepthSource:
    """Reads the D435i directly with pyrealsense2, for ScaleBridge running on the Jetson (PC2) where the camera is plugged
    in. Same callback as `DepthWireSubscriber`. Only one process can open the camera at a time."""

    def __init__(self, callback, hz=10.0, width=424, height=240, fps=30):
        self.callback = callback
        self.period = 1.0 / hz
        self.width, self.height, self.fps = width, height, fps
        self._running = threading.Event()
        self._thread = None

    def start(self):
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = self._pipeline.start(config)
        self._scale = profile.get_device().first_depth_sensor().get_depth_scale()
        i = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self._intrinsics = CameraIntrinsics(fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="RealSenseDepth")
        self._thread.start()
        logger.info(f"[LeggedEstimator] D435i depth {i.width}x{i.height} at {self.fps} fps, fx={i.fx:.1f}.")

    def stop(self):
        self._running.clear()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._pipeline.stop()

    def _loop(self):
        last = 0.0
        while self._running.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                logger.warning("[LeggedEstimator] No D435i frame for 2 s.")
                continue
            now = time.monotonic()
            depth_frame = frames.get_depth_frame()
            if not depth_frame or now - last < self.period:
                continue
            last = now
            self.callback(np.asanyarray(depth_frame.get_data()).astype(np.float32) * self._scale, self._intrinsics, now)
