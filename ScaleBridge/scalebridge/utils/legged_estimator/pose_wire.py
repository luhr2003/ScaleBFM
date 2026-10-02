import json
import threading
import time

import numpy as np
from loguru import logger

TOPIC = "pose"
HEADER_SIZE = 1280  # bytes of JSON header after the topic, right-padded with zeros
_DTYPES = {"u8": "u1", "bool": "u1", "i32": "<i4", "i64": "<i8", "f32": "<f4", "f64": "<f8"}

BODY_FRAMES = {0: "mid360_link", 1: "torso_link", 2: "pelvis"}  # 0: raw FAST-LIO body odometry, 1: torso (TF applied), 2: pelvis


def unpack_pose(raw):
    """Decode one message of the external-odometry pose wire (MagicLoco `sim2real/perception/pose_wire.py`).

    Frame = [topic ASCII][1280-byte JSON header][little-endian payload]; fields seq, t_mono, pos f32[3], quat f32[4] (wxyz), body u8,
    src u8. Returns (seq, position, quat_wxyz, body) or None for a message that is not a pose.
    """
    topic = TOPIC.encode()
    if not raw.startswith(topic) or len(raw) < len(topic) + HEADER_SIZE:
        return None
    header = json.loads(raw[len(topic):len(topic) + HEADER_SIZE].rstrip(b"\x00").decode())
    if header.get("endian", "le") != "le":
        return None
    fields, offset = {}, len(topic) + HEADER_SIZE
    for field in header["fields"]:
        count = int(np.prod(field["shape"])) if field["shape"] else 1
        dtype = np.dtype(_DTYPES[field["dtype"]])
        fields[field["name"]] = np.frombuffer(raw[offset:offset + count * dtype.itemsize], dtype=dtype).reshape(field["shape"] or (1,))
        offset += count * dtype.itemsize
    try:
        position = np.asarray(fields["pos"], dtype=np.float64).reshape(3)
        quat = np.asarray(fields["quat"], dtype=np.float64).reshape(4)
        return int(fields["seq"][0]), position, quat, int(fields["body"][0])
    except (KeyError, ValueError):
        return None


class PoseSubscriber:
    """Receives the LiDAR odometry poses that MagicLoco's `fastlio_bridge.py` publishes on PC2 (ZeroMQ, default port 5606)
    and hands each new one to `callback(position, quat_wxyz, body, receive_time)` on a background thread."""

    def __init__(self, endpoint, callback):
        self.endpoint = endpoint
        self.callback = callback
        self._running = threading.Event()
        self._thread = None
        self._last_seq = None
        self.last_receive_time = None

    def start(self):
        import zmq

        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, TOPIC)
        self._socket.setsockopt(zmq.CONFLATE, 1)  # latest pose only
        self._socket.connect(self.endpoint)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="LidarPoseSubscriber")
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
            if not self._socket.poll(50, zmq.POLLIN):
                continue
            decoded = unpack_pose(self._socket.recv())
            if decoded is None:
                continue
            seq, position, quat, body = decoded
            if seq == self._last_seq:
                continue
            self._last_seq = seq
            self.last_receive_time = time.monotonic()
            self.callback(position, quat, body, self.last_receive_time)
