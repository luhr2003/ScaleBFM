#!/usr/bin/env python3
"""FAST-LIO (ROS 2) -> pose wire bridge for the legged estimator's LiDAR fusion.

ROS 2 port of MagicLoco's `sim2real/perception/fastlio_bridge.py` (ROS 1), odom mode only. Subscribes to FAST-LIO's
nav_msgs/Odometry (the LiDAR body pose in FAST-LIO's odom frame) and republishes every pose on a ZeroMQ PUB socket in the
pose-wire format that `pose_wire.PoseSubscriber` / `LidarOdometryFusion` read (body = 0, mid360_link). It is the only file
of the LiDAR path that touches ROS, so ScaleBridge itself stays ROS-free.

Self-contained on purpose (rclpy + numpy + pyzmq, all in the system Python of a ROS 2 Humble install): run it in a ROS
terminal, not in the ScaleBridge Python environment.

    source /opt/ros/humble/setup.bash && source ~/livox_ws/install/setup.bash
    python3 fastlio_bridge_ros2.py --odom-topic /Odometry --endpoint tcp://*:5606
"""

import argparse
import json
import time

import numpy as np
import rclpy
import zmq
from nav_msgs.msg import Odometry
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

TOPIC = "pose"
HEADER_SIZE = 1280  # bytes of JSON header after the topic, right-padded with zeros (must match pose_wire.py)
BODY_MID360 = 0
SRC_FASTLIO = 0
_NP_DTYPES = {"u8": "u1", "i64": "<i8", "f32": "<f4", "f64": "<f8"}


def pack_pose(seq, t_mono, pos, quat_wxyz, body=BODY_MID360, src=SRC_FASTLIO):
    """One pose-wire message: [topic][1280-byte JSON header][little-endian payload]."""
    fields = [
        ("seq", "i64", [seq]),
        ("t_mono", "f64", [t_mono]),
        ("pos", "f32", pos),
        ("quat", "f32", quat_wxyz),
        ("body", "u8", [body]),
        ("src", "u8", [src]),
    ]
    meta, payload = [], b""
    for name, dtype, value in fields:
        array = np.atleast_1d(np.asarray(value)).astype(_NP_DTYPES[dtype])
        meta.append({"name": name, "dtype": dtype, "shape": list(array.shape)})
        payload += np.ascontiguousarray(array).tobytes()
    header = json.dumps({"v": 1, "endian": "le", "count": 1, "fields": meta}, separators=(",", ":")).encode()
    assert len(header) <= HEADER_SIZE, "pose-wire header too large"
    return TOPIC.encode() + header.ljust(HEADER_SIZE, b"\x00") + payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--odom-topic", default="/Odometry", help="FAST-LIO's nav_msgs/Odometry topic")
    parser.add_argument("--endpoint", default="tcp://*:5606", help="ZeroMQ PUB endpoint (ScaleBridge connects to PC2:5606)")
    parser.add_argument("--log-every", type=int, default=100, help="log one line every N forwarded poses (0 = off)")
    args = parser.parse_args()

    socket = zmq.Context.instance().socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 5)
    socket.bind(args.endpoint)

    rclpy.init()
    node = rclpy.create_node("fastlio_pose_bridge")
    state = {"seq": 0}

    def on_odometry(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        state["seq"] += 1
        # ROS quaternions are xyzw; the wire is wxyz
        socket.send(pack_pose(state["seq"], time.monotonic(), [p.x, p.y, p.z], [q.w, q.x, q.y, q.z]), zmq.NOBLOCK)
        if args.log_every and state["seq"] % args.log_every == 1:
            node.get_logger().info(f"{state['seq']} poses forwarded, last pos [{p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f}] m")

    # Keep only the newest pose; best effort is compatible with FAST-LIO's reliable publisher
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    node.create_subscription(Odometry, args.odom_topic, on_odometry, qos)
    node.get_logger().info(f"Forwarding {args.odom_topic} -> {args.endpoint} (pose wire, body=mid360_link)")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        socket.close(0)


if __name__ == "__main__":
    main()
