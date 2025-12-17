# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import asyncio
import atexit
import os
import sys
import threading
from collections import deque
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import rclpy
import numpy as np
import torch
import websockets
from cv_bridge import CvBridge
from openpi_client import msgpack_numpy

from genie_sim_ros import SimROSNode
from instructions import get_instruction


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_sim_time(sim_ros_node):
    sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
    return sim_time


def resize_rgb(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if img is None:
        return img
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {img.shape}")
    if img.shape[0] == target_h and img.shape[1] == target_w:
        return img.astype(np.uint8, copy=False)
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8, copy=False)


def infer(policy, cfg):
    rclpy.init()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()
    init_frame = True
    bridge = CvBridge()
    count = 0
    SIM_INIT_TIME = 10
    action_queue = None
    instruction = (cfg or {}).get("instruction")
    target_h, target_w = 480, 640

    while rclpy.ok():
        if action_queue:
            is_end = True if len(action_queue) == 1 else False
            sim_ros_node.publish_joint_command(action_queue.popleft(), is_end)

        else:
            img_h_raw = sim_ros_node.get_img_head()
            img_l_raw = sim_ros_node.get_img_left_wrist()
            img_r_raw = sim_ros_node.get_img_right_wrist()
            act_raw = sim_ros_node.get_joint_state()
            infer_start = sim_ros_node.is_infer_start()

            if (init_frame or infer_start) and (
                img_h_raw
                and img_l_raw
                and img_r_raw
                and act_raw
                and img_h_raw.header.stamp
                == img_l_raw.header.stamp
                == img_r_raw.header.stamp
            ):
                sim_time = get_sim_time(sim_ros_node)
                if sim_time > SIM_INIT_TIME:
                    init_frame = False
                    count = count + 1

                    img_h = bridge.compressed_imgmsg_to_cv2(
                        img_h_raw, desired_encoding="rgb8"
                    )

                    img_l = bridge.compressed_imgmsg_to_cv2(
                        img_l_raw, desired_encoding="rgb8"
                    )

                    img_r = bridge.compressed_imgmsg_to_cv2(
                        img_r_raw, desired_encoding="rgb8"
                    )

                    # Remote policy server expects 480x640x3.
                    img_h = resize_rgb(img_h, target_h, target_w)
                    img_l = resize_rgb(img_l, target_h, target_w)
                    img_r = resize_rgb(img_r, target_h, target_w)

                    state = np.array(act_raw.position)

                    # Build an observation matching test_client.py format.
                    # Video observations are batched: (num_frames, H, W, C)
                    obs = {
                        "video.top_head": img_h[None, ...],
                        "video.hand_left": img_l[None, ...],
                        "video.hand_right": img_r[None, ...],
                    }

                    # State remap: demo_infer JointState is
                    #   [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1), waist_pitch(1), waist_lift(1), head_position(2)]
                    # Server expects:
                    #   left_arm(7), right_arm(7), left_effector(1), right_effector(1), head_position(2), waist_pitch(1), waist_lift(1)
                    left_arm = state[0:7]
                    left_eff = state[7:8]
                    right_arm = state[8:15]
                    right_eff = state[15:16]
                    waist_pitch = state[16:17]
                    waist_lift = state[17:18]
                    head_pos = state[18:20]

                    obs.update(
                        {
                            "state.left_arm_joint_position": left_arm[None, :].astype(np.float64, copy=False),
                            "state.right_arm_joint_position": right_arm[None, :].astype(np.float64, copy=False),
                            "state.left_effector_position": left_eff[None, :].astype(np.float64, copy=False),
                            "state.right_effector_position": right_eff[None, :].astype(np.float64, copy=False),
                            "state.head_position": head_pos[None, :].astype(np.float64, copy=False),
                            "state.waist_pitch": waist_pitch[None, :].astype(np.float64, copy=False),
                            "state.waist_lift": waist_lift[None, :].astype(np.float64, copy=False),
                            "annotation.language.action_text": instruction or "",
                        }
                    )

                    action_queue = policy.step(obs)


        sim_ros_node.loop_rate.sleep()

class RemotePolicyClient:
    """Policy client that queries a remote websocket policy server.

    Protocol:
    - Connect to ws://host:port
    - Server sends a metadata msgpack message on connect
    - Client sends msgpack-packed observation dict
    - Server replies with msgpack-packed action payload (dict or ndarray)
    """

    def __init__(
        self,
        uri: str,
        *,
        connect_timeout_s: float = 10.0,
        request_timeout_s: float = 60.0,
        verbose: bool = True,
    ):
        self._uri = uri
        self._connect_timeout_s = float(connect_timeout_s)
        self._request_timeout_s = float(request_timeout_s)
        self._verbose = bool(verbose)

        self._loop = None
        self._thread = None
        self._ws = None
        self._packer = msgpack_numpy.Packer()
        self._connected = threading.Event()
        self._lock = threading.Lock()

        self.recorded_frames = []
        atexit.register(self.save_video)

        self._start_loop_thread()
        self._ensure_connected_sync()

    def save_video(self):
        if not self.recorded_frames:
            return
        
        print(f"[RemotePolicyClient] Saving debug video with {len(self.recorded_frames)} frames...")
        try:
            # Frames are concatenated (H, W*3, C) RGB
            height, width, _ = self.recorded_frames[0].shape
            # Use a filename that won't be overwritten easily or just a fixed one
            filename = "debug_obs.mp4"
            
            # cv2 VideoWriter expects BGR
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
            
            for frame in self.recorded_frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            video.release()
            print(f"[RemotePolicyClient] Saved {filename}")
        except Exception as e:
            print(f"[RemotePolicyClient] Failed to save video: {e}")

    def _start_loop_thread(self):
        loop = asyncio.new_event_loop()
        self._loop = loop

        def _run():
            asyncio.set_event_loop(loop)
            loop.run_forever()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._thread = t

    async def _connect_async(self):
        self._connected.clear()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._ws = await asyncio.wait_for(
            websockets.connect(self._uri, max_size=None), timeout=self._connect_timeout_s
        )

        # Receive metadata (server sends it immediately on connect).
        try:
            metadata_raw = await asyncio.wait_for(
                self._ws.recv(), timeout=self._connect_timeout_s
            )
            metadata = msgpack_numpy.unpackb(metadata_raw)
            if self._verbose:
                print(f"[RemotePolicyClient] Connected to {self._uri}. Metadata: {metadata}")
        except Exception as e:
            if self._verbose:
                print(f"[RemotePolicyClient] Connected to {self._uri}, but failed to read metadata: {e}")

        self._connected.set()

    def _ensure_connected_sync(self):
        fut = asyncio.run_coroutine_threadsafe(self._connect_async(), self._loop)
        fut.result(timeout=self._connect_timeout_s + 5.0)

    async def _step_async(self, obs: dict):
        if self._ws is None:
            await self._connect_async()

        # Serialize + send
        await self._ws.send(self._packer.pack(obs))

        # Receive
        action_raw = await asyncio.wait_for(self._ws.recv(), timeout=self._request_timeout_s)
        if isinstance(action_raw, str):
            raise RuntimeError(f"Server returned error:\n{action_raw}")
        return msgpack_numpy.unpackb(action_raw)

    def step(self, obs: dict) -> deque:
        # Record observation for debugging
        try:
            # Extract latest frames (assuming shape T,H,W,C)
            img_h = obs["video.top_head"][-1]
            img_l = obs["video.hand_left"][-1]
            img_r = obs["video.hand_right"][-1]
            
            # Concatenate horizontally: Left Wrist | Head | Right Wrist
            combined = np.concatenate([img_l, img_h, img_r], axis=1)
            self.recorded_frames.append(combined)
        except Exception:
            pass

        with self._lock:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._step_async(obs), self._loop)
                payload = fut.result(timeout=self._request_timeout_s + 5.0)
            except Exception:
                self._ensure_connected_sync()
                fut = asyncio.run_coroutine_threadsafe(self._step_async(obs), self._loop)
                payload = fut.result(timeout=self._request_timeout_s + 5.0)

        # Simplified action processing
        # Assume payload is a dict with the correct keys
        left_arm = np.asarray(payload["action.left_arm_joint_position"])
        right_arm = np.asarray(payload["action.right_arm_joint_position"])
        left_eff = np.asarray(payload["action.left_effector_position"])
        right_eff = np.asarray(payload["action.right_effector_position"])

        # Ensure 2D shapes (T, D)
        if left_eff.ndim == 1: 
            left_eff = left_eff[:, None]
        if right_eff.ndim == 1: 
            right_eff = right_eff[:, None]

        # Concatenate: [left_arm(7), left_eff(1), right_arm(7), right_eff(1)]
        joint_seq = np.concatenate([left_arm, left_eff, right_arm, right_eff], axis=1)
        
        return deque([step.astype(np.float64) for step in joint_seq])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ROS inference client (websocket remote policy)")
    parser.add_argument("--host", default="localhost", help="Policy server hostname")
    parser.add_argument("--port", type=int, default=6000, help="Policy server port")
    parser.add_argument(
        "--uri",
        default=None,
        help="Full websocket URI (overrides --host/--port), e.g. ws://10.0.0.2:6000",
    )
    parser.add_argument(
        "--task_name",
        default="remove eraser from the whiteboard with right arm",
        help="Language instruction sent as annotation.language.action_text",
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--quiet", action="store_true", help="Reduce client logging")
    args = parser.parse_args()

    uri = args.uri or f"ws://{args.host}:{args.port}"
    policy = RemotePolicyClient(
        uri,
        connect_timeout_s=args.connect_timeout,
        request_timeout_s=args.request_timeout,
        verbose=not args.quiet,
    )

    infer(policy, {"instruction": get_instruction(args.task_name)})
