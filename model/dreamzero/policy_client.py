import asyncio
import atexit
import os
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import websockets
from openpi_client import msgpack_numpy


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
        video_path: str = "",
    ):
        self._uri = uri
        self._connect_timeout_s = float(connect_timeout_s)
        self._request_timeout_s = float(request_timeout_s)
        self._verbose = bool(verbose)
        self._video_path = video_path

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
        
        # Frames are concatenated (T, H, W*3, C) RGB
        _, height, width, _ = self.recorded_frames[0].shape
        
        # Write video with cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(self._video_path, fourcc, 15.0, (width, height))
        recorded_frames = np.concatenate(self.recorded_frames, axis=0)
        for frame in recorded_frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()
        if self._verbose:
            print(f"[RemotePolicyClient] Saving video with {len(recorded_frames)} frames...")

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
            await self._ws.close()
            self._ws = None

        self._ws = await asyncio.wait_for(
            websockets.connect(self._uri, max_size=None), timeout=self._connect_timeout_s
        )

        # Receive metadata (server sends it immediately on connect).
        metadata_raw = await asyncio.wait_for(
            self._ws.recv(), timeout=self._connect_timeout_s
        )
        metadata = msgpack_numpy.unpackb(metadata_raw)
        if self._verbose:
            print(f"[RemotePolicyClient] Connected to {self._uri}. Metadata: {metadata}")

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
        img_h = obs["video.top_head"].copy()
        img_l = obs["video.hand_left"].copy()
        img_r = obs["video.hand_right"].copy()
        combined = np.concatenate([img_l, img_h, img_r], axis=2)
        self.recorded_frames.append(combined)

        with self._lock:
            fut = asyncio.run_coroutine_threadsafe(self._step_async(obs), self._loop)
            payload = fut.result(timeout=self._request_timeout_s + 5.0)

        # Simplified action processing
        # Assume payload is a dict with the correct keys
        left_arm = np.asarray(payload["action.left_arm_joint_position"])
        right_arm = np.asarray(payload["action.right_arm_joint_position"])
        left_eff = np.asarray(payload["action.left_effector_position"])[:, None]
        right_eff = np.asarray(payload["action.right_effector_position"])[:, None]

        # Concatenate: [left_arm(7), left_eff(1), right_arm(7), right_eff(1)]
        joint_seq = np.concatenate(
            [left_arm, left_eff, right_arm, right_eff], axis=1
        ).astype(np.float64)
        return deque([step for step in joint_seq])
