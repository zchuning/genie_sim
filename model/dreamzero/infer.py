# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import asyncio
import atexit
import os
import sys
import threading
from collections import deque
from datetime import datetime
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
from policy_client import RemotePolicyClient


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
    bridge = CvBridge()
    SIM_INIT_TIME = 10
    action_queue = None
    instruction = get_instruction(cfg["task_name"])

    # Buffers: keep dense states and subsampled images
    history_len = cfg["history_len"] 
    num_subsample_frames = cfg["num_subsample_frames"]
    subsample_interval = history_len // num_subsample_frames

    state_buffer = deque(maxlen=1)
    image_buffer = deque(maxlen=num_subsample_frames)
    init_frame = True
    frame_idx = 0
    is_chunk_end = False

    while rclpy.ok():
        # Wait for sim to be ready
        sim_time = get_sim_time(sim_ros_node)
        if sim_time <= SIM_INIT_TIME:
            sim_ros_node.loop_rate.sleep()
            continue
            
        img_h_raw = sim_ros_node.get_img_head()
        img_l_raw = sim_ros_node.get_img_left_wrist()
        img_r_raw = sim_ros_node.get_img_right_wrist()
        act_raw = sim_ros_node.get_joint_state()
        infer_start = sim_ros_node.is_infer_start()

        if ( 
            img_h_raw
            and img_l_raw
            and img_r_raw
            and act_raw
            and img_h_raw.header.stamp == img_l_raw.header.stamp == img_r_raw.header.stamp
        ):
            print("cur sim time", sim_time, img_h_raw.header.stamp) 
            print(f"init_frame: {init_frame}, infer_start: {infer_start}, frame_idx: {frame_idx}")

            if action_queue and not is_chunk_end:
                # Set is_end based on subsampling interval
                is_chunk_end = (len(action_queue) % subsample_interval) == 1
                sim_ros_node.publish_joint_command(action_queue.popleft(), is_chunk_end)
            elif init_frame or infer_start:
                print(f"========================> frame_idx: {frame_idx}, action_queue length: {len(action_queue) if action_queue else 0}")

                # Update image buffer
                img_h = bridge.compressed_imgmsg_to_cv2(img_h_raw, desired_encoding="rgb8")
                img_l = bridge.compressed_imgmsg_to_cv2(img_l_raw, desired_encoding="rgb8")
                img_r = bridge.compressed_imgmsg_to_cv2(img_r_raw, desired_encoding="rgb8")
                img_h = resize_rgb(img_h, *cfg["resize_shape"])
                img_l = resize_rgb(img_l, *cfg["resize_shape"])
                img_r = resize_rgb(img_r, *cfg["resize_shape"])
                image_buffer.append({"img_h": img_h, "img_l": img_l, "img_r": img_r})

                # Update state buffer
                state = np.array(act_raw.position).astype(np.float64)
                state_buffer.append(state)

                frame_idx += 1
                if is_chunk_end:
                    is_chunk_end = False

                if not action_queue:
                    img_h_seq = np.stack([f["img_h"] for f in image_buffer])
                    img_l_seq = np.stack([f["img_l"] for f in image_buffer])
                    img_r_seq = np.stack([f["img_r"] for f in image_buffer])

                    # Save images locally for debugging
                    # concatenate vertically: Left Wrist | Head | Right Wrist and then horizontally stack all frames
                    os.makedirs("debug_img", exist_ok=True)
                    debug_img_h = img_h_seq.transpose((1, 0, 2, 3)).reshape(cfg["resize_shape"][0], -1, 3)
                    debug_img_l = img_l_seq.transpose((1, 0, 2, 3)).reshape(cfg["resize_shape"][0], -1, 3)
                    debug_img_r = img_r_seq.transpose((1, 0, 2, 3)).reshape(cfg["resize_shape"][0], -1, 3)
                    debug_img = np.concatenate([debug_img_l, debug_img_h, debug_img_r], axis=0)
                    cv2.imwrite(f"debug_img/frame_{frame_idx:04d}.png", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

                    # State remap: sim JointState is
                    #   [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1), waist_pitch(1), waist_lift(1), head_position(2)]
                    # Server expects:
                    #   left_arm(7), right_arm(7), left_effector(1), right_effector(1), head_position(2), waist_pitch(1), waist_lift(1)
                    states = np.stack(state_buffer)
                    left_arm = states[:, 0:7]
                    left_eff = states[:, 7:8]
                    right_arm = states[:, 8:15]
                    right_eff = states[:, 15:16]
                    waist_lift = states[:, 16:17]
                    waist_pitch = states[:, 17:18]
                    head_pos = states[:, 18:20]

                    obs = {
                        "video.top_head": img_h_seq,
                        "video.hand_left": img_l_seq,
                        "video.hand_right": img_r_seq,
                        "state.left_arm_joint_position": left_arm,
                        "state.right_arm_joint_position": right_arm,
                        "state.left_effector_position": left_eff,
                        "state.right_effector_position": right_eff,
                        "state.waist_lift": waist_lift,
                        "state.waist_pitch": waist_pitch,
                        "state.head_position": head_pos,
                        "annotation.language.action_text": instruction,
                    }
                    
                    state = {k: v for k, v in obs.items() if k.startswith("state.")}
                    from pprint import pformat
                    pretty_state = pformat(state, indent=2, compact=False)
                    print(f"==============> Inferrring from state\n{pretty_state}")

                    # print all observation shapes
                    for key, value in obs.items():
                        print(f"obs[{key}]: {value.shape if isinstance(value, np.ndarray) else type(value)}")

                    action_queue = policy.step(obs)
                    init_frame = False


        for _ in range(5):
            sim_ros_node.loop_rate.sleep()

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
    video_dir = Path("videos") / datetime.now().strftime("%Y-%m-%d")
    video_dir.mkdir(parents=True, exist_ok=True)
    policy = RemotePolicyClient(
        uri,
        connect_timeout_s=args.connect_timeout,
        request_timeout_s=args.request_timeout,
        verbose=not args.quiet,
        video_path=video_dir / f"{args.task_name}_{datetime.now().strftime('%H-%M-%S')}.mp4",
    )

    cfg = {
        "history_len": 48,
        "num_subsample_frames": 4,
        "resize_shape": (480, 640),
        "task_name": args.task_name,
    }
    infer(policy, cfg)
