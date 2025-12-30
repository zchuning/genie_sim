# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import rclpy
import numpy as np
import torch
from cv_bridge import CvBridge
from openpi_client import image_tools, websocket_client_policy

from genie_sim_ros import SimROSNode
from instructions import get_instruction


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_sim_time(sim_ros_node):
    sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
    return sim_time

def infer(policy, cfg):
    rclpy.init()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()
    bridge = CvBridge()
    SIM_INIT_TIME = 10

    instruction = get_instruction(cfg["task_name"])
    exec_horizon = cfg["exec_horizon"]

    action_queue = None
    state_buffer = deque(maxlen=1)
    image_buffer = deque(maxlen=1)
    frame_idx = 0

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
            # print("cur sim time", sim_time, img_h_raw.header.stamp) 
            # print(f"init_frame: {init_frame}, infer_start: {infer_start}, frame_idx: {frame_idx}")            
            if action_queue:
                is_end = (len(action_queue) == 1)
                sim_ros_node.publish_joint_command(action_queue.popleft(), is_end)
            elif frame_idx == 0 or infer_start:
                # print(f"========================> frame_idx: {frame_idx}, action_queue length: {len(action_queue) if action_queue else 0}")

                # Update image buffer
                img_h = bridge.compressed_imgmsg_to_cv2(img_h_raw, desired_encoding="bgr8")
                img_l = bridge.compressed_imgmsg_to_cv2(img_l_raw, desired_encoding="bgr8")
                img_r = bridge.compressed_imgmsg_to_cv2(img_r_raw, desired_encoding="bgr8")
                img_h = image_tools.resize_with_pad(img_h, *cfg["resize_shape"])
                img_l = image_tools.resize_with_pad(img_l, *cfg["resize_shape"])
                img_r = image_tools.resize_with_pad(img_r, *cfg["resize_shape"])
                image_buffer.append({"img_h": img_h, "img_l": img_l, "img_r": img_r})

                # Update state buffer
                state = np.array(act_raw.position).astype(np.float64)
                state_buffer.append(state)

                if not action_queue:
                    img_h_seq = np.stack([f["img_h"] for f in image_buffer])[0]
                    img_l_seq = np.stack([f["img_l"] for f in image_buffer])[0]
                    img_r_seq = np.stack([f["img_r"] for f in image_buffer])[0]
                    states = np.stack(state_buffer)[0]

                    req = {
                        "observation/top_head": img_h_seq,
                        "observation/hand_left": img_l_seq,
                        "observation/hand_right": img_r_seq,
                        "observation/state": states,
                        "prompt": instruction,
                    }
                    

                    print(f"==============> Inferrring from state\n{state}")
                    for key, value in req.items():
                        print(f"req[{key}]: {value.shape if isinstance(value, np.ndarray) else type(value)}")

                    actions = policy.infer(req)["action"]
                    action_queue = deque([a for a in actions[: exec_horizon]])
                    print(f"==============> executing actions: {actions[0]}")

                frame_idx += 1

        for _ in range(2):
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

    policy = websocket_client_policy.WebsocketClientPolicy("0.0.0.0", args.port)

    cfg = {
        "resize_shape": (224, 224),
        "task_name": args.task_name,
        "exec_horizon": 50,
    }
    infer(policy, cfg)
