# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import sys
from pathlib import Path

import asyncio

sys.path.append(str(Path(__file__).parent.parent))
import rclpy
import threading
from cv_bridge import CvBridge
import numpy as np

import cv2

import websockets
from openpi_client import msgpack_numpy

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import (
    CompressedImage,
    JointState,
)
from std_msgs.msg import Bool
from collections import deque
import torch

QOS_PROFILE_LATEST = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class SimROSNode(Node):
    def __init__(self, node_name="univla_node"):
        super().__init__(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        # publish
        self.pub_joint_command = self.create_publisher(
            JointState,
            "/sim/target_joint_state",
            QOS_PROFILE_LATEST,
        )

        # subscribe
        self.sub_img_head = self.create_subscription(
            CompressedImage,
            "/sim/head_img",
            self.callback_rgb_image_head,
            1,
        )

        self.sub_img_left_wrist = self.create_subscription(
            CompressedImage,
            "/sim/left_wrist_img",
            self.callback_rgb_image_left_wrist,
            1,
        )

        self.sub_img_right_wrist = self.create_subscription(
            CompressedImage,
            "/sim/right_wrist_img",
            self.callback_rgb_image_right_wrist,
            1,
        )

        self.sub_js = self.create_subscription(
            JointState,
            "/joint_states",
            self.callback_joint_state,
            1,
        )

        self.sub_infer_start = self.create_subscription(
            Bool,
            "/sim/infer_start",
            self.callback_infer_start,
            1,
        )

        # msg
        self.lock_img_head = threading.Lock()
        self.lock_img_left_wrist = threading.Lock()
        self.lock_img_right_wrist = threading.Lock()

        self.message_buffer = deque(maxlen=30)
        self.lock_joint_state = threading.Lock()
        self.obs_joint_state = JointState()
        self.cur_joint_state = JointState()
        self.infer_start = False

        # loop
        self.loop_rate = self.create_rate(30.0)

        self.img_head = None
        self.img_left_wrist = None
        self.img_right_wrist = None

    def callback_rgb_image_head(self, msg):
        # print(msg.header)
        with self.lock_img_head:
            self.img_head = msg

    def callback_rgb_image_left_wrist(self, msg):
        # print(msg.header)
        with self.lock_img_left_wrist:
            self.img_left_wrist = msg

    def callback_rgb_image_right_wrist(self, msg):
        # print(msg.header)
        with self.lock_img_right_wrist:
            self.img_right_wrist = msg

    def get_img_head(self):
        with self.lock_img_head:
            return self.img_head

    def get_img_left_wrist(self):
        with self.lock_img_left_wrist:
            return self.img_left_wrist

    def get_img_right_wrist(self):
        with self.lock_img_right_wrist:
            return self.img_right_wrist

    def publish_joint_command(self, action, is_end=False):
        cmd_msg = JointState()
        if is_end:
            cmd_msg.header.frame_id = "-1"

        cmd_msg.name = [
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx41_gripper_l_outer_joint1",
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
            "idx81_gripper_r_outer_joint1",
        ]
        cmd_msg.position = [0.0] * len(cmd_msg.name)
        cmd_msg.position[0] = action[0]
        cmd_msg.position[1] = action[1]
        cmd_msg.position[2] = action[2]
        cmd_msg.position[3] = action[3]
        cmd_msg.position[4] = action[4]
        cmd_msg.position[5] = action[5]
        cmd_msg.position[6] = action[6]
        cmd_msg.position[7] = np.clip((1 - action[7]), 0, 1)
        cmd_msg.position[8] = action[8]
        cmd_msg.position[9] = action[9]
        cmd_msg.position[10] = action[10]
        cmd_msg.position[11] = action[11]
        cmd_msg.position[12] = action[12]
        cmd_msg.position[13] = action[13]
        cmd_msg.position[14] = action[14]
        cmd_msg.position[15] = np.clip((1 - action[15]), 0, 1)

        self.pub_joint_command.publish(cmd_msg)

    def callback_joint_state(self, msg):
        # print(msg.header)
        self.cur_joint_state = msg

        joint_name_state_dict = {}
        for idx, name in enumerate(msg.name):
            joint_name_state_dict[name] = msg.position[idx]

        msg_remap = JointState()
        msg_remap.header = msg.header
        msg_remap.name = []
        msg_remap.velocity = []
        msg_remap.effort = []
        msg_remap.position.append(joint_name_state_dict["idx21_arm_l_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx22_arm_l_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx23_arm_l_joint3"])
        msg_remap.position.append(joint_name_state_dict["idx24_arm_l_joint4"])
        msg_remap.position.append(joint_name_state_dict["idx25_arm_l_joint5"])
        msg_remap.position.append(joint_name_state_dict["idx26_arm_l_joint6"])
        msg_remap.position.append(joint_name_state_dict["idx27_arm_l_joint7"])
        left_gripper_pos = min(1, max(0.0, (0.8 - (joint_name_state_dict["idx41_gripper_l_outer_joint1"]))))
        msg_remap.position.append(left_gripper_pos)

        msg_remap.position.append(joint_name_state_dict["idx61_arm_r_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx62_arm_r_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx63_arm_r_joint3"])
        msg_remap.position.append(joint_name_state_dict["idx64_arm_r_joint4"])
        msg_remap.position.append(joint_name_state_dict["idx65_arm_r_joint5"])
        msg_remap.position.append(joint_name_state_dict["idx66_arm_r_joint6"])
        msg_remap.position.append(joint_name_state_dict["idx67_arm_r_joint7"])
        right_gripper_pos = min(1, max(0.0, (0.8 - (joint_name_state_dict["idx81_gripper_r_outer_joint1"]))))
        msg_remap.position.append(right_gripper_pos)

        msg_remap.position.append(joint_name_state_dict["idx01_body_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx02_body_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx11_head_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx12_head_joint2"])


        with self.lock_joint_state:
            self.obs_joint_state = msg_remap

    def get_joint_state(self):
        with self.lock_joint_state:
            return self.obs_joint_state

    def callback_infer_start(self, msg):
        self.infer_start = msg.data

    def is_infer_start(self):
        return self.infer_start


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
    """Thin client that queries a remote websocket policy server.

    Protocol matches test_client.py:
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

        self._start_loop_thread()
        self._ensure_connected_sync()

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

    @staticmethod
    def _dict_actions_to_joint_sequence(action_payload: dict) -> np.ndarray:
        """Convert a server action dict into a (T,16) joint action sequence.

        Supports payloads like:
          - {"action.left_arm_joint_position": (T,7), ...}
          - {"action": {"left_arm_joint_position": (T,7), ...}}

        Output ordering matches publish_joint_command():
          [left_arm(7), left_eff(1), right_arm(7), right_eff(1)]
        """

        # Normalize to flat keys "action.*"
        flat: dict[str, object] = {}
        for k, v in action_payload.items():
            if k.startswith("action."):
                flat[k] = v

        nested = action_payload.get("action")
        if isinstance(nested, dict):
            for k, v in nested.items():
                kk = k if k.startswith("action.") else f"action.{k}"
                flat[kk] = v

        required = (
            "action.left_arm_joint_position",
            "action.right_arm_joint_position",
            "action.left_effector_position",
            "action.right_effector_position",
        )
        missing = [k for k in required if k not in flat]
        if missing:
            raise ValueError(f"Missing required action keys: {missing}. Available keys: {sorted(flat.keys())}")

        left_arm = np.asarray(flat["action.left_arm_joint_position"])
        right_arm = np.asarray(flat["action.right_arm_joint_position"])
        left_eff = np.asarray(flat["action.left_effector_position"])
        right_eff = np.asarray(flat["action.right_effector_position"])

        if left_arm.ndim != 2 or left_arm.shape[1] != 7:
            raise ValueError(f"action.left_arm_joint_position must be (T,7), got {left_arm.shape}")
        if right_arm.ndim != 2 or right_arm.shape[1] != 7:
            raise ValueError(f"action.right_arm_joint_position must be (T,7), got {right_arm.shape}")

        T = left_arm.shape[0]
        if right_arm.shape[0] != T:
            raise ValueError(
                f"Mismatched time dimension: left_arm T={T}, right_arm T={right_arm.shape[0]}"
            )

        # Effector positions may arrive as (T,) or (T,1)
        if left_eff.ndim == 1:
            left_eff = left_eff.reshape(-1, 1)
        if right_eff.ndim == 1:
            right_eff = right_eff.reshape(-1, 1)

        if left_eff.ndim != 2 or left_eff.shape[1] != 1:
            raise ValueError(f"action.left_effector_position must be (T,) or (T,1), got {left_eff.shape}")
        if right_eff.ndim != 2 or right_eff.shape[1] != 1:
            raise ValueError(f"action.right_effector_position must be (T,) or (T,1), got {right_eff.shape}")
        if left_eff.shape[0] != T or right_eff.shape[0] != T:
            raise ValueError(
                f"Mismatched time dimension: T={T}, left_eff T={left_eff.shape[0]}, right_eff T={right_eff.shape[0]}"
            )

        joint_seq = np.concatenate([left_arm, left_eff, right_arm, right_eff], axis=1).astype(np.float64, copy=False)
        return joint_seq

    @staticmethod
    def _extract_action_array(action_payload):
        """Extract an action array from multiple possible server payload formats."""
        if isinstance(action_payload, np.ndarray):
            return action_payload

        if isinstance(action_payload, dict):
            return RemotePolicyClient._dict_actions_to_joint_sequence(action_payload)

        raise ValueError(f"Unsupported action payload type: {type(action_payload)}")

    @staticmethod
    def _to_action_queue(action_array: np.ndarray) -> deque:
        arr = np.asarray(action_array)
        arr = np.squeeze(arr)

        # Accept shapes like (16,), (T,16), (1,T,16)
        if arr.ndim == 1:
            if arr.size != 16:
                raise ValueError(f"Expected 16D action, got shape {arr.shape}")
            return deque([arr.astype(np.float64, copy=False)])

        if arr.ndim == 2:
            if arr.shape[1] != 16:
                raise ValueError(f"Expected (T,16) action sequence, got shape {arr.shape}")
            return deque([arr[i].astype(np.float64, copy=False) for i in range(arr.shape[0])])

        raise ValueError(f"Unsupported action array shape: {arr.shape}")

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
        # Prevent concurrent send/recv on one websocket.
        with self._lock:
            try:
                fut = asyncio.run_coroutine_threadsafe(self._step_async(obs), self._loop)
                payload = fut.result(timeout=self._request_timeout_s + 5.0)
            except Exception:
                # Attempt one reconnect + retry.
                self._ensure_connected_sync()
                fut = asyncio.run_coroutine_threadsafe(self._step_async(obs), self._loop)
                payload = fut.result(timeout=self._request_timeout_s + 5.0)

        action_array = self._extract_action_array(payload)
        return self._to_action_queue(action_array)


def get_instruction(task_name):
    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
    elif task_name == "iros_clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm."
    elif task_name == "iros_stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "iros_heat_the_food_in_the_microwave":
        lang = "Open the door of the microwave oven with the right arm.;Pick up the plate with bread on the table with the right arm.;Put the plate containing bread into the microwave oven with the right arm.;Push the plate that was not placed properly into the microwave oven the right arm.;Close the door of the microwave oven with the left arm.;Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "iros_open_drawer_and_store_items":
        lang = "Pull the top drawer of the drawer cabinet with the right arm.;Pick up the Rubik's Cube on the drawer cabinet with the right arm.;Place the Rubik's Cube into the drawer with the right arm.;Push the top drawer of the drawer cabinet with the right arm."
    elif task_name == "iros_pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm;Place the hand cream held in the right arm into the box on the table"
    elif task_name == "iros_pickup_items_from_the_freezer":
        lang = "Open the freezer door with the right arm;Pick up the caviar from the freezer with the right arm;Place the caviar held in the right arm into the shopping cart;Close the freezer door with both arms"
    elif task_name == "iros_make_a_sandwich":
        lang = "Pick up the bread slice from the toaster on the table with the right arm;Place the picked bread slice into the plate on the table with the right arm;Pick up the ham slice from the box on the table with the left arm;Place the picked ham slice onto the bread slice in the plate on the table with the left arm;Pick up the lettuce slice from the box on the table with the right arm;Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm;Pick up the bread slice from the toaster on the table with the right arm;Place the bread slice onto the lettuce slice in the plate on the table with the right arm"
    else:
        raise ValueError("task does not exist")
    return lang


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
