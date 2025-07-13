"""
强化学习鸭子机器人控制系统
用于控制14自由度的鸭子机器人进行实时行走控制

主要功能：
1. 基于ONNX模型的强化学习控制
2. Xbox手柄输入控制
3. IMU传感器数据处理
4. 足部接触传感器
5. 表情控制（眼睛、天线、投影仪）
6. 50Hz实时控制循环
"""

import time
import pickle

import numpy as np
from .rustypot_position_hwi import HWI
from .onnx_infer import OnnxInfer

from .raw_imu import Imu
from .poly_reference_motion import PolyReferenceMotion
from .xbox_controller import XBoxController
from .feet_contacts import FeetContacts
from .eyes import Eyes
from .antennas import Antennas
from .projector import Projector
from .rl_utils import make_action_dict, LowPassActionFilter
from .duck_config import DuckConfig

import os

HOME_DIR = os.path.expanduser("~")


class RLWalk:
    """
    强化学习步行控制器
    
    该类实现了基于强化学习的鸭子机器人实时控制系统，支持：
    - 14自由度电机控制
    - IMU姿态反馈
    - 足部接触检测
    - Xbox手柄命令输入
    - 表情和声音控制
    - 50Hz实时控制循环
    """
    
    def __init__(
        self,
        onnx_model_path: str,  # ONNX模型路径
        duck_config_path: str = None,  # 鸭子配置文件路径
        serial_port: str = "/dev/ttyACM0",  # 串口设备路径
        control_freq: float = 50,  # 控制频率（Hz）
        pid=[30, 0, 0],  # PID控制参数 [P, I, D]
        action_scale=0.25,  # 动作缩放因子
        commands=False,  # 是否启用外部命令控制
        pitch_bias=0,  # 俯仰角偏置
        save_obs=False,  # 是否保存观测数据
        replay_obs=None,  # 重放观测数据文件路径
        cutoff_frequency=None,  # 低通滤波器截止频率
    ):
        # 设置默认配置文件路径
        if duck_config_path is None:
            duck_config_path = os.path.join(os.path.dirname(__file__), "duck_config.json")
        
        # 加载鸭子配置
        self.duck_config = DuckConfig(config_json_path=duck_config_path)

        # 存储配置参数
        self.commands = commands
        self.pitch_bias = pitch_bias

        # 初始化ONNX强化学习模型
        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)

        # 机器人物理参数
        self.num_dofs = 14  # 自由度数量（不包括天线）
        self.max_motor_velocity = 5.24  # 最大电机速度 (rad/s)

        # 控制参数
        self.control_freq = control_freq
        self.pid = pid

        # 观测数据保存和重放
        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        # 动作滤波器（可选）
        self.action_filter = None
        if cutoff_frequency is not None:
            self.action_filter = LowPassActionFilter(
                self.control_freq, cutoff_frequency
            )

        # 初始化硬件接口
        self.hwi = HWI(self.duck_config, serial_port)

        # 启动电机系统
        self.start()

        # 初始化IMU传感器
        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            user_pitch_bias=self.pitch_bias,
            upside_down=self.duck_config.imu_upside_down,
        )

        # 初始化足部接触传感器
        self.feet_contacts = FeetContacts()

        # 动作缩放参数
        self.action_scale = action_scale

        # 存储历史动作（用于网络输入）
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        # 电机初始位置
        self.init_pos = list(self.hwi.init_pos.values())

        # 电机目标位置
        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        # 上一次的命令输入
        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 暂停状态
        self.paused = self.duck_config.start_paused

        # Xbox手柄控制
        self.command_freq = 20  # 命令频率 (Hz)
        if self.commands:
            self.xbox_controller = XBoxController(self.command_freq)

        # 参考运动生成器（用于相位信息）
        poly_coeff_path = os.path.join(os.path.dirname(__file__), "polynomial_coefficients.pkl")
        self.PRM = PolyReferenceMotion(poly_coeff_path)
        self.imitation_i = 0  # 模仿索引
        self.imitation_phase = np.array([0, 0])  # 相位信息
        self.phase_frequency_factor = 1.0  # 相位频率因子
        self.phase_frequency_factor_offset = (
            self.duck_config.phase_frequency_factor_offset
        )

        # 可选的表情控制组件
        if self.duck_config.eyes:
            self.eyes = Eyes()
        if self.duck_config.projector:
            self.projector = Projector()
        # 不再初始化sounds，因为由小智智能体处理
        if self.duck_config.antennas:
            self.antennas = Antennas()

    def get_obs(self):
        """
        获取机器人当前状态观测数据
        
        Returns:
            numpy.ndarray: 包含以下信息的观测向量
                - IMU陀螺仪数据 (3维)
                - IMU加速度计数据 (3维)
                - 命令输入 (7维)
                - 关节位置偏差 (14维)
                - 关节速度 (14维，缩放0.05)
                - 历史动作 (14维 × 3)
                - 电机目标位置 (14维)
                - 足部接触状态 (4维)
                - 模仿相位 (2维)
        """
        # 获取IMU数据
        imu_data = self.imu.get_data()

        # 获取关节位置（排除天线）
        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        # 获取关节速度（排除天线）
        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        # 数据有效性检查
        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs:
            print(f"ERROR len(dof_pos) != {self.num_dofs}")
            return None

        if len(dof_vel) != self.num_dofs:
            print(f"ERROR len(dof_vel) != {self.num_dofs}")
            return None

        # 获取命令输入
        cmds = self.last_commands

        # 获取足部接触状态
        feet_contacts = self.feet_contacts.get()

        # 组合观测向量
        obs = np.concatenate(
            [
                imu_data["gyro"],  # 陀螺仪数据
                imu_data["accelero"],  # 加速度计数据
                cmds,  # 命令输入
                dof_pos - self.init_pos,  # 关节位置偏差
                dof_vel * 0.05,  # 关节速度（缩放）
                self.last_action,  # 上一次动作
                self.last_last_action,  # 上上次动作
                self.last_last_last_action,  # 上上上次动作
                self.motor_targets,  # 电机目标位置
                feet_contacts,  # 足部接触状态
                self.imitation_phase,  # 模仿相位
            ]
        )

        return obs

    def start(self):
        """
        启动电机系统
        设置PID参数并开启电机
        """
        # 设置PID参数
        kps = [self.pid[0]] * 14  # 比例增益
        kds = [self.pid[2]] * 14  # 微分增益

        # 头部电机使用较小的增益
        kps[5:9] = [8, 8, 8, 8]

        # 应用PID参数
        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        # 等待电机启动
        time.sleep(2)

    def get_phase_frequency_factor(self, x_velocity):
        """
        根据x方向速度计算相位频率因子
        
        Args:
            x_velocity: x方向速度
            
        Returns:
            float: 相位频率因子
        """
        max_phase_frequency = 1.2
        min_phase_frequency = 1.0

        # 线性插值计算频率因子
        freq = min_phase_frequency + (abs(x_velocity) / 0.15) * (
            max_phase_frequency - min_phase_frequency
        )

        return freq

    def run(self):
        """
        主控制循环
        
        实现50Hz的实时控制循环，包括：
        1. 处理Xbox手柄输入
        2. 获取机器人状态观测
        3. 运行强化学习策略
        4. 应用动作到电机
        5. 控制表情组件
        """
        i = 0
        try:
            print("Starting")
            start_t = time.time()
            
            while True:
                # 初始化触发器状态
                left_trigger = 0
                right_trigger = 0
                t = time.time()

                # 处理Xbox手柄输入
                if self.commands:
                    self.last_commands, self.buttons, left_trigger, right_trigger = (
                        self.xbox_controller.get_last_command()
                    )
                    
                    # 方向键上：增加相位频率偏移
                    if self.buttons.dpad_up.triggered:
                        self.phase_frequency_factor_offset += 0.05
                        print(
                            f"Phase frequency factor offset {round(self.phase_frequency_factor_offset, 3)}"
                        )

                    # 方向键下：减少相位频率偏移
                    if self.buttons.dpad_down.triggered:
                        self.phase_frequency_factor_offset -= 0.05
                        print(
                            f"Phase frequency factor offset {round(self.phase_frequency_factor_offset, 3)}"
                        )

                    # LB键：加速行走
                    if self.buttons.LB.is_pressed:
                        self.phase_frequency_factor = 1.3
                    else:
                        self.phase_frequency_factor = 1.0

                    # X键：切换投影仪
                    if self.buttons.X.triggered:
                        if self.duck_config.projector:
                            self.projector.switch()

                    # B键：播放随机声音
                    if self.buttons.B.triggered:
                        if self.duck_config.speaker:
                            self.sounds.play_random_sound()

                    # 左右触发器：控制天线位置
                    if self.duck_config.antennas:
                        self.antennas.set_position_left(right_trigger)
                        self.antennas.set_position_right(left_trigger)

                    # A键：暂停/继续
                    if self.buttons.A.triggered:
                        self.paused = not self.paused
                        if self.paused:
                            print("PAUSE")
                        else:
                            print("UNPAUSE")

                # 如果暂停，跳过控制循环
                if self.paused:
                    time.sleep(0.1)
                    continue

                # 获取机器人状态观测
                obs = self.get_obs()
                if obs is None:
                    continue

                # 更新模仿相位
                self.imitation_i += 1 * (
                    self.phase_frequency_factor + self.phase_frequency_factor_offset
                )
                self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                self.imitation_phase = np.array(
                    [
                        np.cos(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                        np.sin(
                            self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
                        ),
                    ]
                )

                # 保存观测数据（可选）
                if self.save_obs:
                    self.saved_obs.append(obs)

                # 重放观测数据（可选）
                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        print("BREAKING ")
                        break

                # 运行强化学习策略
                action = self.policy.infer(obs)

                # 更新历史动作
                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                # 计算电机目标位置
                self.motor_targets = self.init_pos + action * self.action_scale

                # 可选：速度限制（已注释）
                # self.motor_targets = np.clip(
                #     self.motor_targets,
                #     self.prev_motor_targets
                #     - self.max_motor_velocity * (1 / self.control_freq),  # control dt
                #     self.prev_motor_targets
                #     + self.max_motor_velocity * (1 / self.control_freq),  # control dt
                # )

                # 应用低通滤波器（可选）
                if self.action_filter is not None:
                    self.action_filter.push(self.motor_targets)
                    filtered_motor_targets = self.action_filter.get_filtered_action()
                    if (
                        time.time() - start_t > 1
                    ):  # 给滤波器时间稳定
                        self.motor_targets = filtered_motor_targets

                # 保存上一次的电机目标位置
                self.prev_motor_targets = self.motor_targets.copy()

                # 头部电机目标位置 = 命令输入 + 策略输出
                head_motor_targets = self.last_commands[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets

                # 创建动作字典
                action_dict = make_action_dict(
                    self.motor_targets, list(self.hwi.joints.keys())
                )

                # 发送电机位置命令
                self.hwi.set_position_all(action_dict)

                i += 1

                # 计算循环时间并保持控制频率
                took = time.time() - t
                # print("Full loop took", took, "fps : ", np.around(1 / took, 2))
                if (1 / self.control_freq - took) < 0:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3),
                    )
                # 等待以保持控制频率
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            # 优雅关闭
            if self.duck_config.antennas:
                self.antennas.stop()

        # 保存观测数据（如果启用）
        if self.save_obs:
            pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
        print("TURNING OFF")


if __name__ == "__main__":
    """
    命令行参数解析和主程序入口
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)  # ONNX模型路径
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",  # 鸭子配置文件路径
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)  # 动作缩放
    parser.add_argument("-p", type=int, default=30)  # PID比例增益
    parser.add_argument("-i", type=int, default=0)   # PID积分增益
    parser.add_argument("-d", type=int, default=0)   # PID微分增益
    parser.add_argument("-c", "--control_freq", type=int, default=50)  # 控制频率
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")  # 俯仰角偏置
    parser.add_argument(
        "--commands",
        action="store_true",
        default=True,
        help="external commands, keyboard or gamepad. Launch control_server.py on host computer",
    )
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="save the run's observations",  # 保存观测数据
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",  # 重放观测数据
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)  # 滤波器截止频率

    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    print("Done parsing args")
    # 创建RLWalk实例
    rl_walk = RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        commands=args.commands,
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
    )
    print("Done instantiating RLWalk")
    # 运行主控制循环
    rl_walk.run()
