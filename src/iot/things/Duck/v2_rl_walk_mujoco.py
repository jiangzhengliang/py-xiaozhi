"""
强化学习鸭子机器人控制系统
用于控制14自由度的鸭子机器人进行实时行走控制

主要功能：
1. 基于ONNX模型的强化学习控制
2. 语音命令控制
3. IMU传感器数据处理
4. 足部接触传感器
5. 表情控制（眼睛、天线、投影仪）
6. 50Hz实时控制循环
"""

import time
import pickle
import threading

import numpy as np
from src.iot.things.Duck.rustypot_position_hwi import HWI
from src.iot.things.Duck.onnx_infer import OnnxInfer
from src.iot.things.Duck.raw_imu import Imu
from src.iot.things.Duck.poly_reference_motion import PolyReferenceMotion
from src.iot.things.Duck.feet_contacts import FeetContacts
from src.iot.things.Duck.eyes import Eyes
from src.iot.things.Duck.antennas import Antennas
from src.iot.things.Duck.projector import Projector
from src.iot.things.Duck.rl_utils import make_action_dict, LowPassActionFilter
from src.iot.things.Duck.duck_config import DuckConfig

import os

HOME_DIR = os.path.expanduser("~")


class RLWalk:
    """
    强化学习步行控制器
    
    该类实现了基于强化学习的鸭子机器人实时控制系统，支持：
    - 14自由度电机控制
    - IMU姿态反馈
    - 足部接触检测
    - 语音命令控制
    - 表情控制
    - 50Hz实时控制循环
    """
    
    def __init__(
        self,
        onnx_model_path: str,  # ONNX模型路径
        duck_config_path: str = None,  # 鸭子配置文件路径
        serial_port: str = "/dev/ttyACM0",  # 串口设备路径
        control_freq: float = 40,  # 控制频率（Hz）
        pid=[30, 0, 0],  # PID控制参数 [P, I, D]
        action_scale=0.25,  # 动作缩放因子
        pitch_bias=0,  # 俯仰角偏置
        save_obs=False,  # 是否保存观测数据
        replay_obs=None,  # 重放观测数据文件路径
        cutoff_frequency=None,  # 低通滤波器截止频率
        hardware_available=True,  # 硬件是否可用
    ):
        # 设置默认配置文件路径
        if duck_config_path is None:
            duck_config_path = os.path.join(os.path.dirname(__file__), "duck_config.json")
        
        # 硬件可用性
        self.hardware_available = hardware_available
        
        # 加载鸭子配置
        try:
            self.duck_config = DuckConfig(config_json_path=duck_config_path, ignore_default=True)
        except Exception as e:
            print(f"Warning: Could not load duck config: {e}")
            self.duck_config = DuckConfig(config_json_path=None, ignore_default=True)

        # 存储配置参数
        self.pitch_bias = pitch_bias
        self.running = False
        self.control_thread = None

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

        # 语音控制命令
        self.voice_command = None
        self.voice_command_start_time = 0
        self.voice_command_duration = 5.0  # 语音命令持续时间（秒）
        self.voice_command_lock = threading.Lock()

        if self.hardware_available:
            try:
                # 初始化ONNX强化学习模型
                self.onnx_model_path = onnx_model_path
                self.policy = OnnxInfer(self.onnx_model_path, awd=True)

                # 初始化硬件接口
                self.hwi = HWI(self.duck_config, serial_port)

                # 启动电机系统
                self.start_hardware()

                # 初始化IMU传感器
                self.imu = Imu(
                    sampling_freq=int(self.control_freq),
                    user_pitch_bias=self.pitch_bias,
                    upside_down=self.duck_config.imu_upside_down,
                )

                # 初始化足部接触传感器
                self.feet_contacts = FeetContacts()

                # 可选的表情控制组件
                if self.duck_config.eyes:
                    self.eyes = Eyes()
                if self.duck_config.projector:
                    self.projector = Projector()
                if self.duck_config.antennas:
                    self.antennas = Antennas()
                    
            except Exception as e:
                print(f"Hardware initialization failed: {e}")
                self.hardware_available = False

        # 动作缩放参数
        self.action_scale = action_scale

        # 存储历史动作（用于网络输入）
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        if self.hardware_available:
            # 电机初始位置
            self.init_pos = list(self.hwi.init_pos.values())
            # 电机目标位置
            self.motor_targets = np.array(self.init_pos.copy())
            self.prev_motor_targets = np.array(self.init_pos.copy())
        else:
            # 模拟模式的默认位置
            self.init_pos = [0.0] * self.num_dofs
            self.motor_targets = np.array(self.init_pos.copy())
            self.prev_motor_targets = np.array(self.init_pos.copy())

        # 控制命令（7维，但只使用前3维进行语音控制）
        self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [lin_vel_x, lin_vel_y, ang_vel, neck_pitch, head_pitch, head_yaw, head_roll]

        # 暂停状态
        self.paused = self.duck_config.start_paused if self.hardware_available else False

        # 参考运动生成器（用于相位信息）
        if self.hardware_available:
            poly_coeff_path = os.path.join(os.path.dirname(__file__), "polynomial_coefficients.pkl")
            if os.path.exists(poly_coeff_path):
                self.PRM = PolyReferenceMotion(poly_coeff_path)
            else:
                print("Warning: polynomial_coefficients.pkl not found, using default phase")
                self.PRM = None
        else:
            self.PRM = None
            
        self.imitation_i = 0  # 模仿索引
        self.imitation_phase = np.array([0, 0])  # 相位信息
        self.phase_frequency_factor = 1.0  # 相位频率因子
        self.phase_frequency_factor_offset = (
            self.duck_config.phase_frequency_factor_offset if self.hardware_available else 0.0
        )

    def set_voice_command(self, command):
        """
        设置语音命令
        
        Args:
            command (str): 语音命令 ('forward', 'backward', 'left', 'right', 'turn_left', 'turn_right', 'stop')
        """
        with self.voice_command_lock:
            self.voice_command = command
            self.voice_command_start_time = time.time()
            # 强制切换为非头部控制模式
            self.paused = False
            print(f"Voice command set: {command}")

    def get_voice_command(self):
        """
        获取当前有效的语音命令
        
        Returns:
            str or None: 当前有效的语音命令，如果超时则返回None
        """
        with self.voice_command_lock:
            if self.voice_command is None:
                return None
            
            # 检查命令是否超时
            if time.time() - self.voice_command_start_time > self.voice_command_duration:
                self.voice_command = None
                return None
            
            return self.voice_command

    def get_obs(self):
        """
        获取机器人当前状态观测数据
        
        Returns:
            numpy.ndarray: 包含以下信息的观测向量
        """
        if not self.hardware_available:
            # 模拟模式返回虚拟观测数据
            # 构建与硬件模式相同维度的观测向量
            obs = np.concatenate([
                np.zeros(3),  # IMU陀螺仪数据 (3维)
                np.zeros(3),  # IMU加速度计数据 (3维)
                self.last_commands,  # 命令输入 (4维)
                np.zeros(self.num_dofs),  # 关节位置偏差 (14维)
                np.zeros(self.num_dofs),  # 关节速度 (14维)
                self.last_action,  # 上一次动作 (14维)
                self.last_last_action,  # 上上次动作 (14维)
                self.last_last_last_action,  # 上上上次动作 (14维)
                self.motor_targets,  # 电机目标位置 (14维)
                np.zeros(4),  # 足部接触状态 (4维)
                self.imitation_phase,  # 模仿相位 (2维)
            ])
            return obs
            
        try:
            # 获取IMU数据
            imu_data = self.imu.get_data()

            # 获取关节位置（排除天线）
            dof_pos = self.hwi.get_present_positions(
                ignore=[
                    "left_antenna",
                    "right_antenna",
                ]
            )

            # 获取关节速度（排除天线）
            dof_vel = self.hwi.get_present_velocities(
                ignore=[
                    "left_antenna",
                    "right_antenna",
                ]
            )

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
        except Exception as e:
            print(f"Error getting observation: {e}")
            return None

    def start_hardware(self):
        """
        启动电机系统
        设置PID参数并开启电机
        """
        if not self.hardware_available:
            return
            
        try:
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
        except Exception as e:
            print(f"Error starting hardware: {e}")
            self.hardware_available = False

    def update_commands_from_voice(self):
        """
        根据语音命令更新控制命令
        """
        voice_cmd = self.get_voice_command()
        
        if voice_cmd is None:
            # 没有语音命令，停止移动
            self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            return

        # 根据语音命令设置控制参数（只修改前3维，保持头部控制为0）
        if voice_cmd == 'forward':
            self.last_commands = [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 向前
        elif voice_cmd == 'backward':
            self.last_commands = [-0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 向后
        elif voice_cmd == 'left':
            self.last_commands = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]  # 向左
        elif voice_cmd == 'right':
            self.last_commands = [0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0]  # 向右
        elif voice_cmd == 'turn_left':
            self.last_commands = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]  # 左转
        elif voice_cmd == 'turn_right':
            self.last_commands = [0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0]  # 右转
        elif voice_cmd == 'stop':
            self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 停止
        else:
            self.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 未知命令，停止

    def start_control_loop(self):
        """
        启动控制循环
        """
        if self.running:
            return
            
        self.running = True
        self.control_thread = threading.Thread(target=self.run, daemon=True)
        self.control_thread.start()

    def stop_control_loop(self):
        """
        停止控制循环
        """
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)

    def run(self):
        """
        主控制循环
        
        实现50Hz的实时控制循环，包括：
        1. 处理语音命令
        2. 获取机器人状态观测
        3. 运行强化学习策略
        4. 应用动作到电机
        """
        i = 0
        try:
            print("Starting duck control loop")
            start_t = time.time()
            
            while self.running:
                t = time.time()

                # 更新语音命令
                self.update_commands_from_voice()

                # 如果暂停，跳过控制循环
                if self.paused:
                    time.sleep(0.1)
                    continue

                if self.hardware_available:
                    # 获取机器人状态观测
                    obs = self.get_obs()
                    if obs is None:
                        continue

                    # 更新模仿相位
                    if self.PRM is not None:
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
                            print("BREAKING replay")
                            break

                    # 运行强化学习策略
                    action = self.policy.infer(obs)

                    # 更新历史动作
                    self.last_last_last_action = self.last_last_action.copy()
                    self.last_last_action = self.last_action.copy()
                    self.last_action = action.copy()

                    # 计算电机目标位置
                    self.motor_targets = self.init_pos + action * self.action_scale

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

                    # 创建动作字典
                    action_dict = make_action_dict(
                        self.motor_targets, list(self.hwi.joints.keys())
                    )

                    # 发送电机位置命令
                    self.hwi.set_position_all(action_dict)
                else:
                    # 模拟模式
                    print(f"Simulating command: {self.last_commands}")

                i += 1

                # 计算循环时间并保持控制频率
                took = time.time() - t
                if (1 / self.control_freq - took) < 0:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3),
                    )
                # 等待以保持控制频率
                time.sleep(max(0, 1 / self.control_freq - took))

        except Exception as e:
            print(f"Control loop error: {e}")
        finally:
            # 优雅关闭
            if self.hardware_available and self.duck_config.antennas:
                try:
                    self.antennas.stop()
                except:
                    pass

            # 保存观测数据（如果启用）
            if self.save_obs:
                pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
            print("Duck control loop stopped")

    def pause(self):
        """暂停控制"""
        self.paused = True

    def resume(self):
        """恢复控制"""
        self.paused = False

    def get_status(self):
        """获取机器人状态"""
        return {
            "running": self.running,
            "paused": self.paused,
            "hardware_available": self.hardware_available,
            "current_command": self.last_commands,
            "voice_command": self.get_voice_command()
        }


if __name__ == "__main__":
    """
    命令行参数解析和主程序入口
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-a", "--action_scale", type=float, default=0.25)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("--pitch_bias", type=float, default=0, help="deg")
    parser.add_argument(
        "--save_obs",
        type=str,
        required=False,
        default=False,
        help="save the run's observations",
    )
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run",
    )
    parser.add_argument("--cutoff_frequency", type=float, default=None)

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
        pitch_bias=args.pitch_bias,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs,
        cutoff_frequency=args.cutoff_frequency,
    )
    print("Done instantiating RLWalk")
    # 运行主控制循环
    rl_walk.run()
