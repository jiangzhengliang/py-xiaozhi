import os
import threading
import time
from typing import Dict, Any

import numpy as np

from src.iot.thing import Thing, Parameter, ValueType
from src.utils.logging_config import get_logger
from .v2_rl_walk_mujoco import RLWalk
from .rl_utils import make_action_dict

logger = get_logger(__name__)


class Duck(Thing):
    """
    鸭子机器人控制系统
    
    实现基于强化学习的鸭子机器人控制，支持：
    - 语音控制（向前/后/左/右移动、左右转弯）
    - Xbox手柄控制
    - 表情控制
    - 优先级处理：Xbox手柄优先于语音命令
    """
    
    def __init__(self):
        super().__init__("Duck", "14自由度鸭子机器人")
        
        # 状态变量
        self.is_initialized = False
        self.is_running = False
        self.current_action = "idle"
        self.last_voice_command = None
        self.voice_command_end_time = 0
        
        # 鸭子控制器
        self.rl_walk = None
        self.control_thread = None
        self.stop_event = threading.Event()
        
        # 语音命令映射
        self.voice_commands = {
            "forward": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # 向前走 (左摇杆Y轴)
            "backward": [-0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 向后走 (左摇杆Y轴)
            "left": [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],        # 向左走 (左摇杆X轴)
            "right": [0.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0],      # 向右走 (左摇杆X轴)
            "turn_left": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],   # 向左转弯 (右摇杆X轴)
            "turn_right": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], # 向右转弯 (右摇杆X轴)
        }
        
        # 初始化属性
        self.add_property("is_initialized", "机器人是否已初始化", lambda: self.is_initialized)
        self.add_property("is_running", "机器人是否正在运行", lambda: self.is_running)
        self.add_property("current_action", "当前动作", lambda: self.current_action)
        self.add_property("last_voice_command", "最后一次语音命令", lambda: self.last_voice_command)
        
        # 注册方法
        self._register_methods()
        
        # 初始化机器人
        self._initialize_robot()
        
        logger.info("Duck机器人初始化完成")
    
    def _register_methods(self):
        """注册所有控制方法"""
        
        # 基本控制方法
        self.add_method(
            "Initialize",
            "初始化鸭子机器人",
            [],
            lambda params: self._initialize_robot()
        )
        
        self.add_method(
            "Start",
            "启动鸭子机器人控制循环",
            [],
            lambda params: self._start_control()
        )
        
        self.add_method(
            "Stop",
            "停止鸭子机器人控制循环",
            [],
            lambda params: self._stop_control()
        )
        
        # 语音控制方法
        self.add_method(
            "MoveForward",
            "向前走5秒",
            [],
            lambda params: self._voice_command("forward")
        )
        
        self.add_method(
            "MoveBackward",
            "向后走5秒",
            [],
            lambda params: self._voice_command("backward")
        )
        
        self.add_method(
            "MoveLeft",
            "向左走5秒",
            [],
            lambda params: self._voice_command("left")
        )
        
        self.add_method(
            "MoveRight",
            "向右走5秒",
            [],
            lambda params: self._voice_command("right")
        )
        
        self.add_method(
            "TurnLeft",
            "向左转弯5秒",
            [],
            lambda params: self._voice_command("turn_left")
        )
        
        self.add_method(
            "TurnRight",
            "向右转弯5秒",
            [],
            lambda params: self._voice_command("turn_right")
        )
        
        # 手动控制方法
        self.add_method(
            "SetCommand",
            "设置控制命令",
            [
                Parameter("x", "X方向速度", ValueType.FLOAT, True),
                Parameter("y", "Y方向速度", ValueType.FLOAT, True),
                Parameter("yaw", "偏航角速度", ValueType.FLOAT, True),
                Parameter("duration", "持续时间（秒）", ValueType.FLOAT, False)
            ],
            lambda params: self._set_command(params)
        )
        
        # 状态查询方法
        self.add_method(
            "GetStatus",
            "获取机器人状态",
            [],
            lambda params: self._get_status()
        )
    
    def _initialize_robot(self):
        """初始化鸭子机器人"""
        try:
            # 获取ONNX模型路径
            onnx_model_path = os.path.join(os.path.dirname(__file__), "BEST_WALK_ONNX_2.onnx")
            
            if not os.path.exists(onnx_model_path):
                logger.error(f"ONNX模型文件不存在: {onnx_model_path}")
                return {"status": "error", "message": "ONNX模型文件不存在"}
            
            # 初始化RLWalk控制器
            self.rl_walk = RLWalk(
                onnx_model_path=onnx_model_path,
                commands=True,  # 启用外部命令控制
                control_freq=50,
                action_scale=0.25,
                pid=[30, 0, 0]
            )
            
            self.is_initialized = True
            logger.info("鸭子机器人初始化成功")
            
            return {"status": "success", "message": "鸭子机器人初始化成功"}
            
        except Exception as e:
            logger.error(f"初始化鸭子机器人失败: {e}")
            return {"status": "error", "message": f"初始化失败: {e}"}
    
    def _start_control(self):
        """启动控制循环"""
        try:
            if not self.is_initialized:
                return {"status": "error", "message": "机器人尚未初始化"}
            
            if self.is_running:
                return {"status": "success", "message": "机器人已在运行"}
            
            self.stop_event.clear()
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            
            self.is_running = True
            logger.info("鸭子机器人控制循环已启动")
            
            return {"status": "success", "message": "控制循环已启动"}
            
        except Exception as e:
            logger.error(f"启动控制循环失败: {e}")
            return {"status": "error", "message": f"启动失败: {e}"}
    
    def _stop_control(self):
        """停止控制循环"""
        try:
            if not self.is_running:
                return {"status": "success", "message": "机器人已停止"}
            
            self.stop_event.set()
            if self.control_thread:
                self.control_thread.join(timeout=2)
            
            self.is_running = False
            self.current_action = "idle"
            logger.info("鸭子机器人控制循环已停止")
            
            return {"status": "success", "message": "控制循环已停止"}
            
        except Exception as e:
            logger.error(f"停止控制循环失败: {e}")
            return {"status": "error", "message": f"停止失败: {e}"}
    
    def _voice_command(self, command: str):
        """处理语音命令"""
        try:
            if command not in self.voice_commands:
                return {"status": "error", "message": f"未知命令: {command}"}
            
            # 如果控制循环未启动，自动启动
            if self.is_initialized and not self.is_running:
                start_result = self._start_control()
                if start_result["status"] == "error":
                    return start_result
            
            # 设置语音命令
            self.last_voice_command = command
            self.voice_command_end_time = time.time() + 5.0  # 5秒后结束
            
            # 强制切换到非头部控制模式
            if self.rl_walk and hasattr(self.rl_walk, 'xbox_controller') and self.rl_walk.xbox_controller:
                self.rl_walk.xbox_controller.head_control_mode = False
            
            # 更新当前动作
            self.current_action = command
            
            logger.info(f"执行语音命令: {command}")
            
            return {"status": "success", "message": f"开始执行: {command}"}
            
        except Exception as e:
            logger.error(f"执行语音命令失败: {e}")
            return {"status": "error", "message": f"执行失败: {e}"}
    
    def _set_command(self, params: Dict[str, Any]):
        """设置手动控制命令"""
        try:
            x = params["x"].get_value()
            y = params["y"].get_value()
            yaw = params["yaw"].get_value()
            duration = params.get("duration", None)
            
            if duration is None:
                duration = 5.0
            else:
                duration = duration.get_value()
            
            # 设置手动命令
            self.last_voice_command = "manual"
            self.voice_command_end_time = time.time() + duration
            
            # 创建命令数组
            manual_command = [x, y, yaw, 0.0, 0.0, 0.0, 0.0]
            self.voice_commands["manual"] = manual_command
            
            self.current_action = "manual"
            
            logger.info(f"设置手动命令: x={x}, y={y}, yaw={yaw}, duration={duration}")
            
            return {"status": "success", "message": f"设置手动命令成功"}
            
        except Exception as e:
            logger.error(f"设置手动命令失败: {e}")
            return {"status": "error", "message": f"设置失败: {e}"}
    
    def _get_status(self):
        """获取机器人状态"""
        return {
            "status": "success",
            "data": {
                "is_initialized": self.is_initialized,
                "is_running": self.is_running,
                "current_action": self.current_action,
                "last_voice_command": self.last_voice_command,
                "voice_command_remaining": max(0, self.voice_command_end_time - time.time()) if self.last_voice_command else 0
            }
        }
    
    def _control_loop(self):
        """主控制循环"""
        logger.info("开始控制循环")
        
        try:
            # 直接运行带语音控制的RLWalk方法
            self._run_with_voice_control()
        except Exception as e:
            logger.error(f"控制循环出错: {e}")
        finally:
            logger.info("控制循环结束")
    
    def _run_with_voice_control(self):
        """带语音控制的RLWalk运行方法"""
        try:
            if not self.rl_walk:
                logger.error("RLWalk未初始化")
                return
                
            # 修改RLWalk的run方法以支持语音控制
            i = 0
            start_t = time.time()
            
            while not self.stop_event.is_set():
                current_time = time.time()
                
                # 初始化触发器状态
                left_trigger = 0
                right_trigger = 0
                
                # 获取Xbox手柄输入
                if self.rl_walk.commands and hasattr(self.rl_walk, 'xbox_controller') and self.rl_walk.xbox_controller:
                    try:
                        xbox_commands, buttons, left_trigger, right_trigger = (
                            self.rl_walk.xbox_controller.get_last_command()
                        )
                        
                        # 检查Xbox手柄是否有输入
                        xbox_has_input = (abs(xbox_commands[0]) > 0.01 or 
                                         abs(xbox_commands[1]) > 0.01 or 
                                         abs(xbox_commands[2]) > 0.01)
                        
                        if xbox_has_input:
                            # Xbox手柄有输入，优先使用Xbox手柄
                            self.rl_walk.last_commands = xbox_commands
                            self.current_action = "xbox_control"
                        else:
                            # 检查是否有语音命令
                            if (self.last_voice_command and 
                                current_time < self.voice_command_end_time and
                                self.last_voice_command in self.voice_commands):
                                
                                # 强制切换到非头部控制模式
                                self.rl_walk.xbox_controller.head_control_mode = False
                                
                                # 使用语音命令
                                voice_command = self.voice_commands[self.last_voice_command]
                                self.rl_walk.last_commands = voice_command
                                self.current_action = self.last_voice_command
                            else:
                                # 没有语音命令，使用默认命令
                                self.rl_walk.last_commands = xbox_commands
                                self.current_action = "idle"
                    except Exception as e:
                        logger.warning(f"Xbox手柄读取失败: {e}")
                        # 如果Xbox手柄失败，只使用语音命令
                        if (self.last_voice_command and 
                            current_time < self.voice_command_end_time and
                            self.last_voice_command in self.voice_commands):
                            
                            voice_command = self.voice_commands[self.last_voice_command]
                            self.rl_walk.last_commands = voice_command
                            self.current_action = self.last_voice_command
                        else:
                            self.rl_walk.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                            self.current_action = "idle"
                else:
                    # 没有Xbox手柄，检查语音命令
                    if (self.last_voice_command and 
                        current_time < self.voice_command_end_time and
                        self.last_voice_command in self.voice_commands):
                        
                        voice_command = self.voice_commands[self.last_voice_command]
                        self.rl_walk.last_commands = voice_command
                        self.current_action = self.last_voice_command
                    else:
                        self.rl_walk.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        self.current_action = "idle"
                
                # 暂停控制
                if hasattr(self.rl_walk, 'paused') and self.rl_walk.paused:
                    time.sleep(0.1)
                    continue
                
                # 获取观测数据
                try:
                    obs = self.rl_walk.get_obs()
                    if obs is None:
                        logger.warning("观测数据为空，跳过此次循环")
                        time.sleep(0.02)
                        continue
                except Exception as e:
                    logger.warning(f"获取观测数据失败: {e}")
                    time.sleep(0.02)
                    continue
                
                # 保存观测数据（如果需要）
                if hasattr(self.rl_walk, 'save_obs') and self.rl_walk.save_obs:
                    self.rl_walk.saved_obs.append(obs)
                
                # 运行策略推理
                if hasattr(self.rl_walk, 'replay_obs') and self.rl_walk.replay_obs is not None:
                    # 重放模式
                    if i < len(self.rl_walk.replay_obs):
                        obs = self.rl_walk.replay_obs[i]
                
                # 获取动作
                try:
                    if hasattr(self.rl_walk, 'policy') and self.rl_walk.policy:
                        action = self.rl_walk.policy.infer(obs)
                    else:
                        action = np.zeros(self.rl_walk.num_dofs)
                except Exception as e:
                    logger.warning(f"策略推理失败: {e}")
                    action = np.zeros(self.rl_walk.num_dofs)
                
                # 应用动作滤波器
                if hasattr(self.rl_walk, 'action_filter') and self.rl_walk.action_filter:
                    try:
                        action = self.rl_walk.action_filter.filter(action)
                    except:
                        pass
                
                # 更新历史动作
                self.rl_walk.last_last_last_action = self.rl_walk.last_last_action
                self.rl_walk.last_last_action = self.rl_walk.last_action
                self.rl_walk.last_action = action
                
                # 计算电机目标位置
                self.rl_walk.prev_motor_targets = self.rl_walk.motor_targets.copy()
                self.rl_walk.motor_targets = (
                    np.array(self.rl_walk.init_pos) + action * self.rl_walk.action_scale
                )
                
                # 创建动作字典并发送给硬件
                try:
                    if make_action_dict:
                        action_dict = make_action_dict(
                            self.rl_walk.motor_targets,
                            self.rl_walk.prev_motor_targets,
                            ignore=[
                                "left_antenna",
                                "right_antenna",
                            ],
                        )
                        
                        if hasattr(self.rl_walk, 'hwi') and self.rl_walk.hwi:
                            self.rl_walk.hwi.set_position_all(action_dict)
                except Exception as e:
                    logger.warning(f"硬件控制失败: {e}")
                
                # 控制表情组件
                try:
                    if hasattr(self.rl_walk, 'antennas') and self.rl_walk.antennas:
                        self.rl_walk.antennas.set_position_left(left_trigger)
                        self.rl_walk.antennas.set_position_right(right_trigger)
                except Exception as e:
                    logger.warning(f"天线控制失败: {e}")
                
                # 控制频率
                target_time = start_t + (i + 1) / self.rl_walk.control_freq
                current_time = time.time()
                sleep_time = target_time - current_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.001)  # 最小延迟
                
                i += 1
                
        except Exception as e:
            logger.error(f"控制循环运行出错: {e}")
        finally:
            logger.info("控制循环运行结束")
    
    def __del__(self):
        """析构函数"""
        try:
            self._stop_control()
        except Exception:
            pass 