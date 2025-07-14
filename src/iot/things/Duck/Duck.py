"""
鸭子机器人控制系统
基于强化学习的14自由度鸭子机器人，支持语音控制
"""

import os
import threading
import time
from typing import Dict, Any

from src.iot.thing import Thing
from src.iot.things.Duck.v2_rl_walk_mujoco import RLWalk


class Duck(Thing):
    """
    鸭子机器人控制器
    
    基于强化学习的14自由度鸭子机器人，支持语音控制：
    - 向前走
    - 向后走
    - 向左走
    - 向右走
    - 向左转弯
    - 向右转弯
    """
    
    def __init__(self):
        super().__init__("Duck", "基于强化学习的14自由度鸭子机器人")
        
        # 初始化状态
        self.is_initialized = False
        self.is_running = False
        self.current_status = "idle"
        self.hardware_available = True
        self.rl_walk = None
        self.status_lock = threading.Lock()
        
        # 配置路径
        self.duck_config_path = os.path.join(os.path.dirname(__file__), "duck_config.json")
        self.onnx_model_path = os.path.join(os.path.dirname(__file__), "BEST_WALK_ONNX_2.onnx")
        
        # 检查必要文件是否存在
        if not os.path.exists(self.onnx_model_path):
            print(f"Warning: ONNX model not found at {self.onnx_model_path}")
            self.hardware_available = False
        
        print("[鸭子机器人] 初始化完成")
        
        # 注册属性
        self.add_property("is_initialized", "鸭子机器人是否已初始化", lambda: self.is_initialized)
        self.add_property("is_running", "鸭子机器人是否正在运行", lambda: self.is_running)
        self.add_property("current_status", "当前状态", lambda: self.current_status)
        self.add_property("hardware_available", "硬件是否可用", lambda: self.hardware_available)
        
        # 注册方法
        self.add_method("Start", "启动鸭子机器人（自动初始化并启动控制循环）", [], lambda params: self._start())
        self.add_method("MoveForward", "向前走", [], lambda params: self._move_forward())
        self.add_method("MoveBackward", "向后走", [], lambda params: self._move_backward())
        self.add_method("MoveLeft", "向左走", [], lambda params: self._move_left())
        self.add_method("MoveRight", "向右走", [], lambda params: self._move_right())
        self.add_method("TurnLeft", "向左转弯", [], lambda params: self._turn_left())
        self.add_method("TurnRight", "向右转弯", [], lambda params: self._turn_right())
        self.add_method("Pause", "暂停鸭子机器人", [], lambda params: self._pause())
        self.add_method("Resume", "恢复鸭子机器人", [], lambda params: self._resume())
        self.add_method("GetStatus", "获取鸭子机器人状态", [], lambda params: self._get_status())
        
    def _initialize(self):
        """初始化鸭子机器人"""
        if self.is_initialized:
            return {"status": "success", "message": "鸭子机器人已经初始化"}
        
        try:
            # 检查硬件可用性
            if not self.hardware_available:
                print("[鸭子机器人] 硬件不可用，使用模拟模式")
            
            # 创建RLWalk实例
            self.rl_walk = RLWalk(
                onnx_model_path=self.onnx_model_path,
                duck_config_path=self.duck_config_path,
                control_freq=50,
                pid=[30, 0, 0],
                action_scale=0.25,
                hardware_available=self.hardware_available
            )
            
            self.is_initialized = True
            self.current_status = "initialized"
            
            print("[鸭子机器人] 初始化成功")
            return {"status": "success", "message": "鸭子机器人初始化成功"}
            
        except Exception as e:
            error_msg = f"鸭子机器人初始化失败: {str(e)}"
            print(f"[鸭子机器人] {error_msg}")
            self.current_status = "error"
            return {"status": "error", "message": error_msg}
    
    def _start(self):
        """启动鸭子机器人（自动初始化并启动控制循环）"""
        if self.is_running:
            return {"status": "success", "message": "鸭子机器人已经在运行"}
        
        try:
            # 如果未初始化，先初始化
            if not self.is_initialized:
                init_result = self._initialize()
                if init_result["status"] != "success":
                    return init_result
            
            # 启动控制循环
            self.rl_walk.start_control_loop()
            self.is_running = True
            self.current_status = "running"
            
            print("[鸭子机器人] 启动成功（已初始化并启动控制循环）")
            return {"status": "success", "message": "鸭子机器人启动成功"}
            
        except Exception as e:
            error_msg = f"启动鸭子机器人失败: {str(e)}"
            print(f"[鸭子机器人] {error_msg}")
            self.current_status = "error"
            return {"status": "error", "message": error_msg}
    

    
    def _ensure_running(self):
        """确保鸭子机器人正在运行"""
        if not self.is_running:
            # 如果未运行，自动启动
            print("[鸭子机器人] 检测到机器人未运行，自动启动...")
            start_result = self._start()
            if start_result["status"] != "success":
                return {"status": "error", "message": "鸭子机器人自动启动失败，请先说'启动鸭子机器人'"}
            return {"status": "success"}
        return {"status": "success"}
    
    def _move_forward(self):
        """向前走5秒"""
        try:
            # 检查是否正在运行
            running_check = self._ensure_running()
            if running_check["status"] != "success":
                return running_check
            
            # 确保机器人处于运行状态（如果被暂停则恢复）
            self.rl_walk.resume()
            
            # 设置语音命令
            self.rl_walk.set_voice_command('forward')
            
            self.current_status = "moving_forward"
            print("[鸭子机器人] 开始向前走")
            
            return {"status": "success", "message": "鸭子机器人开始向前走"}
            
        except Exception as e:
            error_msg = f"向前走失败: {str(e)}"
            print(f"[鸭子机器人] {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def _move_backward(self):
        """向后走5秒"""
        try:
            # 检查是否正在运行
            running_check = self._ensure_running()
            if running_check["status"] != "success":
                return running_check
            
            # 确保机器人处于运行状态（如果被暂停则恢复）
            self.rl_walk.resume()
            
            # 设置语音命令
            self.rl_walk.set_voice_command('backward')
            
            self.current_status = "moving_backward"
            print("[鸭子机器人] 开始向后走")
            
            return {"status": "success", "message": "鸭子机器人开始向后走"}
            
        except Exception as e:
            error_msg = f"向后走失败: {str(e)}"
            print(f"[鸭子机器人] {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def _move_left(self):
        """向左走5秒"""
        try:
            # 检查是否正在运行
            running_check = self._ensure_running()
            if running_check["status"] != "success":
                return running_check
            
            # 确保机器人处于运行状态（如果被暂停则恢复）
            self.rl_walk.resume()
            
            # 设置语音命令
            self.rl_walk.set_voice_command('left')
            
            self.current_status = "moving_left"
            print("[鸭子机器人] 开始向左走")
            
            return {"status": "success", "message": "鸭子机器人开始向左走"}
            
        except Exception as e:
            error_msg = f"向左走失败: {str(e)}"
            print(f"[鸭子机器人] {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def _move_right(self):
        """向右走5秒"""
        try:
            # 检查是否正在运行
            running_check = self._ensure_running()
            if running_check["status"] != "success":
                return running_check
            
            # 确保机器人处于运行状态（如果被暂停则恢复）
            self.rl_walk.resume()
            
            # 设置语音命令
            self.rl_walk.set_voice_command('right')
            
            self.current_status = "moving_right"
            print("[鸭子机器人] 开始向右走")
            
            return {"status": "success", "message": "鸭子机器人开始向右走"}
            
        except Exception as e:
            error_msg = f"向右走失败: {str(e)}"
            print(f"[鸭子机器人] {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def _turn_left(self):
        """向左转弯5秒"""
        try:
            # 检查是否正在运行
            running_check = self._ensure_running()
            if running_check["status"] != "success":
                return running_check
            
            # 确保机器人处于运行状态（如果被暂停则恢复）
            self.rl_walk.resume()
            
            # 设置语音命令
            self.rl_walk.set_voice_command('turn_left')
            
            self.current_status = "turning_left"
            print("[鸭子机器人] 开始向左转弯")
            
            return {"status": "success", "message": "鸭子机器人开始向左转弯"}
            
        except Exception as e:
            error_msg = f"向左转弯失败: {str(e)}"
            print(f"[鸭子机器人] {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def _turn_right(self):
        """向右转弯5秒"""
        try:
            # 检查是否正在运行
            running_check = self._ensure_running()
            if running_check["status"] != "success":
                return running_check
            
            # 确保机器人处于运行状态（如果被暂停则恢复）
            self.rl_walk.resume()
            
            # 设置语音命令
            self.rl_walk.set_voice_command('turn_right')
            
            self.current_status = "turning_right"
            print("[鸭子机器人] 开始向右转弯")
            
            return {"status": "success", "message": "鸭子机器人开始向右转弯"}
            
        except Exception as e:
            error_msg = f"向右转弯失败: {str(e)}"
            print(f"[鸭子机器人] {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def _pause(self):
        """暂停鸭子机器人"""
        if not self.is_initialized or not self.rl_walk:
            return {"status": "error", "message": "鸭子机器人未初始化"}
        
        if not self.is_running:
            return {"status": "error", "message": "鸭子机器人未启动，请先说'启动鸭子机器人'"}
        
        try:
            self.rl_walk.pause()
            self.current_status = "paused"
            print("[鸭子机器人] 暂停成功")
            
            return {"status": "success", "message": "鸭子机器人暂停成功"}
            
        except Exception as e:
            error_msg = f"暂停鸭子机器人失败: {str(e)}"
            print(f"[鸭子机器人] {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def _resume(self):
        """恢复鸭子机器人"""
        if not self.is_initialized or not self.rl_walk:
            return {"status": "error", "message": "鸭子机器人未初始化"}
        
        if not self.is_running:
            return {"status": "error", "message": "鸭子机器人未启动，请先说'启动鸭子机器人'"}
        
        try:
            self.rl_walk.resume()
            self.current_status = "running"
            print("[鸭子机器人] 恢复成功")
            
            return {"status": "success", "message": "鸭子机器人恢复成功"}
            
        except Exception as e:
            error_msg = f"恢复鸭子机器人失败: {str(e)}"
            print(f"[鸭子机器人] {error_msg}")
            return {"status": "error", "message": error_msg}
    
    def _get_status(self):
        """获取鸭子机器人状态"""
        status = {
            "initialized": self.is_initialized,
            "running": self.is_running,
            "current_status": self.current_status,
            "hardware_available": self.hardware_available
        }
        
        if self.is_initialized and self.rl_walk:
            try:
                rl_status = self.rl_walk.get_status()
                status.update(rl_status)
            except Exception as e:
                status["rl_walk_error"] = str(e)
        
        return {"status": "success", "data": status}
    
    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            if self.is_running and self.rl_walk:
                # 不再调用stop_control_loop，因为该方法已被删除
                pass
        except:
            pass
