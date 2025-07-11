import asyncio
import logging
import threading
from pathlib import Path

from src.application import Application
from src.constants.constants import DeviceState
from src.iot.thing import Thing
from src.iot.things.Duck import DuckController

logger = logging.getLogger("Duck")


class Duck(Thing):
    def __init__(self):
        super().__init__("Duck", "四足鸭子机器人控制")
        
        """初始化鸭子机器人控制器."""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        
        # 应用程序实例
        self.app = None
        
        # 鸭子控制器
        self.duck_controller = None
        self.is_initialized = False
        self.is_running = False
        self.photo_result = ""
        
        # 加载配置
        from src.utils.config_manager import ConfigManager
        self.config = ConfigManager.get_instance()
        
        # 初始化鸭子控制器
        self.duck_controller = DuckController.DuckController.get_instance()
        
        self.add_property_and_method()  # 定义设备方法与状态属性

    def add_property_and_method(self):
        """添加属性和方法定义"""
        # 定义属性
        self.add_property("initialized", "鸭子是否已初始化", lambda: self.is_initialized)
        self.add_property("running", "鸭子是否正在运行", lambda: self.is_running)
        self.add_property("photo_result", "拍照分析结果", lambda: self.photo_result)
        
        # 定义方法
        self.add_method(
            "initialize", 
            "初始化鸭子机器人", 
            [], 
            lambda params: self.initialize_duck()
        )
        
        self.add_method(
            "move_forward", 
            "向前移动2秒", 
            [], 
            lambda params: self.move_forward()
        )
        
        self.add_method(
            "move_backward", 
            "向后移动2秒", 
            [], 
            lambda params: self.move_backward()
        )
        
        self.add_method(
            "turn_left", 
            "左转2秒", 
            [], 
            lambda params: self.turn_left()
        )
        
        self.add_method(
            "turn_right", 
            "右转2秒", 
            [], 
            lambda params: self.turn_right()
        )
        
        self.add_method(
            "move_left", 
            "向左走2秒，当指令为与左走有关时，调用该函数", 
            [], 
            lambda params: self.move_left()
        )
        
        self.add_method(
            "move_right", 
            "向右走2秒", 
            [], 
            lambda params: self.move_right()
        )
        
        self.add_method(
            "move_antenna", 
            "动耳朵", 
            [], 
            lambda params: self.move_antenna()
        )
        
        self.add_method(
            "take_photo", 
            "拍照并分析内容", 
            [], 
            lambda params: self.take_photo()
        )
        
        self.add_method(
            "stop_movement", 
            "停止鸭子运动", 
            [], 
            lambda params: self.stop_movement()
        )

    def initialize_duck(self):
        """初始化鸭子机器人"""
        logger.info("开始初始化鸭子机器人...")
        
        result = self.duck_controller.initialize()
        
        if result["status"] == "success":
            self.is_initialized = True
            self.is_running = True
            logger.info("鸭子机器人初始化成功")
        else:
            logger.error(f"鸭子机器人初始化失败: {result['message']}")
        
        return result

    def move_forward(self):
        """向前移动"""
        if not self.is_initialized:
            return {"status": "error", "message": "鸭子未初始化，请先初始化"}
        
        logger.info("鸭子开始向前移动")
        result = self.duck_controller.move_forward()
        self._trigger_voice_feedback("鸭子开始向前移动")
        return result

    def move_backward(self):
        """向后移动"""
        if not self.is_initialized:
            return {"status": "error", "message": "鸭子未初始化，请先初始化"}
        
        logger.info("鸭子开始向后移动")
        result = self.duck_controller.move_backward()
        self._trigger_voice_feedback("鸭子开始向后移动")
        return result

    def turn_left(self):
        """左转"""
        if not self.is_initialized:
            return {"status": "error", "message": "鸭子未初始化，请先初始化"}
        
        logger.info("鸭子开始左转")
        result = self.duck_controller.turn_left()
        self._trigger_voice_feedback("鸭子开始左转")
        return result

    def turn_right(self):
        """右转"""
        if not self.is_initialized:
            return {"status": "error", "message": "鸭子未初始化，请先初始化"}
        
        logger.info("鸭子开始右转")
        result = self.duck_controller.turn_right()
        self._trigger_voice_feedback("鸭子开始右转")
        return result

    def move_left(self):
        """向左走"""
        if not self.is_initialized:
            return {"status": "error", "message": "鸭子未初始化，请先初始化"}
        
        logger.info("鸭子开始向左走")
        result = self.duck_controller._set_movement_command(vy=0.2, duration=2.0)
        self._trigger_voice_feedback("鸭子开始向左走")
        return {"status": "success", "message": "鸭子开始向左走2秒"}

    def move_right(self):
        """向右走"""
        if not self.is_initialized:
            return {"status": "error", "message": "鸭子未初始化，请先初始化"}
        
        logger.info("鸭子开始向右走")
        result = self.duck_controller._set_movement_command(vy=-0.2, duration=2.0)
        self._trigger_voice_feedback("鸭子开始向右走")
        return {"status": "success", "message": "鸭子开始向右走2秒"}

    def move_antenna(self):
        """动耳朵"""
        if not self.is_initialized:
            return {"status": "error", "message": "鸭子未初始化，请先初始化"}
        
        logger.info("鸭子开始动耳朵")
        result = self.duck_controller.move_antenna()
        self._trigger_voice_feedback("鸭子开始动耳朵")
        return result

    def take_photo(self):
        """拍照并分析"""
        if not self.is_initialized:
            return {"status": "error", "message": "鸭子未初始化，请先初始化"}
        
        logger.info("鸭子开始拍照...")
        result = self.duck_controller.take_photo()
        
        if result["status"] == "success":
            # 提取分析结果
            message = result.get("message", "")
            if "分析结果：" in message:
                self.photo_result = message.split("分析结果：")[1]
            else:
                self.photo_result = message
            
            logger.info(f"拍照完成，分析结果: {self.photo_result}")
            
            # 触发语音播报
            self._trigger_voice_feedback(f"拍照完成，{self.photo_result}")
        else:
            logger.error(f"拍照失败: {result['message']}")
        
        return result

    def stop_movement(self):
        """停止运动"""
        if not self.is_initialized:
            return {"status": "error", "message": "鸭子未初始化"}
        
        logger.info("停止鸭子运动")
        result = self.duck_controller.stop()
        self._trigger_voice_feedback("鸭子已停止运动")
        return result

    def _trigger_voice_feedback(self, message):
        """触发语音反馈"""
        try:
            # 获取应用程序实例
            self.app = Application.get_instance()
            
            # 设置设备状态并触发语音播报
            self.app.set_device_state(DeviceState.LISTENING)
            asyncio.create_task(self.app.protocol.send_wake_word_detected(message))
            
        except Exception as e:
            logger.error(f"触发语音反馈失败: {e}")

    def shutdown(self):
        """关闭鸭子控制器"""
        if self.duck_controller:
            self.duck_controller.shutdown()
        self.is_initialized = False
        self.is_running = False
        logger.info("鸭子控制器已关闭") 