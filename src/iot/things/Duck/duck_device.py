import sys
import os
import time
import threading
import numpy as np
import base64
from pathlib import Path

from src.iot.thing import Thing, Parameter, ValueType

# 添加Open_Duck_Mini_Runtime路径
HOME_DIR = os.path.expanduser("~")
DUCK_RUNTIME_PATH = os.path.join(HOME_DIR, "Open_Duck_Mini_Runtime_Rockchip")
sys.path.append(DUCK_RUNTIME_PATH)

try:
    from mini_bdx_runtime.rustypot_position_hwi import HWI
    from mini_bdx_runtime.onnx_infer import OnnxInfer
    from mini_bdx_runtime.raw_imu import Imu
    from mini_bdx_runtime.poly_reference_motion import PolyReferenceMotion
    from mini_bdx_runtime.feet_contacts import FeetContacts
    from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter
    from mini_bdx_runtime.duck_config import DuckConfig
    from mini_bdx_runtime.camera import Cam
    DUCK_RUNTIME_AVAILABLE = True
except ImportError as e:
    print(f"[警告] Open_Duck_Mini_Runtime模块导入失败: {e}")
    DUCK_RUNTIME_AVAILABLE = False

# OpenAI API相关
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("[警告] OpenAI模块未安装，拍照功能将不可用")
    OPENAI_AVAILABLE = False


class Duck(Thing):
    def __init__(self):
        super().__init__("Duck", "一个可以移动和拍照的真实鸭子机器人")
        
        # 初始化标志
        self.initialized = False
        self.running = False
        self.control_thread = None
        
        # 运动控制参数
        self.current_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [vx, vy, vyaw, head_pitch, head_yaw, head_roll, antenna]
        self.command_lock = threading.Lock()
        
        # 硬件组件
        self.hwi = None
        self.policy = None
        self.imu = None
        self.feet_contacts = None
        self.camera = None
        
        # 控制参数
        self.control_freq = 50
        self.action_scale = 0.25
        self.num_dofs = 14
        self.init_pos = None
        self.motor_targets = None
        self.prev_motor_targets = None
        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        
        print("[真实设备] 鸭子设备开始初始化...")

        # 定义属性
        self.add_property("initialized", "鸭子是否已初始化", lambda: self.initialized)
        self.add_property("running", "鸭子是否正在运行", lambda: self.running)

        # 定义方法
        self.add_method(
            "Initialize", 
            "初始化鸭子机器人", 
            [], 
            lambda params: self._initialize()
        )
        
        self.add_method(
            "MoveForward", 
            "向前移动2秒", 
            [], 
            lambda params: self._move_forward()
        )
        
        self.add_method(
            "MoveBackward", 
            "向后移动2秒", 
            [], 
            lambda params: self._move_backward()
        )
        
        self.add_method(
            "TurnLeft", 
            "左转2秒", 
            [], 
            lambda params: self._turn_left()
        )
        
        self.add_method(
            "TurnRight", 
            "右转2秒", 
            [], 
            lambda params: self._turn_right()
        )
        
        self.add_method(
            "TakePhoto", 
            "拍照并分析内容", 
            [], 
            lambda params: self._take_photo()
        )
        
        self.add_method(
            "Stop", 
            "停止鸭子运动", 
            [], 
            lambda params: self._stop()
        )

    def _initialize(self):
        """初始化鸭子机器人"""
        if not DUCK_RUNTIME_AVAILABLE:
            return {"status": "error", "message": "Open_Duck_Mini_Runtime模块不可用"}
        
        try:
            print("[真实设备] 开始初始化鸭子硬件...")
            
            # 加载配置
            duck_config_path = os.path.join(HOME_DIR, "duck_config.json")
            if not os.path.exists(duck_config_path):
                # 使用示例配置
                example_config_path = os.path.join(DUCK_RUNTIME_PATH, "example_config.json")
                duck_config_path = example_config_path
            
            self.duck_config = DuckConfig(config_json_path=duck_config_path)
            
            # 查找ONNX模型文件
            onnx_model_path = self._find_onnx_model()
            if not onnx_model_path:
                return {"status": "error", "message": "未找到ONNX模型文件"}
            
            # 初始化策略模型
            self.policy = OnnxInfer(onnx_model_path, awd=True)
            print("[真实设备] ONNX模型加载完成")
            
            # 初始化硬件接口
            self.hwi = HWI(self.duck_config, "/dev/ttyACM0")
            self._start_motors()
            print("[真实设备] 电机控制初始化完成")
            
            # 初始化IMU
            self.imu = Imu(
                sampling_freq=int(self.control_freq),
                user_pitch_bias=0,
                upside_down=self.duck_config.imu_upside_down,
            )
            print("[真实设备] IMU初始化完成")
            
            # 初始化脚部接触传感器
            self.feet_contacts = FeetContacts()
            
            # 初始化摄像头（如果可用）
            if self.duck_config.camera:
                try:
                    self.camera = Cam()
                    print("[真实设备] 摄像头初始化完成")
                except Exception as e:
                    print(f"[警告] 摄像头初始化失败: {e}")
            
            # 初始化运动参数
            self.init_pos = list(self.hwi.init_pos.values())
            self.motor_targets = np.array(self.init_pos.copy())
            self.prev_motor_targets = np.array(self.init_pos.copy())
            
            # 初始化参考运动
            poly_coeff_path = os.path.join(DUCK_RUNTIME_PATH, "scripts", "polynomial_coefficients.pkl")
            if os.path.exists(poly_coeff_path):
                self.PRM = PolyReferenceMotion(poly_coeff_path)
                self.imitation_i = 0
                self.imitation_phase = np.array([0, 0])
            
            self.initialized = True
            
            # 启动控制循环
            self._start_control_loop()
            
            print("[真实设备] 鸭子机器人初始化完成！")
            return {"status": "success", "message": "鸭子机器人初始化成功"}
            
        except Exception as e:
            print(f"[错误] 鸭子初始化失败: {e}")
            return {"status": "error", "message": f"初始化失败: {str(e)}"}

    def _find_onnx_model(self):
        """查找ONNX模型文件"""
        possible_paths = [
            os.path.join(HOME_DIR, "BEST_WALK_ONNX_2.onnx"),
            os.path.join(DUCK_RUNTIME_PATH, "BEST_WALK_ONNX_2.onnx"),
            os.path.join(DUCK_RUNTIME_PATH, "scripts", "BEST_WALK_ONNX_2.onnx"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"[真实设备] 找到ONNX模型: {path}")
                return path
        
        print("[错误] 未找到ONNX模型文件，请下载BEST_WALK_ONNX_2.onnx")
        return None

    def _start_motors(self):
        """启动电机"""
        kps = [30] * 14
        kds = [0] * 14
        
        # 降低头部电机的刚度
        kps[5:9] = [8, 8, 8, 8]
        
        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()
        
        time.sleep(2)

    def _start_control_loop(self):
        """启动控制循环"""
        if self.control_thread and self.control_thread.is_alive():
            return
        
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def _control_loop(self):
        """主控制循环"""
        print("[真实设备] 启动控制循环")
        
        try:
            while self.running and self.initialized:
                start_time = time.time()
                
                # 获取观测数据
                obs = self._get_obs()
                if obs is None:
                    time.sleep(0.01)
                    continue
                
                # 更新相位
                if hasattr(self, 'PRM'):
                    self.imitation_i += 1.0
                    self.imitation_i = self.imitation_i % self.PRM.nb_steps_in_period
                    self.imitation_phase = np.array([
                        np.cos(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                        np.sin(self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi),
                    ])
                
                # 策略推理
                action = self.policy.infer(obs)
                
                # 更新动作历史
                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()
                
                # 计算电机目标
                self.motor_targets = self.init_pos + action * self.action_scale
                
                # 头部控制
                with self.command_lock:
                    head_motor_targets = self.current_command[3:] + self.motor_targets[5:9]
                self.motor_targets[5:9] = head_motor_targets
                
                self.prev_motor_targets = self.motor_targets.copy()
                
                # 发送电机命令
                action_dict = make_action_dict(self.motor_targets, list(self.hwi.joints.keys()))
                self.hwi.set_position_all(action_dict)
                
                # 控制频率
                elapsed = time.time() - start_time
                sleep_time = max(0, 1/self.control_freq - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"[错误] 控制循环异常: {e}")
        finally:
            print("[真实设备] 控制循环结束")

    def _get_obs(self):
        """获取观测数据"""
        try:
            # 获取IMU数据
            imu_data = self.imu.get_data()
            
            # 获取关节位置和速度
            dof_pos = self.hwi.get_present_positions(ignore=["left_antenna", "right_antenna"])
            dof_vel = self.hwi.get_present_velocities(ignore=["left_antenna", "right_antenna"])
            
            if dof_pos is None or dof_vel is None or len(dof_pos) != self.num_dofs or len(dof_vel) != self.num_dofs:
                return None
            
            # 获取当前命令
            with self.command_lock:
                cmds = self.current_command.copy()
            
            # 获取脚部接触
            feet_contacts = self.feet_contacts.get()
            
            # 组合观测
            obs = np.concatenate([
                imu_data["gyro"],
                imu_data["accelero"],
                cmds,
                dof_pos - self.init_pos,
                dof_vel * 0.05,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                feet_contacts,
                self.imitation_phase if hasattr(self, 'imitation_phase') else [0, 0],
            ])
            
            return obs
            
        except Exception as e:
            print(f"[错误] 获取观测数据失败: {e}")
            return None

    def _set_movement_command(self, vx=0.0, vy=0.0, vyaw=0.0, duration=2.0):
        """设置运动命令"""
        if not self.initialized:
            return {"status": "error", "message": "鸭子未初始化"}
        
        with self.command_lock:
            self.current_command = [vx, vy, vyaw, 0.0, 0.0, 0.0, 0.0]
        
        # 运动指定时间后停止
        def stop_after_duration():
            time.sleep(duration)
            with self.command_lock:
                self.current_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        threading.Thread(target=stop_after_duration, daemon=True).start()

    def _move_forward(self):
        """向前移动2秒"""
        print("[真实设备] 鸭子开始向前移动")
        self._set_movement_command(vx=0.15, duration=2.0)
        return {"status": "success", "message": "鸭子开始向前移动2秒"}

    def _move_backward(self):
        """向后移动2秒"""
        print("[真实设备] 鸭子开始向后移动")
        self._set_movement_command(vx=-0.15, duration=2.0)
        return {"status": "success", "message": "鸭子开始向后移动2秒"}

    def _turn_left(self):
        """左转2秒"""
        print("[真实设备] 鸭子开始左转")
        self._set_movement_command(vyaw=0.5, duration=2.0)
        return {"status": "success", "message": "鸭子开始左转2秒"}

    def _turn_right(self):
        """右转2秒"""
        print("[真实设备] 鸭子开始右转")
        self._set_movement_command(vyaw=-0.5, duration=2.0)
        return {"status": "success", "message": "鸭子开始右转2秒"}

    def _take_photo(self):
        """拍照并分析内容"""
        if not self.initialized:
            return {"status": "error", "message": "鸭子未初始化"}
        
        if not self.camera:
            return {"status": "error", "message": "摄像头不可用"}
        
        if not OPENAI_AVAILABLE:
            return {"status": "error", "message": "OpenAI模块不可用"}
        
        try:
            print("[真实设备] 鸭子开始拍照...")
            
            # 拍照并获取base64编码的图像
            encoded_image = self.camera.get_encoded_image()
            
            # 调用OpenAI Vision API
            response = self._analyze_image_with_openai(encoded_image)
            
            print(f"[真实设备] 图像分析结果: {response}")
            return {"status": "success", "message": f"拍照完成，分析结果：{response}"}
            
        except Exception as e:
            error_msg = f"拍照失败: {str(e)}"
            print(f"[错误] {error_msg}")
            return {"status": "error", "message": error_msg}

    def _analyze_image_with_openai(self, encoded_image):
        """使用OpenAI API分析图像"""
        try:
            # 尝试导入配置文件
            try:
                import sys
                config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "openai_config.py")
                sys.path.append(os.path.dirname(config_path))
                from openai_config import get_openai_config
                config = get_openai_config()
                
                if config["api_key"] == "your-openai-api-key-here":
                    return "请先配置OpenAI API密钥。请复制openai_config_template.py为openai_config.py并填入您的API密钥。"
                
            except ImportError:
                return "未找到openai_config.py配置文件。请复制openai_config_template.py为openai_config.py并配置您的API密钥。"
            
            # 设置OpenAI配置
            from openai import OpenAI
            client = OpenAI(
                api_key=config["api_key"],
                base_url=config["api_base"]
            )
            
            # 调用Vision API
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": config["prompt"]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=config["max_tokens"]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"图像分析失败: {str(e)}"

    def _stop(self):
        """停止鸭子运动"""
        print("[真实设备] 停止鸭子运动")
        
        with self.command_lock:
            self.current_command = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        return {"status": "success", "message": "鸭子已停止运动"}

    def __del__(self):
        """析构函数"""
        self.running = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=1) 