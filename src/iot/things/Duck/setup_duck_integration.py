#!/usr/bin/env python3
"""
Duck机器人集成设置脚本
该脚本帮助设置py-xiaozhi与Open_Duck_Mini_Runtime_Rockchip的集成环境
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_status(message, is_error=False):
    """打印状态信息"""
    prefix = "[错误]" if is_error else "[信息]"
    print(f"{prefix} {message}")

def check_duck_runtime():
    """检查Open_Duck_Mini_Runtime_Rockchip是否存在"""
    home_dir = os.path.expanduser("~")
    duck_path = os.path.join(home_dir, "Open_Duck_Mini_Runtime_Rockchip")
    
    if os.path.exists(duck_path):
        print_status(f"找到Open_Duck_Mini_Runtime_Rockchip: {duck_path}")
        return True
    else:
        print_status(f"未找到Open_Duck_Mini_Runtime_Rockchip，路径: {duck_path}", True)
        return False

def check_onnx_model():
    """检查ONNX模型文件是否存在"""
    home_dir = os.path.expanduser("~")
    possible_paths = [
        os.path.join(home_dir, "BEST_WALK_ONNX_2.onnx"),
        os.path.join(home_dir, "Open_Duck_Mini_Runtime_Rockchip", "BEST_WALK_ONNX_2.onnx"),
        os.path.join(home_dir, "Open_Duck_Mini_Runtime_Rockchip", "scripts", "BEST_WALK_ONNX_2.onnx"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print_status(f"找到ONNX模型: {path}")
            return True
    
    print_status("未找到ONNX模型文件 BEST_WALK_ONNX_2.onnx", True)
    print_status("请从以下链接下载模型文件:")
    print_status("https://github.com/apirrone/Open_Duck_Mini/blob/v2/BEST_WALK_ONNX_2.onnx")
    print_status(f"并放置到以下任意位置:")
    for path in possible_paths:
        print_status(f"  - {path}")
    return False

def check_duck_config():
    """检查或创建鸭子配置文件"""
    home_dir = os.path.expanduser("~")
    config_path = os.path.join(home_dir, "duck_config.json")
    
    if os.path.exists(config_path):
        print_status(f"找到鸭子配置文件: {config_path}")
        return True
    
    # 创建默认配置
    default_config = {
        "start_paused": False,
        "imu_upside_down": False,
        "phase_frequency_factor_offset": 0.0,
        "expression_features": {
            "eyes": False,
            "projector": False,
            "antennas": False,
            "speaker": False,
            "microphone": False,
            "camera": True  # 启用摄像头功能
        },
        "joints_offsets": {
            "left_hip_yaw": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_pitch": 0.0,
            "left_knee": 0.0,
            "left_ankle": 0.0,
            "neck_pitch": 0.0,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "right_hip_yaw": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_pitch": 0.0,
            "right_knee": 0.0,
            "right_ankle": 0.0
        }
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        print_status(f"创建默认鸭子配置文件: {config_path}")
        return True
    except Exception as e:
        print_status(f"创建配置文件失败: {e}", True)
        return False

def install_dependencies():
    """安装必要的依赖包"""
    dependencies = [
        "numpy",
        "opencv-python", 
        "onnxruntime",
        "openai"
    ]
    
    print_status("开始安装Python依赖包...")
    
    for dep in dependencies:
        try:
            print_status(f"安装 {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
            print_status(f"✓ {dep} 安装成功")
        except subprocess.CalledProcessError as e:
            print_status(f"✗ {dep} 安装失败: {e}", True)
            return False
    
    return True

def check_hardware():
    """检查硬件连接"""
    print_status("检查硬件连接...")
    
    # 检查串口设备
    serial_devices = ["/dev/ttyACM0", "/dev/ttyUSB0"]
    found_serial = False
    
    for device in serial_devices:
        if os.path.exists(device):
            print_status(f"找到串口设备: {device}")
            found_serial = True
            break
    
    if not found_serial:
        print_status("未找到串口设备，请检查鸭子机器人是否正确连接", True)
        return False
    
    # 检查I2C设备（IMU）
    i2c_device = "/dev/i2c-1"
    if os.path.exists(i2c_device):
        print_status(f"找到I2C设备: {i2c_device}")
    else:
        print_status("未找到I2C设备，请检查I2C是否启用", True)
        print_status("可以运行 'sudo raspi-config' 启用I2C")
    
    return True

def setup_openai_config():
    """设置OpenAI API配置"""
    print_status("设置OpenAI API配置...")
    print_status("要使用拍照分析功能，需要配置OpenAI API密钥")
    print_status("请编辑 py-xiaozhi/src/iot/things/duck.py 文件")
    print_status("在 _analyze_image_with_openai 方法中设置您的API密钥")
    print_status("取消注释相关代码并替换 'your-api-key-here' 为您的实际API密钥")

def main():
    """主函数"""
    print_status("=== Duck机器人集成设置 ===")
    
    success = True
    
    # 检查基本环境
    if not check_duck_runtime():
        success = False
    
    if not check_onnx_model():
        success = False
    
    if not check_duck_config():
        success = False
    
    # 安装依赖
    if not install_dependencies():
        success = False
    
    # 检查硬件
    if not check_hardware():
        print_status("硬件检查未完全通过，但这在某些环境下是正常的")
    
    # OpenAI配置提示
    setup_openai_config()
    
    print_status("=== 设置完成 ===")
    
    if success:
        print_status("✓ 基本设置完成！")
        print_status("现在可以在py-xiaozhi中使用鸭子设备了")
        print_status("使用步骤:")
        print_status("1. 启动py-xiaozhi")
        print_status("2. 对小智说: '初始化鸭子'")
        print_status("3. 然后可以说: '鸭子向前移动'、'鸭子拍照' 等命令")
    else:
        print_status("✗ 设置过程中遇到问题，请检查上述错误信息", True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 