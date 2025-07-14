# 鸭子机器人控制系统

## 概述

这个系统实现了基于强化学习的14自由度鸭子机器人控制，集成到py-xiaozhi语音智能体中，支持语音控制。

## 功能特性

- **语音控制**: 支持向前/后/左/右移动、左右转弯等语音命令（已移除Xbox手柄支持）
- **强化学习控制**: 基于ONNX模型的50Hz实时控制循环
- **表情控制**: 支持眼睛、天线、投影仪等表情组件
- **硬件适配**: 支持真实硬件和模拟模式，硬件不可用时自动降级
- **IoT设备集成**: 作为Thing设备注册到py-xiaozhi智能体系统

## 硬件组件

### 14自由度电机
- **腿部**: 左右各5个关节（hip_yaw, hip_roll, hip_pitch, knee, ankle）
- **头部**: 4个关节（neck_pitch, head_pitch, head_yaw, head_roll）

### 传感器系统
- **IMU传感器**: BNO055，提供姿态反馈
- **足部接触传感器**: 检测脚部接触地面状态

### 表情控制
- **眼睛**: GPIO控制LED眨眼
- **天线**: PWM控制伺服电机
- **投影仪**: 显示表情图案

## 语音控制命令

| 语音命令 | 功能描述 | 持续时间 | Thing方法名 |
|---------|---------|---------|------------|
| 向前走 | MoveForward | 5秒 | MoveForward |
| 向后走 | MoveBackward | 5秒 | MoveBackward |
| 向左走 | MoveLeft | 5秒 | MoveLeft |
| 向右走 | MoveRight | 5秒 | MoveRight |
| 向左转弯 | TurnLeft | 5秒 | TurnLeft |
| 向右转弯 | TurnRight | 5秒 | TurnRight |

## 安装和配置

### 1. 环境要求
```bash
# 激活conda环境
conda activate py-xiaozhi

# 安装依赖（如果需要）
pip install onnxruntime adafruit-circuitpython-bno055 adafruit-extended-bus periphery rustypot numpy scipy
```

### 2. 配置文件
确保以下文件存在于 `src/iot/things/Duck/` 目录：
- `duck_config.json` - 鸭子配置文件
- `BEST_WALK_ONNX_2.onnx` - 强化学习模型
- `polynomial_coefficients.pkl` - 步态参考数据

### 3. 配置示例 (duck_config.json)
```json
{
    "start_paused": false,
    "imu_upside_down": false,
    "phase_frequency_factor_offset": 0.0,
    "expression_features": {
        "eyes": true,
        "projector": true,
        "antennas": true,
        "speaker": false,
        "microphone": false,
        "camera": false
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
```

## 使用方法

### 1. 作为IoT设备使用
鸭子机器人会自动注册为IoT设备到py-xiaozhi系统中，用户可以通过语音命令控制：

```
"向前走"  -> 鸭子机器人向前移动5秒
"向左转弯" -> 鸭子机器人向左转弯5秒
```

### 2. 编程使用
```python
from src.iot.things.Duck.Duck import Duck

# 创建鸭子实例
duck = Duck()

# 初始化（自动进行）
result = duck._initialize()

# 控制命令
duck._move_forward()    # 向前走
duck._turn_left()       # 左转
duck._pause()           # 暂停
duck._resume()          # 恢复
```

### 3. 测试功能
```bash
# 基本集成测试
python test_duck_integration.py

# 语音控制功能测试
python test_duck_voice_control.py
```

## 运行模式

### 硬件模式
- 需要连接真实的鸭子机器人硬件
- 支持所有功能，包括电机控制、传感器读取、表情控制

### 模拟模式
- 在硬件不可用时自动启用
- 模拟命令执行，用于测试和开发
- 保持语音交互功能

## 控制逻辑

### 语音控制流程
1. 用户发出语音命令（如"向前走"）
2. py-xiaozhi识别语音并调用对应的Thing方法（MoveForward）
3. Duck类自动确保机器人已初始化并运行
4. 强制切换为非头部控制模式
5. 设置语音命令并开始5秒计时
6. 控制循环执行相应的移动命令
7. 5秒后自动停止命令

### 优先级和状态管理
- **语音命令优先**: 新的语音命令会覆盖之前的命令
- **自动初始化**: 语音控制时自动初始化和启动控制循环
- **强制解除暂停**: 语音控制时强制切换到非头部控制模式
- **超时保护**: 语音命令自动超时，防止意外持续执行

## Thing设备注册

系统自动注册为IoT设备，支持以下方法：

- `Initialize`: 初始化机器人
- `Start`: 启动控制循环
- `Stop`: 停止控制循环
- `MoveForward/Backward/Left/Right`: 语音移动控制
- `TurnLeft/TurnRight`: 语音转向控制
- `Pause/Resume`: 暂停和恢复控制
- `GetStatus`: 获取机器人状态

## 安全特性

- **错误处理**: 完善的异常处理，防止硬件故障影响系统
- **优雅降级**: 硬件不可用时自动切换到模拟模式
- **超时保护**: 语音命令自动超时，防止意外持续执行
- **状态监控**: 实时状态跟踪和错误报告

## 故障排除

### 常见问题

1. **ONNX模型维度错误**
   - 检查模型文件是否正确
   - 确认输入观测数据维度匹配
   - 可以使用模拟模式进行测试

2. **硬件连接问题**
   - 检查串口设备 `/dev/ttyACM0` 是否可用
   - 确认I2C设备权限
   - 查看设备连接状态

3. **语音命令无响应**
   - 确认Duck设备已在IoT系统中注册
   - 检查语音识别是否正确
   - 查看控制循环是否启动

### 调试模式
```python
# 查看详细状态
duck = Duck()
status = duck._get_status()
print(status)

# 检查硬件可用性
print(f"硬件可用: {duck.hardware_available}")
```

## 依赖要求

```
onnxruntime
adafruit-circuitpython-bno055
adafruit-extended-bus
periphery
rustypot
numpy
scipy
threading
pickle
```

## 文件结构

```
src/iot/things/Duck/
├── Duck.py                     # 主要的Duck类实现
├── v2_rl_walk_mujoco.py       # 强化学习控制器
├── duck_config.py             # 配置管理
├── duck_config.json           # 配置文件
├── BEST_WALK_ONNX_2.onnx     # ONNX模型
├── polynomial_coefficients.pkl # 步态参考数据
├── rustypot_position_hwi.py   # 硬件接口
├── onnx_infer.py              # ONNX推理
├── raw_imu.py                 # IMU传感器
├── eyes.py                    # 眼部控制
├── antennas.py                # 天线控制
├── projector.py               # 投影仪控制
├── feet_contacts.py           # 足部接触
├── rl_utils.py                # 强化学习工具
├── poly_reference_motion.py   # 参考运动
└── README.md                  # 本文档
```

## 版本历史

- **v2.0**: 集成到py-xiaozhi，支持语音控制
- **v1.0**: 原始Xbox手柄控制版本（已移除） 