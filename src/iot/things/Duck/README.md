# 鸭子机器人控制系统

## 概述

这个系统实现了基于强化学习的14自由度鸭子机器人控制，集成到py-xiaozhi语音智能体中，支持语音控制和Xbox手柄控制。

## 功能特性

- **语音控制**: 支持向前/后/左/右移动、左右转弯等语音命令
- **Xbox手柄控制**: 支持Xbox手柄实时控制，手柄输入优先于语音命令
- **强化学习控制**: 基于ONNX模型的50Hz实时控制循环
- **表情控制**: 支持眼睛、天线、投影仪等表情组件
- **硬件适配**: 支持真实硬件和模拟模式，硬件不可用时自动降级
- **按键监控**: 监控Xbox手柄按键状态，支持切换控制模式

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

| 语音命令 | 功能描述 | 持续时间 |
|---------|---------|---------|
| 向前走 | MoveForward | 5秒 |
| 向后走 | MoveBackward | 5秒 |
| 向左走 | MoveLeft | 5秒 |
| 向右走 | MoveRight | 5秒 |
| 向左转弯 | TurnLeft | 5秒 |
| 向右转弯 | TurnRight | 5秒 |

## 控制优先级

1. **Xbox手柄优先**: 当检测到手柄输入时，优先执行手柄命令
2. **语音命令**: 在没有手柄输入时执行语音命令
3. **自动切换**: 语音控制时强制切换为非头部控制模式

## 配置文件

### duck_config.json
```json
{
    "start_paused": false,
    "imu_upside_down": false,
    "phase_frequency_factor_offset": 0.0,
    "expression_features": {
        "eyes": false,
        "projector": false,
        "antennas": false,
        "speaker": false,
        "microphone": false,
        "camera": false
    },
    "joints_offsets": {
        // 14个关节的偏移量配置
    }
}
```

## 运行模式

### 硬件模式
- 需要连接真实的鸭子机器人硬件
- 支持所有功能，包括电机控制、传感器读取、表情控制

### 模拟模式
- 在硬件不可用时自动启用
- 模拟命令执行，用于测试和开发
- 保持语音交互功能

## 安全特性

- **错误处理**: 完善的异常处理，防止硬件故障影响系统
- **优雅降级**: 硬件不可用时自动切换到模拟模式
- **超时保护**: 语音命令自动超时，防止意外持续执行

## IoT设备注册

系统自动注册为IoT设备，支持以下方法：

- `Initialize`: 初始化机器人
- `Start`: 启动控制循环
- `Stop`: 停止控制循环
- `MoveForward/Backward/Left/Right`: 语音移动控制
- `TurnLeft/TurnRight`: 语音转向控制
- `SetCommand`: 手动设置控制命令
- `GetStatus`: 获取机器人状态

## 依赖要求

```
onnxruntime
adafruit-circuitpython-bno055
adafruit-extended-bus
periphery
rustypot
numpy
scipy
pygame
```

## 使用示例

### 启动语音控制
只需对小智说："向前走"、"向左转弯"等命令，机器人会自动执行相应动作。

### Xbox手柄控制
连接Xbox手柄后，可以直接使用摇杆控制机器人移动，手柄输入会覆盖语音命令。

### 状态查询
可以询问小智："鸭子机器人的状态如何？"来获取当前运行状态。 