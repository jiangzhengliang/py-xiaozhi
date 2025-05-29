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

## 用户操作步骤和语音命令

### 第一步：启动鸭子机器人
用户必须先说以下命令来启动机器人：
- "启动鸭子机器人" → 调用 `Start` 方法（会自动先初始化）

**系统响应**：
- 检查硬件连接和必要文件
- 加载ONNX强化学习模型
- 初始化电机系统和传感器
- 启动50Hz控制循环
- 返回启动状态

### 第二步：执行移动命令
启动成功后，用户可以说以下移动命令：

| 语音命令 | 功能描述 | 持续时间 | Thing方法名 | 内部命令 |
|---------|---------|---------|------------|----------|
| 向前走 | MoveForward | 5秒 | MoveForward | forward |
| 向后走 | MoveBackward | 5秒 | MoveBackward | backward |
| 向左走 | MoveLeft | 5秒 | MoveLeft | left |
| 向右走 | MoveRight | 5秒 | MoveRight | right |
| 向左转弯 | TurnLeft | 5秒 | TurnLeft | turn_left |
| 向右转弯 | TurnRight | 5秒 | TurnRight | turn_right |

### 第三步：控制命令（可选）
在移动过程中，用户还可以说：
- "暂停鸭子机器人" → 暂停当前动作
- "恢复鸭子机器人" → 恢复动作执行

### 第四步：状态查询
用户可以随时查询机器人状态：
- "鸭子机器人状态" → 获取当前状态信息

## 完整的语音命令列表

### 启动命令
| 语音命令 | 功能描述 | Thing方法名 | 说明 |
|---------|---------|------------|------|
| 启动鸭子机器人 | 启动机器人（自动初始化并启动控制循环） | Start | 会自动先初始化再启动 |

### 移动控制命令
| 语音命令 | 功能描述 | 持续时间 | Thing方法名 | 内部命令 |
|---------|---------|---------|------------|----------|
| 向前走 | 鸭子机器人向前移动 | 5秒 | MoveForward | forward |
| 向后走 | 鸭子机器人向后移动 | 5秒 | MoveBackward | backward |
| 向左走 | 鸭子机器人向左移动 | 5秒 | MoveLeft | left |
| 向右走 | 鸭子机器人向右移动 | 5秒 | MoveRight | right |
| 向左转弯 | 鸭子机器人向左转弯 | 5秒 | TurnLeft | turn_left |
| 向右转弯 | 鸭子机器人向右转弯 | 5秒 | TurnRight | turn_right |

### 控制命令
| 语音命令 | 功能描述 | Thing方法名 | 说明 |
|---------|---------|------------|------|
| 暂停鸭子机器人 | 暂停当前动作 | Pause | 暂停移动但不停止控制循环 |
| 恢复鸭子机器人 | 恢复动作执行 | Resume | 恢复被暂停的动作 |

### 状态查询命令
| 语音命令 | 功能描述 | Thing方法名 | 返回信息 |
|---------|---------|------------|----------|
| 鸭子机器人状态 | 获取当前状态信息 | GetStatus | 初始化状态、运行状态、硬件可用性等 |

## 语音命令映射到控制命令

| 语音命令 | 控制命令 | 线性速度X | 线性速度Y | 角速度 |
|---------|---------|-----------|-----------|--------|
| 向前走 | forward | 0.5 | 0.0 | 0.0 |
| 向后走 | backward | -0.5 | 0.0 | 0.0 |
| 向左走 | left | 0.0 | 0.5 | 0.0 |
| 向右走 | right | 0.0 | -0.5 | 0.0 |
| 向左转弯 | turn_left | 0.0 | 0.0 | 0.5 |
| 向右转弯 | turn_right | 0.0 | 0.0 | -0.5 |

**注意事项**：
- 如果用户直接说"向前走"而没有先启动，系统会自动先执行启动
- 每个移动命令执行5秒后自动停止
- 新的移动命令会覆盖正在执行的命令
- 停止鸭子机器人后，再次启动不需要重新初始化，系统会直接启动控制循环
- **自动恢复机制**：启动后机器人默认处于暂停状态，但说移动命令时会自动恢复，无需手动说"恢复鸭子机器人"

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

### 用户操作流程
1. **启动阶段**：用户说"启动鸭子机器人"
2. **移动控制阶段**：用户说移动命令（如"向前走"、"向左转弯"等）
3. **可选控制阶段**：用户说控制命令（如"暂停"、"恢复"、"停止"）
4. **状态查询阶段**：用户说"鸭子机器人状态"查询当前状态

### 语音控制流程
1. 用户发出语音命令（如"向前走"）
2. py-xiaozhi识别语音并调用对应的Thing方法（MoveForward）
3. Duck类自动确保机器人已启动并运行（如果未启动会自动启动）
4. 强制切换为非头部控制模式
5. 设置语音命令并开始5秒计时
6. 控制循环执行相应的移动命令
7. 5秒后自动停止命令

### 自动启动机制
- 如果用户直接说移动命令而没有先启动，系统会自动调用 `_ensure_running()` 方法
- 该方法会检查启动状态，如果未启动会自动执行启动
- 确保用户无需记住必须先启动，可以直接说移动命令

### 优先级和状态管理
- **语音命令优先**: 新的语音命令会覆盖之前的命令
- **自动初始化**: 语音控制时自动初始化和启动控制循环
- **强制解除暂停**: 语音控制时强制切换到非头部控制模式
- **超时保护**: 语音命令自动超时，防止意外持续执行

## Thing设备注册

系统自动注册为IoT设备，支持以下方法：

- `Start`: 启动机器人（自动初始化并启动控制循环）
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