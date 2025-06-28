# Duck机器人集成使用说明

本文档说明如何将py-xiaozhi语音助手与Open_Duck_Mini_Runtime_Rockchip四足机器人集成使用。

## 功能特性

- **语音控制机器人运动**：通过语音命令控制鸭子机器人的移动
- **智能拍照分析**：机器人可以拍照并使用AI分析图像内容
- **实时运动控制**：基于强化学习的自然步态控制
- **多种运动模式**：支持前进、后退、左转、右转

## 前置条件

### 硬件要求
- Open Duck Mini四足机器人
- 树莓派(推荐Zero 2W或4B)
- 摄像头模块(可选，用于拍照功能)
- 麦克风和扬声器(用于语音交互)

### 软件要求
- Python 3.9-3.12
- py-xiaozhi语音助手
- Open_Duck_Mini_Runtime_Rockchip运行时

## 安装步骤

### 1. 环境设置

首先运行设置脚本：

```bash
cd py-xiaozhi
python setup_duck_integration.py
```

这个脚本会：
- 检查必要的文件和依赖
- 安装Python包
- 创建配置文件
- 检查硬件连接

### 2. 下载ONNX模型

下载机器人运动控制模型：

```bash
# 下载到home目录
cd ~
wget https://github.com/apirrone/Open_Duck_Mini/raw/v2/BEST_WALK_ONNX_2.onnx
```

或者将文件手动下载并放置到以下任一位置：
- `~/BEST_WALK_ONNX_2.onnx`
- `~/Open_Duck_Mini_Runtime_Rockchip/BEST_WALK_ONNX_2.onnx`
- `~/Open_Duck_Mini_Runtime_Rockchip/scripts/BEST_WALK_ONNX_2.onnx`

### 3. 配置OpenAI API (拍照功能)

如果要使用拍照分析功能：

1. 复制配置模板：
```bash
cp openai_config_template.py openai_config.py
```

2. 编辑`openai_config.py`，填入您的OpenAI API密钥：
```python
OPENAI_API_KEY = "sk-your-actual-api-key-here"
```

### 4. 硬件连接检查

确保以下硬件正确连接：
- USB串口连接到机器人控制板(`/dev/ttyACM0`)
- I2C连接到IMU传感器(`/dev/i2c-1`)
- 摄像头模块(如果使用拍照功能)

## 使用方法

### 1. 启动py-xiaozhi

```bash
cd py-xiaozhi
python main.py
```

### 2. 初始化鸭子机器人

对小智说：
- "初始化鸭子"
- "启动鸭子机器人"

初始化成功后，机器人会开始运行控制循环。

### 3. 控制机器人运动

可以使用以下语音命令：

#### 运动控制
- "鸭子向前移动" / "让鸭子向前走"
- "鸭子向后移动" / "让鸭子后退"
- "鸭子左转" / "让鸭子向左转"
- "鸭子右转" / "让鸭子向右转"
- "停止鸭子" / "让鸭子停下"

#### 拍照功能
- "鸭子拍照" / "让鸭子拍张照片"
- "鸭子看看周围" / "用鸭子的眼睛看看"

### 4. 运动参数

每次运动命令会让机器人执行2秒的动作：
- **前进/后退**：速度约0.15 m/s
- **左转/右转**：角速度约0.5 rad/s

## 配置文件说明

### duck_config.json

机器人硬件配置文件，位于`~/duck_config.json`：

```json
{
  "start_paused": false,
  "imu_upside_down": false,
  "expression_features": {
    "camera": true,
    "eyes": false,
    "projector": false,
    "antennas": false,
    "speaker": false,
    "microphone": false
  },
  "joints_offsets": {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    // ... 其他关节偏移
  }
}
```

### openai_config.py

OpenAI API配置文件：

```python
OPENAI_API_KEY = "your-api-key"
VISION_MODEL = "gpt-4-vision-preview"
MAX_TOKENS = 300
IMAGE_ANALYSIS_PROMPT = "请详细描述这张图片中看到的内容"
```

## 故障排除

### 常见问题

1. **"Open_Duck_Mini_Runtime模块不可用"**
   - 检查Open_Duck_Mini_Runtime_Rockchip目录是否存在
   - 确保路径正确：`~/Open_Duck_Mini_Runtime_Rockchip`

2. **"未找到ONNX模型文件"**
   - 下载BEST_WALK_ONNX_2.onnx模型文件
   - 确保文件放在正确位置

3. **"串口设备不存在"**
   - 检查USB连接
   - 确认设备权限：`sudo chmod 666 /dev/ttyACM0`

4. **"IMU初始化失败"**
   - 启用I2C：`sudo raspi-config` -> Interface Options -> I2C
   - 检查I2C连接

5. **"摄像头不可用"**
   - 确保摄像头模块已连接
   - 安装picamzero：`pip install picamzero`
   - 启用摄像头：`sudo raspi-config` -> Interface Options -> Camera

### 日志查看

机器人运行时会输出详细日志：
- `[真实设备]`：来自鸭子设备的日志
- `[错误]`：错误信息
- `[警告]`：警告信息

### 性能优化

1. **控制频率调整**
   - 默认50Hz，可在代码中调整`self.control_freq`
   - 降低频率可减少CPU负载

2. **动作平滑**
   - 可启用低通滤波器来平滑动作
   - 在初始化时设置`cutoff_frequency`参数

## 扩展功能

### 添加新的运动模式

可以在`Duck`类中添加新的运动方法：

```python
def _custom_movement(self):
    """自定义运动"""
    # 设置运动参数 [vx, vy, vyaw, head_pitch, head_yaw, head_roll, antenna]
    self._set_movement_command(vx=0.1, vy=0.1, vyaw=0.2, duration=3.0)
    return {"status": "success", "message": "执行自定义运动"}
```

### 添加表情控制

如果机器人配备了表情组件：

```python
# 在配置中启用
"expression_features": {
    "eyes": true,
    "projector": true,
    "antennas": true
}
```

## 技术架构

```
py-xiaozhi (语音助手)
    ↓
Duck IoT设备 (运动控制接口)
    ↓
Open_Duck_Mini_Runtime_Rockchip
    ├── 电机控制 (HWI)
    ├── IMU传感器 (Imu)
    ├── 强化学习策略 (OnnxInfer)
    ├── 摄像头 (Cam)
    └── 其他传感器
```

## 贡献

欢迎提交问题和改进建议：
- py-xiaozhi：https://github.com/huangjunsen0406/py-xiaozhi
- Open_Duck_Mini_Runtime：https://github.com/apirrone/Open_Duck_Mini_Runtime

## 许可证

本集成代码遵循MIT许可证。 