# Duck机器人设备模块

这是py-xiaozhi语音助手中的Duck四足机器人控制模块，按照CameraVL的组织方式结构化。

## 文件结构

### 主要模块文件

- `Duck.py` - 主要设备接口类，类似于CameraVL中的Camera.py
- `DuckController.py` - 核心控制逻辑，类似于CameraVL中的VL.py
- `duck_device.py` - 原始的完整设备实现（备份）

### Open_Duck_Mini_Runtime依赖文件

从Open_Duck_Mini_Runtime_Rockchip复制的核心模块：

- `rustypot_position_hwi.py` - 硬件接口控制
- `onnx_infer.py` - ONNX模型推理
- `imu.py` / `raw_imu.py` - IMU传感器处理
- `poly_reference_motion.py` - 参考运动模式
- `feet_contacts.py` - 脚部接触传感器
- `rl_utils.py` - 强化学习工具
- `duck_config.py` - 配置文件处理
- `duck_camera.py` - 摄像头控制（重命名避免冲突）

### 配置和设置文件

- `duck_config.json` - 默认配置文件
- `openai_config_template.py` - OpenAI API配置模板
- `setup_duck_integration.py` - 环境设置脚本
- `DUCK_INTEGRATION_README.md` - 详细使用说明

## 使用方法

1. **运行设置脚本**：
   ```bash
   cd py-xiaozhi/src/iot/things/Duck
   python setup_duck_integration.py
   ```

2. **配置OpenAI API**（可选，用于拍照功能）：
   ```bash
   cp openai_config_template.py ../../../../openai_config.py
   # 编辑openai_config.py填入API密钥
   ```

3. **在py-xiaozhi中使用**：
   - 启动py-xiaozhi
   - 语音命令："初始化鸭子"
   - 控制命令："鸭子向前移动"、"鸭子拍照"等

## 架构说明

这个模块采用分层设计：

```
Duck.py (设备接口层)
    ↓
DuckController.py (控制逻辑层)
    ↓
Open_Duck_Mini_Runtime模块 (硬件抽象层)
    ↓
实际硬件 (电机、IMU、摄像头等)
```

## 依赖要求

- Python 3.9-3.12
- numpy
- opencv-python
- onnxruntime
- openai (可选)
- picamzero (树莓派摄像头)

## 硬件要求

- Open Duck Mini四足机器人
- 树莓派
- USB串口连接 (/dev/ttyACM0)
- I2C总线 (/dev/i2c-1)
- 摄像头模块（可选）

更多详细信息请参考 `DUCK_INTEGRATION_README.md` 