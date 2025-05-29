#!/bin/bash

echo "🦆 鸭子机器人启动脚本"
echo "=================================================="

# 检查当前目录
echo "当前目录: $(pwd)"

# 检查必要文件
if [ ! -f "setup_hardware_permissions.py" ]; then
    echo "❌ 缺少 setup_hardware_permissions.py"
    exit 1
fi

if [ ! -f "main.py" ]; then
    echo "❌ 缺少 main.py"
    exit 1
fi

echo "✅ 所有必要文件已找到"

# 步骤1: 执行硬件权限设置
echo ""
echo "=================================================="
echo "步骤1: 设置硬件权限"
echo "=================================================="

echo "执行命令: sudo python3 setup_hardware_permissions.py"
if sudo python3 setup_hardware_permissions.py; then
    echo "✅ 硬件权限设置成功"
else
    echo "❌ 硬件权限设置失败"
    echo "是否继续启动py-xiaozhi? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "用户选择退出"
        exit 1
    fi
fi

# 步骤2: 启动py-xiaozhi CLI模式
echo ""
echo "=================================================="
echo "步骤2: 启动py-xiaozhi CLI模式"
echo "=================================================="

# 检查conda环境
echo "检查conda环境..."
if [ "$CONDA_DEFAULT_ENV" != "py-xiaozhi" ]; then
    echo "❌ 当前未在py-xiaozhi虚拟环境下，请先运行 'conda activate py-xiaozhi' 后再执行本脚本。"
    exit 1
fi
echo "✅ 已在py-xiaozhi环境，直接启动..."
python3 main.py --mode cli 