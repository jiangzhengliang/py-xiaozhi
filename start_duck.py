#!/usr/bin/env python3
"""
鸭子机器人启动脚本
先执行硬件权限设置，再启动py-xiaozhi的CLI模式
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """执行命令并显示结果"""
    print(f"\n=== {description} ===")
    print(f"执行命令: {command}")
    
    try:
        # 执行命令
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # 显示输出
        if result.stdout:
            print("输出:")
            print(result.stdout)
        
        if result.stderr:
            print("错误:")
            print(result.stderr)
        
        # 检查返回码
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            return True
        else:
            print(f"❌ {description} 失败 (返回码: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {description} 执行异常: {e}")
        return False

def main():
    """主函数"""
    print("🦆 鸭子机器人启动脚本")
    print("=" * 50)
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"当前目录: {current_dir}")
    
    # 检查必要文件
    required_files = [
        "setup_hardware_permissions.py",
        "main.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 缺少必要文件: {missing_files}")
        print("请确保在py-xiaozhi-rkduck目录下运行此脚本")
        return False
    
    print("✅ 所有必要文件已找到")
    
    # 步骤1: 执行硬件权限设置
    print("\n" + "=" * 50)
    print("步骤1: 设置硬件权限")
    print("=" * 50)
    
    # 检查是否已经有sudo权限
    if os.geteuid() == 0:
        print("当前已有root权限，直接执行权限设置")
        success1 = run_command("python3 setup_hardware_permissions.py", "硬件权限设置")
    else:
        print("需要sudo权限执行硬件设置")
        success1 = run_command("sudo python3 setup_hardware_permissions.py", "硬件权限设置")
    
    if not success1:
        print("❌ 硬件权限设置失败，是否继续启动py-xiaozhi? (y/n)")
        response = input().lower().strip()
        if response != 'y':
            print("用户选择退出")
            return False
    
    # 步骤2: 启动py-xiaozhi CLI模式
    print("\n" + "=" * 50)
    print("步骤2: 启动py-xiaozhi CLI模式")
    print("=" * 50)
    
    # 检查conda环境
    print("检查conda环境...")
    if os.environ.get("CONDA_DEFAULT_ENV") != "py-xiaozhi":
        print("❌ 当前未在py-xiaozhi虚拟环境下，请先运行 'conda activate py-xiaozhi' 后再执行本脚本。")
        return False
    print("✅ 已在py-xiaozhi环境，直接启动...")
    success2 = run_command("python3 main.py --mode cli", "启动py-xiaozhi CLI模式")
    
    if not success2:
        print("❌ py-xiaozhi启动失败")
        return False
    
    print("\n✅ 鸭子机器人启动脚本执行完成")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 启动成功！现在可以使用语音命令控制鸭子机器人了")
            print("示例命令:")
            print("  - '启动鸭子机器人'")
            print("  - '向前走'")
            print("  - '向左转弯'")
            print("  - '鸭子机器人状态'")
        else:
            print("\n💥 启动失败，请检查错误信息")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断启动")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 启动脚本异常: {e}")
        sys.exit(1) 