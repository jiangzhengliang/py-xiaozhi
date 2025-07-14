#!/usr/bin/env python3
"""
Open Duck Mini 硬件权限设置脚本
================================

简洁版本，只处理必要的硬件设备：
- GPIO 设备 (LED、投影仪、脚部传感器)
- PWM 设备 (天线控制)
- I2C 设备 (IMU传感器)
"""

import os
import sys
import glob
import time
import subprocess

def check_root():
    """检查是否有root权限"""
    if os.geteuid() != 0:
        print("❌ 错误：此脚本必须使用 sudo 权限运行！")
        print("用法: sudo python3 setup_hardware_permissions.py")
        sys.exit(1)
    print("✓ 检测到root权限")

def setup_gpio():
    """设置GPIO设备权限"""
    print("\n📍 配置 GPIO 设备...")
    
    # 项目需要的GPIO引脚配置
    gpio_configs = [
        {'pin': 133, 'name': 'LEFT_EYE_GPIO', 'direction': 'out'},
        {'pin': 119, 'name': 'RIGHT_EYE_GPIO', 'direction': 'out'},
        {'pin': 125, 'name': 'PROJECTOR_GPIO', 'direction': 'out'},
        {'pin': 124, 'name': 'INPUT_GPIO_1', 'direction': 'in'},
        {'pin': 138, 'name': 'INPUT_GPIO_2', 'direction': 'in'},
    ]
    
    # 设置GPIO sysfs接口权限
    if os.path.exists('/sys/class/gpio/export'):
        try:
            os.chmod('/sys/class/gpio/export', 0o666)
            print("  ✓ 设置权限: /sys/class/gpio/export")
        except OSError as e:
            print(f"  ✗ 设置权限失败 /sys/class/gpio/export: {e}")
    
    if os.path.exists('/sys/class/gpio/unexport'):
        try:
            os.chmod('/sys/class/gpio/unexport', 0o666)
            print("  ✓ 设置权限: /sys/class/gpio/unexport")
        except OSError as e:
            print(f"  ✗ 设置权限失败 /sys/class/gpio/unexport: {e}")
    
    # 导出和配置项目需要的GPIO引脚
    for config in gpio_configs:
        pin = config['pin']
        name = config['name']
        direction = config['direction']
        gpio_path = f"/sys/class/gpio/gpio{pin}"
        
        print(f"  配置 GPIO {pin} ({name})...")
        
        # 导出GPIO（如果未导出）
        if not os.path.exists(gpio_path):
            try:
                # 使用subprocess来模拟echo命令
                subprocess.run(['bash', '-c', f'echo {pin} > /sys/class/gpio/export'], 
                             check=True, capture_output=True)
                time.sleep(0.2)  # 等待导出完成
                print(f"    ✓ 导出 GPIO {pin}")
            except subprocess.CalledProcessError as e:
                print(f"    ✗ 导出 GPIO {pin} 失败: {e}")
                continue
        else:
            print(f"    ✓ GPIO {pin} 已导出")
        
        # 设置GPIO文件权限
        if os.path.exists(gpio_path):
            control_files = ['direction', 'value', 'active_low', 'edge']
            for file_name in control_files:
                file_path = f"{gpio_path}/{file_name}"
                if os.path.exists(file_path):
                    try:
                        os.chmod(file_path, 0o666)
                        print(f"    ✓ 设置权限: {file_name}")
                    except OSError as e:
                        print(f"    ✗ 设置权限失败 {file_name}: {e}")
            
            # 设置GPIO方向
            try:
                subprocess.run(['bash', '-c', f'echo {direction} > {gpio_path}/direction'], 
                             check=True, capture_output=True)
                print(f"    ✓ 设置方向: {direction}")
            except subprocess.CalledProcessError as e:
                print(f"    ✗ 设置方向失败: {e}")
            
            print(f"    🎉 GPIO {pin} ({name}) 配置完成")

def setup_pwm():
    """设置PWM设备权限"""
    print("\n📍 配置 PWM 设备...")
    
    # PWM配置 - 只配置项目需要的pwmchip2和pwmchip3
    pwm_configs = [
        {'chip': 'pwmchip2', 'channel': 0, 'name': 'PWM2 Channel 0 (Pin 31)'},
        {'chip': 'pwmchip3', 'channel': 0, 'name': 'PWM3 Channel 0 (Pin 34)'}
    ]
    
    for config in pwm_configs:
        chip = config['chip']
        channel = config['channel']
        name = config['name']
        chip_path = f"/sys/class/pwm/{chip}"
        pwm_path = f"{chip_path}/pwm{channel}"
        
        print(f"  配置 {name}...")
        
        # 检查PWM芯片是否存在
        if not os.path.exists(chip_path):
            print(f"    ✗ PWM芯片 {chip} 不存在")
            continue
        
        # 设置export文件权限
        export_file = f"{chip_path}/export"
        if os.path.exists(export_file):
            try:
                os.chmod(export_file, 0o666)
                print(f"    ✓ 设置权限: {export_file}")
            except OSError as e:
                print(f"    ✗ 设置权限失败 {export_file}: {e}")
        
        # 导出PWM通道（如果未导出）
        if not os.path.exists(pwm_path):
            try:
                # 使用subprocess来模拟echo命令
                subprocess.run(['bash', '-c', f'echo {channel} > {chip_path}/export'], 
                             check=True, capture_output=True)
                time.sleep(0.2)  # 等待导出完成
                print(f"    ✓ 导出PWM通道 {channel}")
            except subprocess.CalledProcessError as e:
                print(f"    ✗ 导出PWM通道失败: {e}")
                continue
        else:
            print(f"    ✓ PWM通道 {channel} 已导出")
        
        # 设置PWM控制文件权限
        if os.path.exists(pwm_path):
            control_files = ['period', 'duty_cycle', 'enable', 'polarity']
            for control_file in control_files:
                file_path = f"{pwm_path}/{control_file}"
                if os.path.exists(file_path):
                    try:
                        os.chmod(file_path, 0o666)
                        print(f"    ✓ 设置权限: {control_file}")
                    except OSError as e:
                        print(f"    ✗ 设置权限失败 {control_file}: {e}")
                else:
                    print(f"    ⚠ 控制文件不存在: {control_file}")
            
            print(f"    🎉 {name} 初始化完成")
        
        # 设置unexport文件权限
        unexport_file = f"{chip_path}/unexport"
        if os.path.exists(unexport_file):
            try:
                os.chmod(unexport_file, 0o666)
                print(f"    ✓ 设置权限: {unexport_file}")
            except OSError as e:
                print(f"    ✗ 设置权限失败 {unexport_file}: {e}")

def setup_i2c():
    """设置I2C设备权限"""
    print("\n📍 配置 I2C 设备...")
    
    # 项目只需要I2C-3
    required_i2c = '/dev/i2c-3'
    
    # I2C设备通常不需要手动导出，只需要设置权限
    if os.path.exists(required_i2c):
        try:
            os.chmod(required_i2c, 0o666)
            print(f"  ✓ 设置权限: {required_i2c}")
        except OSError as e:
            print(f"  ✗ 设置权限失败 {required_i2c}: {e}")
    else:
        print(f"  ⚠ 未找到项目需要的I2C设备: {required_i2c}")
    
    # 检查I2C适配器是否存在
    i2c_adapter = '/sys/class/i2c-adapter/i2c-3'
    if os.path.exists(i2c_adapter):
        print(f"  ✓ I2C-3 适配器存在: {i2c_adapter}")
    else:
        print(f"  ⚠ I2C-3 适配器不存在: {i2c_adapter}")

def setup_serial():
    """设置串口设备权限"""
    print("\n📍 配置 串口设备...")
    
    # 项目需要的串口设备 - 用于电机控制
    required_serial = '/dev/ttyACM0'
    
    if os.path.exists(required_serial):
        try:
            os.chmod(required_serial, 0o666)
            print(f"  ✓ 设置权限: {required_serial}")
        except OSError as e:
            print(f"  ✗ 设置权限失败 {required_serial}: {e}")
    else:
        print(f"  ⚠ 未找到项目需要的串口设备: {required_serial}")
        print(f"  💡 提示: 请确保电机控制器已连接并被系统识别")

def setup_user_groups():
    """添加用户到必要的组"""
    print("\n📍 配置用户组...")
    
    # 获取当前用户（不是root）
    current_user = os.environ.get('SUDO_USER', 'linaro')
    
    # 必要的用户组
    groups = ['dialout', 'i2c', 'gpio', 'input']
    
    for group in groups:
        try:
            # 检查组是否存在，不存在则创建
            try:
                subprocess.run(['getent', 'group', group], check=True, capture_output=True)
                print(f"  ✓ 组 '{group}' 已存在")
            except subprocess.CalledProcessError:
                subprocess.run(['groupadd', group], check=True)
                print(f"  ✓ 创建组: {group}")
            
            # 添加用户到组
            subprocess.run(['usermod', '-aG', group, current_user], check=True)
            print(f"  ✓ 添加用户 {current_user} 到组: {group}")
            
        except subprocess.CalledProcessError as e:
            print(f"  ✗ 组配置失败 {group}: {e}")

def verify_setup():
    """验证设置结果"""
    print("\n📍 验证配置...")
    
    success_count = 0
    total_checks = 0
    
    # 检查项目需要的GPIO引脚
    required_gpios = [133, 119, 125, 124, 138]
    for pin in required_gpios:
        total_checks += 1
        gpio_path = f"/sys/class/gpio/gpio{pin}"
        if os.path.exists(gpio_path):
            print(f"  ✓ GPIO {pin} 已导出")
            success_count += 1
        else:
            print(f"  ✗ GPIO {pin} 未导出")
    
    # 检查项目需要的I2C设备
    required_i2c = '/dev/i2c-3'
    total_checks += 1
    if os.path.exists(required_i2c):
        try:
            stat_info = os.stat(required_i2c)
            perms = oct(stat_info.st_mode)[-3:]
            if perms == '666':
                print(f"  ✓ I2C设备 {required_i2c} 权限正确: {perms}")
                success_count += 1
            else:
                print(f"  ⚠ I2C设备 {required_i2c} 权限不正确: {perms}")
        except OSError as e:
            print(f"  ✗ 无法检查 {required_i2c}: {e}")
    else:
        print(f"  ✗ I2C设备 {required_i2c} 不存在")
    
    # 检查项目需要的PWM通道
    required_pwm_channels = ['/sys/class/pwm/pwmchip2/pwm0', '/sys/class/pwm/pwmchip3/pwm0']
    for channel in required_pwm_channels:
        total_checks += 1
        if os.path.isdir(channel):
            print(f"  ✓ PWM通道已导出: {channel}")
            success_count += 1
        else:
            print(f"  ✗ PWM通道未导出: {channel}")
    
    # 检查项目需要的串口设备
    required_serial = '/dev/ttyACM0'
    total_checks += 1
    if os.path.exists(required_serial):
        try:
            stat_info = os.stat(required_serial)
            perms = oct(stat_info.st_mode)[-3:]
            if perms == '666':
                print(f"  ✓ 串口设备 {required_serial} 权限正确: {perms}")
                success_count += 1
            else:
                print(f"  ⚠ 串口设备 {required_serial} 权限不正确: {perms}")
        except OSError as e:
            print(f"  ✗ 无法检查 {required_serial}: {e}")
    else:
        print(f"  ✗ 串口设备 {required_serial} 不存在")
    
    print(f"\n📊 验证结果: {success_count}/{total_checks} 项检查通过")
    
    if success_count == total_checks and total_checks > 0:
        print("🎉 所有硬件配置成功！")
        return True
    else:
        print("⚠ 部分硬件配置可能存在问题")
        return False

def main():
    """主函数"""
    print("🤖 Open Duck Mini 硬件权限设置")
    print("=" * 50)
    
    try:
        check_root()
        setup_user_groups()
        setup_gpio()
        setup_pwm()
        setup_i2c()
        setup_serial()
        
        print("\n" + "=" * 50)
        if verify_setup():
            print("\n✅ 硬件权限设置完成！")
            print("💡 建议重启系统或重新登录以确保所有更改生效")
            print("🧪 然后可以运行测试脚本验证功能：")
            print("   python3 test_gpio_setup.py")
            print("   python3 test_pwm_setup.py")
            print("   python3 test_i2c_setup.py")
        else:
            print("\n⚠ 硬件权限设置完成，但部分项目可能需要手动检查")
            
    except KeyboardInterrupt:
        print("\n⚠ 用户中断了设置过程")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 设置过程出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 