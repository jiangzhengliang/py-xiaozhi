from periphery import pwm
import numpy as np
import time
from threading import Thread

LEFT_ANTENNA_PWM = 2
RIGHT_ANTENNA_PWM = 3

LEFT_SIGN = 1
RIGHT_SIGN = -1


class Antennas:
    def __init__(self):

        self.pwm1 = pwm.PWM(LEFT_ANTENNA_PWM,0)
        self.pwm2 = pwm.PWM(RIGHT_ANTENNA_PWM,0)

        self.pwm1.frequency = 50
        self.pwm2.frequency = 50

        self.pwm1.duty_cycle = 0
        self.pwm2.duty_cycle = 0

        self.pwm1.polarity = 'normal'
        self.pwm2.polarity = 'normal'

        self.pwm1.enable()
        self.pwm2.enable()

        # 耳朵动作参数
        self.ear_movement_duration = 0.5  # 耳朵动作持续时间（秒）
        self.ear_movement_interval = 15.0  # 耳朵动作间隔（秒）
        self.ear_movement_amplitude = 0.3  # 耳朵动作幅度（-1到1之间）

        # 启动后台线程
        Thread(target=self.run, daemon=True).start()

    def map_input_to_angle(self, value):
        return 90 + (value * 90)

    def set_position_left(self, position):
        self.set_position(1, position, LEFT_SIGN)

    def set_position_right(self, position):
        self.set_position(2, position, RIGHT_SIGN)

    def set_position(self, servo, value, sign=1):
        """
        Moves the servo based on an input value in the range [-1, 1].
        :param servo: 1 for the first servo, 2 for the second servo
        :param value: A float between -1 and 1
        """

        # if value == 0:
        #     return
        if -1 <= value <= 1:
            angle = self.map_input_to_angle(value * sign)

            duty = 2 + (angle / 18)  # Convert angle to duty cycle (1ms-2ms)
            if servo == 1:
                self.pwm1.duty_cycle = duty / 100
            elif servo == 2:
                self.pwm2.duty_cycle = duty / 100
            else:
                print("Invalid servo number!")
            # time.sleep(0.01)  # Allow time for movement
        else:
            print("Invalid input! Enter a value between -1 and 1.")

    def move_ears(self):
        """
        执行耳朵动作：左右摆动一下
        """
        try:
            # 向左摆动
            self.set_position_left(self.ear_movement_amplitude)
            self.set_position_right(-self.ear_movement_amplitude)
            time.sleep(self.ear_movement_duration)
            
            # 向右摆动
            self.set_position_left(-self.ear_movement_amplitude)
            self.set_position_right(self.ear_movement_amplitude)
            time.sleep(self.ear_movement_duration)
            
            # 回到中间位置
            self.set_position_left(0)
            self.set_position_right(0)
            
        except Exception as e:
            print(f"耳朵动作执行失败: {e}")

    def run(self):
        """
        后台线程：每隔15秒动一下耳朵
        """
        while True:
            try:
                # 等待指定的间隔时间
                time.sleep(self.ear_movement_interval)
                
                # 执行耳朵动作
                self.move_ears()
                
            except Exception as e:
                print(f"耳朵动作线程错误: {e}")
                time.sleep(1)  # 出错时等待1秒再继续

    def stop(self):
        self.pwm1.close()
        self.pwm2.close()


if __name__ == "__main__":
    antennas = Antennas()
    print("耳朵动作测试：每隔15秒动一下耳朵")
    print("按 Ctrl+C 停止")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止耳朵动作")
        antennas.stop()
        
