from periphery import pwm
import numpy as np
import time

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

    def stop(self):
        self.pwm1.close()
        self.pwm2.close()


if __name__ == "__main__":
    antennas = Antennas()

    s = time.time()
    while True:
        antennas.set_position_left(np.sin(2 * np.pi * 1 * time.time()))
        antennas.set_position_right(np.sin(2 * np.pi * 1 * time.time()))

        time.sleep(1 / 50)

        if time.time() - s > 5:
            break
        
