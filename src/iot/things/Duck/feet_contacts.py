from periphery import GPIO
import numpy as np


LEFT_FOOT_PIN = 124
RIGHT_FOOT_PIN = 138


class FeetContacts:
    def __init__(self):
        self.left_foot_gpio = GPIO(LEFT_FOOT_PIN, 'in')
        self.right_foot_gpio = GPIO(RIGHT_FOOT_PIN, 'in')

    def get(self):
        left = False
        right = False
        if self.left_foot_gpio.read() == False:
            left = True
        if self.right_foot_gpio.read() == False:
            right = True
        return np.array([left, right])


if __name__ == "__main__":
    import time

    feet_contacts = FeetContacts()
    while True:
        print(feet_contacts.get())
        time.sleep(0.05)
