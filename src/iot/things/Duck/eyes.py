from periphery import GPIO
import numpy as np
import time
from threading import Thread

LEFT_EYE_GPIO = 133
RIGHT_EYE_GPIO = 119


class Eyes:
    def __init__(self):

        self.left_eye=GPIO(LEFT_EYE_GPIO, 'out')
        self.right_eye=GPIO(RIGHT_EYE_GPIO, 'out')
        self.left_eye.write(True)
        self.right_eye.write(True)

        self.blink_duration = 0.1

        Thread(target=self.run, daemon=True).start()

    def run(self):
        while True:
            self.left_eye.write(False)
            self.right_eye.write(False)
            time.sleep(self.blink_duration)
            self.left_eye.write(True)
            self.right_eye.write(True)

            next_blink = np.random.rand() * 4  # seconds

            time.sleep(next_blink)


if __name__ == "__main__":
	e = Eyes()
	while True:
		time.sleep(1)
