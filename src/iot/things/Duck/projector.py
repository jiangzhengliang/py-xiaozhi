from periphery import GPIO
import time

PROJECTOR_GPIO = 125

class Projector:
    def __init__(self):
        self.projector=GPIO(PROJECTOR_GPIO, 'out')
        self.projector.write(False)
        self.on = False

    def switch(self):
        self.on = not self.on

        if self.on:
            self.projector.write(True)
        else:
            self.projector.write(False)


if __name__ == "__main__":
    p = Projector()
    while True:

        p.switch()
        time.sleep(1)
