from Phidget22.Phidget import *
from Phidget22.Devices.VoltageInput import *
from Phidget22.Devices.DigitalOutput import *
import time


def main():
    digitalOutput0 = DigitalOutput()
    digitalOutput1 = DigitalOutput()

    digitalOutput0.setChannel(0)
    digitalOutput1.setChannel(1)

    digitalOutput0.openWaitForAttachment(5000)
    digitalOutput1.openWaitForAttachment(5000)

    digitalOutput0.setState(True)
    digitalOutput1.setState(True)
    time.sleep(1)

    digitalOutput0.setState(False)
    digitalOutput1.setState(False)

    digitalOutput0.close()
    digitalOutput1.close()

main()