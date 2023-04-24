import nidaqmx
import time

task = nidaqmx.Task()
task.do_channels.add_do_chan("Dev1/port0/line0")
task.start()
print('task started')


task.write(True)
print('value written')
time.sleep(3)

task.write(False)

task.stop()
task.close()
print('stopped and closed task')
