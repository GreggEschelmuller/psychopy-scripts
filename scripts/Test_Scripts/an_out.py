import nidaqmx
import time

task = nidaqmx.Task()
task.ao_channels.add_ao_voltage_chan("Dev1/ao0", 'mychannel', 0, 5)
task.start()
print('task started')


value = 4
task.write(4)
print('value written')
time.sleep(3)

task.write(0)

task.stop()
task.close()
print('stopped and closed task')


