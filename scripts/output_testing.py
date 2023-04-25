import nidaqmx
from psychopy import core

fs = 500

# Inputs
input_task = nidaqmx.Task()
input_task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=0, max_val=5)
input_task.ai_channels.add_ai_voltage_chan("Dev1/ai1", min_val=0, max_val=5)
input_task.timing.cfg_samp_clk_timing(
    fs, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
)

# Outputs - have to create separate tasks for input/output
output_task = nidaqmx.Task()
output_task.do_channels.add_do_chan("Dev1/port0/line0")
output_task.do_channels.add_do_chan("Dev1/port0/line1")


output_task.start()

# Test dual vibration
output1 = [True, True]
output2 = [True, False]
output3 = [False, True]

turn_off = [False, False]

timer = core.Clock()

timer.reset()
while timer < 3:
    output_task.write(output1)

output_task.write(turn_off)

while timer < 3:
    output_task.write(output2)

output_task.write(turn_off)

while timer < 3:
    output_task.write(output3)

output_task.write(turn_off)

output_task.stop()
output_task.close()
