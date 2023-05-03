import matplotlib.pyplot as plt
import helper_functions as hf
import nidaqmx

fs = 500

input_task = nidaqmx.Task()
input_task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=0, max_val=5)
input_task.ai_channels.add_ai_voltage_chan("Dev1/ai1", min_val=0, max_val=5)
input_task.timing.cfg_samp_clk_timing(
    fs, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
)

input_task.start()
print("Starting Collection")
current_pos = []
for i in range(3000):
    current_pos.append(hf.get_xy(input_task))

plt.figure()
plt.plot(current_pos)
plt.show()