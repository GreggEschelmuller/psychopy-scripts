# Imports
from psychopy import visual, core, event
import numpy as np
import pandas as pd
import helper_functions as hf
import pickle
from datetime import datetime
import copy
import os
import nidaqmx

# ------------------Blocks to run ------------------
# Use this to run whole protocol
# make sure the strings match the names of the sheets in the excel
# ExpBlocks = [
#     "Practice",
#     "Baseline",
#     "Exposure",
#     "Post"
#     ]

# For testing a few trials
ExpBlocks = ['Testing']

# ----------- Participant info ----------------
participant = 99
study_id = "Wrist Visuomotor Rotation"
experimenter = "Gregg"
current_date = datetime.now()
date_time_str = current_date.strftime("%Y-%m-%d %H:%M:%S")

study_info = {
    "Participant ID": participant,
    "Date_Time": date_time_str,
    "Study ID": study_id,
    "Experimenter": experimenter
}

print(study_info)
input("""
    Make sure changed the participant info is correct before continuing.
    Press enter to continue.
    """)

# Check if directory exists and if it is empty
dir_path = "data/P" + str(participant)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print("""
    Directory didn't exist so one was created. Continuing with program.
    """)
elif len(os.listdir(dir_path)) == 0:
    print("""
    Directory already exists and is empty. Continuing with program.""")
elif os.path.exists(dir_path) and not len(dir_path) == 0:
    print("""
    This directory exists and isn't empty, exiting program.
    Please check the contents of the directory before continuing.
    """)
    exit()

# set up file path
file_path = "data/P" + str(participant) + "/participant_" + str(participant)

# saves study information
with open(file_path + '_studyinfo.pkl', 'wb') as f:
    pickle.dump(study_info, f)

print('Setting everything up...')

# ------------------------ Set up --------------------------------

# Variables set up
cursor_size = 0.1
target_size = 0.1
home_size = 0.15
home_range_size = home_size * 5
fs = 500

# 0 deg rotation matrix to be used between trials (i.e. finding home)
no_rot = hf.make_rot_mat(0)

# Create your NI channels
# Inputs
task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=0, max_val=5)
task.ai_channels.add_ai_voltage_chan("Dev1/ai1", min_val=0, max_val=5)
task.timing.cfg_samp_clk_timing(fs, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

# Outputs - have to create separate tasks for input/output
task2 = nidaqmx.Task()
task2.do_channels.add_do_chan("Dev1/port0/line0")


# Load data structs

with open('template_data_dict.pkl', 'rb') as f:
    template_data_dict = pickle.load(f)

with open('template_trial_dict.pkl', 'rb') as f:
    template_trial_dict = pickle.load(f)

## Psychopy set up
# Creates window
win = visual.Window(fullscr=True, monitor='testMonitor',
                    units='pix', color='black', waitBlanking=False, screen=1, size=[1920, 1080])

# set up clocks
move_clock = core.Clock()
home_clock = core.Clock()

# set up stimuli
home = visual.Circle(
    win, radius=hf.cm_to_pixel(home_size), lineColor='red', fillColor=None)  # home position
home_range = visual.Circle(
    win, radius=hf.cm_to_pixel(home_range_size), lineColor=None)  # home range position
int_cursor = visual.Circle(
    win, radius=hf.cm_to_pixel(cursor_size), fillColor='Black')  # integrated pos
target = visual.Circle(
    win, radius=hf.cm_to_pixel(target_size), fillColor='green')  # initial target


print('Done set up')

# -------------- start practice trial loop ------------------------------------
input("Press enter to continue to first block ... ")
for block in range(len(ExpBlocks)):
    condition = hf.read_trial_data('Trials.xlsx', ExpBlocks[block])

    # Summary data dictionaries for this block
    end_point_data = copy.deepcopy(template_data_dict)

    # starts NI DAQ task for data collection
    task.start()

    for i in range(len(condition.trial_num)):
        # set up params
        full_feedback = condition.full_feedback[i]
        terminal_feedback = condition.terminal_feedback[i] # Load this from the excel
        vibration = condition.vibration[i]
        timeLimit = 3

        # Creates dictionary for single trial
        current_trial = copy.deepcopy(template_trial_dict)

        if condition.rotation[i]:
            rot_mat = hf.make_rot_mat(np.radians(condition.rotation_angle[i]))
        else:
            rot_mat = hf.make_rot_mat(0)

        home.draw()
        int_cursor.color = None
        int_cursor.draw()
        win.flip()

        # Checks if cursor is close to home and turns cursor white
        in_range = False
        current_pos = hf.get_xy(task)
        int_cursor.pos = current_pos
        while not in_range:
            if hf.contains(int_cursor, home_range):
                in_range = True
                int_cursor.color = 'white'
                int_cursor.draw()
                win.flip()
        current_pos = hf.get_xy(task)
        hf.set_position(current_pos, int_cursor, rot_mat)
        home.draw()
        win.flip()

        # Checks if cursor is in home position
        is_home = False
        while not is_home:
            prev_pos = int_cursor.pos
            if hf.contains(int_cursor, home):
                home_clock.reset()
                while True:
                    current_pos = hf.get_xy(task)
                    home.draw()
                    hf.set_position(current_pos, int_cursor, rot_mat)
                    win.flip()
                    
                    if home_clock.getTime() > 0.5:
                        is_home = True
                        break
                    if not hf.contains(int_cursor, home):
                        break

            current_pos = hf.get_xy(task)
            home.draw()
            hf.set_position(current_pos, int_cursor, rot_mat)
            win.flip()


        # Sets up target position
        current_target_pos = hf.calc_target_pos(condition.target_pos[i], condition.target_amp[i])
        hf.set_position(current_target_pos, target, no_rot)
        win.flip()

        # Run trial
        # Waits to continue until cursor leaves home position
        while hf.contains(int_cursor, home):
            current_pos = hf.get_xy(task)
            home.draw()
            hf.set_position(current_pos, int_cursor, rot_mat)
            target.draw()
            win.flip()
            
        if full_feedback:
            int_cursor.color = 'White'

        if vibration:
            task2.write(True)

        # run trial until time limit is reached or target is reached
        move_clock.reset()
        while move_clock.getTime() <= timeLimit:
            # Run trial
            current_time = move_clock.getTime()
            current_pos = hf.get_xy(task)
            target.draw()
            hf.set_position(current_pos, int_cursor, rot_mat)
            win.flip()

            # Saves the current position data
            current_trial = hf.save_position_data(current_trial, int_cursor, current_pos, current_time)
            
            if hf.calc_amplitude(current_pos) >= hf.cm_to_pixel(condition.target_amp[i]):
                if vibration:
                    task2.write(False)
                # Append trial data to storage variables
                if terminal_feedback:
                    int_cursor.color = 'White'
                    int_cursor.draw()
                    win.flip()
                end_point_data = hf.save_end_point(end_point_data, current_time, current_pos, int_cursor, condition, i)
                current_trial = hf.save_end_point(current_trial, current_time, current_pos, int_cursor, condition, i)
                break

        # Leave current window for 200ms
        core.wait(0.2, hogCPUperiod=0.2)
        int_cursor.color = None
        int_cursor.draw()
        win.flip()
        print(f"Trial {i} done.")
        print(f"Movement time: {current_time}")
        print(f"Target position: {condition.target_pos[i]}     Cursor Position: {round(np.degrees(np.arctan2(int_cursor.pos[1], int_cursor.pos[0])), 2)}")

        # Save current trial as pkl
        with open(file_path + '_practice_trial_' + str(i) + '.pkl', 'wb') as f:
            pickle.dump(current_trial, f)
        del current_trial

    print('Saving Data')
    # Save dict to excel as a backup
    file_ext = ExpBlocks[block]

    output = pd.DataFrame.from_dict(end_point_data)
    output['error'] = output['Target_pos'] - output['End_Angles']
    output.to_excel(file_path + file_ext + ".xlsx")

    # Save dict to pickle
    with open(file_path + "_" + file_ext + '.pkl', 'wb') as f:
        pickle.dump(end_point_data, f)
    print('Data Succesfully Saved')

    # This probably isn't needed but it makes it clear that we are replacing these for a new condition
    del output, end_point_data, condition
    task.stop()
    input("Press enter to continue to next block ... ")

task.close()
task2.close()
print("Experiment Done")

