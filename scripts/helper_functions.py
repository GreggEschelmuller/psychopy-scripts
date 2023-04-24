"""
This file contains a series of functions used for a wrist-based cursor control experiment.
The experiment is coded in psychopy. The functions and code were written by Gregg Eschelmuller.
"""

import numpy as np
import pandas as pd
import nidaqmx


# 24 inch diag - resololution 1920x1080
def cm_to_pixel(cm):
    return cm * 91.79


def pixel_to_cm(pix):
    return pix / 91.79


def read_trial_data(file_name, sheet=0):
    # Reads in the trial data from the excel file
    return pd.read_excel(file_name, sheet_name=sheet, engine="openpyxl")


def config_channel(ch_num1, ch_num2, fs):
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan("Dev1/ai" + str(ch_num1), min_val=0, max_val=5)
    task.ai_channels.add_ai_voltage_chan("Dev1/ai" + str(ch_num2), min_val=0, max_val=5)
    task.do_channels.add_do_chan("Dev1/port0/line0")
    task.timing.cfg_samp_clk_timing(
        fs, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS
    )
    return task


def make_rot_mat(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def exp_filt(pos0, pos1, alpha=0.5):
    x = (pos0[0] * alpha) + (pos1[0] * (alpha - 1))
    y = (pos0[1] * alpha) + (pos1[1] * (alpha - 1))
    return [x, y]


def get_xy(task):
    while True:
        vals = task.read(
            number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE
        )
        if vals == None:
            continue
        elif not vals == None:
            x_data = vals[0]
            y_data = vals[1]

            # If buffere returns multiple data points take the last one
            if len(x_data) > 1:
                x_data = [x_data[-1]]
            if len(y_data) > 1:
                y_data = [y_data[-1]]

            if not len(vals[0]) == 0:
                x = x_data[0] - 2.2
                y = y_data[0] - 2.2

                # Cursor gain
                x *= 550
                y *= 550
                return [x, y]


def set_position(pos, circ, rot_mat=make_rot_mat(0)):
    circ.pos = np.matmul(rot_mat, pos)
    circ.draw()


def calc_target_pos(angle, amp=8):
    # Calculates the target position based on the angle and amplitude
    magnitude = cm_to_pixel(amp)
    x = np.cos(angle * (np.pi / 180)) * magnitude
    y = np.sin(angle * (np.pi / 180)) * magnitude
    return x, y


def calc_amplitude(pos):
    # Calculates the amplitude of the cursor relative to middle
    amp = np.sqrt(np.dot(pos, pos))
    return amp


# defines rotation matrix for integrated cursor
def make_rot_mat(theta):
    # Makes a rotation matrix for the integrated cursor
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def contains(small_circ, large_circ):
    d = np.sqrt(
        (small_circ.pos[0] - large_circ.pos[0]) * 2
        + (small_circ.pos[1] - large_circ.pos[1]) * 2
    )
    return (d + small_circ.radius) < large_circ.radius


def save_end_point(data_dict, current_time, current_pos, int_cursor, condition, t_num):
    data_dict["Move_Times"].append(current_time)
    data_dict["Wrist_x_end"].append(current_pos[0])
    data_dict["Wrist_y_end"].append(current_pos[1])
    data_dict["Curs_x_end"].append(int_cursor.pos[0])
    data_dict["Curs_y_end"].append(int_cursor.pos[1])
    data_dict["Target_pos"].append(condition.target_pos[t_num])
    data_dict["Rotation"].append(condition.rotation[t_num])
    data_dict["End_Angles"].append(
        np.degrees(np.arctan2(int_cursor.pos[1], int_cursor.pos[0]))
    )
    return data_dict


def save_position_data(data_dict, int_cursor, current_pos, current_time):
    data_dict["Curs_x_pos"] = int_cursor.pos[0]
    data_dict["Curs_y_pos"] = int_cursor.pos[1]
    data_dict["Wrist_x_pos"] = current_pos[0]
    data_dict["Wrist_y_pos"] = current_pos[1]
    data_dict["Time"] = current_time
    return data_dict
