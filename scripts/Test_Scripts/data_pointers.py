def append_data(data_dict):
    data_dict['End_Angles'].append(1)
    return data_dict

position_data_template = {
    "Curs_y_pos": [],
    "Curs_x_pos": [],
    "Wrist_x_pos": [],
    "Wrist_y_pos": [],
    "Time": [],
}

# Dictionary for end point data
template_data_dict = {
    "End_Angles": [],
    "Curs_x_end": [],
    "Curs_y_end": [],
    "Wrist_x_end": [],
    "Wrist_y_end": [],
    "Move_Times": [],
    "Target_pos": [],
    "Rotation": [],
}

# Template to store data for each trial
template_trial_dict = {
    "Curs_y_end": [],
    "Curs_x_end": [],
    "Wrist_x_end": [],
    "Wrist_y_end": [],
    "Time": [],
    "End_Angles": [],
    "Curs_x_pos": [],
    "Curs_y_pos": [],
    "Wrist_x_pos": [],
    "Wrist_y_pos": [],
    "Move_Times": [],
    "Target_pos": [],
    "Rotation": [],
}

import copy

# Summary end point data dictionaries
practice_end_data = dict(template_data_dict)
print(f"practice {id(practice_end_data)}")
baseline_end_data = dict(template_data_dict)
print(f"baseline {id(baseline_end_data)}")
exposure_end_data = dict(template_data_dict)
print(f"exposure {id(exposure_end_data)}")
post_end_data = dict(template_data_dict)

practice_end_data = append_data(practice_end_data)
print(f"practice {id(practice_end_data)}")
print(practice_end_data)
print(f"baseline {id(baseline_end_data)}")
print(baseline_end_data)

