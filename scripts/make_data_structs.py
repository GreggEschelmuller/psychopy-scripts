import pickle

# Dictionary for end point data for whole block - will be used to generate excel
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

with open('template_data_dict.pkl', 'wb') as f:
    pickle.dump(template_data_dict, f)

with open('template_trial_dict.pkl', 'wb') as f:
    pickle.dump(template_trial_dict, f)