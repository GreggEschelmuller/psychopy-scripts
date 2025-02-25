# Visuomotor Rotation Experiment

This repository contains a PsychoPy-based experiment for studying visuomotor rotations. The experiment supports multiple conditions including target/alignment tasks and extension/flexion movements.

## Requirements

- Python 3.7+
- PsychoPy
- numpy
- pandas
- nidaqmx (for DAQ device interaction)
- dataclasses

Install required packages using:

```bash
conda create -n psychopy python=3.7
conda activate psychopy
conda install psychopy numpy pandas
conda install -c anaconda dataclasses
pip install nidaqmx
```

## Hardware Requirements

- NI DAQ device (configured as Dev1)
- Display monitor (default configuration for 34" ultrawide, 3440x1440)
- Appropriate input sensors connected to DAQ device

## Directory Structure

```
.
├── main.py        # Main experiment script
├── src/
│   └── lib.py     # Helper functions and utilities
├── data/          # Generated during experiment runs
│   └── [experiment_type]/
│       └── p[participant_id]/
└── Trials.xlsx    # Experiment trial configurations
```

## Usage

1. Ensure your DAQ device is properly connected and configured
2. Prepare your Trials.xlsx file with the required trial configurations
3. Run the experiment:
   ```bash
   python main.py
   ```
4. Follow the prompts to enter:
   - Participant ID
   - Experiment type (options: target_ext, target_flex, align_ext, align_flex)
   - Block names (comma-separated, must match sheet names in Trials.xlsx)

## Trials.xlsx Configuration

The `Trials.xlsx` file is crucial for experiment configuration. Each sheet in the file represents a different block of trials.

### Required Sheet Names

- Practice sheets can be named "Practice" or "Testing"
- Experimental blocks should be named according to condition (e.g., "ext-targ", "ext-align")

### Required Columns

Each sheet must contain the following columns:

- `trial_num`: Sequential trial number
- `target_amp`: Target amplitude in cm
- `vibration`: Vibration condition (0: none, 1: dual, 2: triceps, 3: biceps)
- `full_feedback`: Boolean (True/False) for cursor visibility

### Example Format

```
| trial_num | target_amp | vibration | full_feedback |
|-----------|------------|-----------|---------------|
| 1         | 8.0       | 0         | True          |
| 2         | 12.0      | 1         | False         |
| 3         | 8.0       | 2         | True          |
```

### Tips

- Ensure all required columns are present
- Double-check sheet names match your experimental blocks
- Values must be in the correct format (numbers for trial_num, target_amp, vibration; boolean for full_feedback)
- Save the file in .xlsx format (not .xls)

## Experiment Types

- **target_ext**: Target reaching task with extension movements
- **target_flex**: Target reaching task with flexion movements
- **align_ext**: Alignment task with extension movements
- **align_flex**: Alignment task with flexion movements

## Data Output

For each experiment run, the following files are generated in the data directory:

- `p[ID]_study_information.csv`: General experiment information
- `p[ID]_[block]_trial_[N]_position_data.csv`: Detailed position data for each trial
- `p[ID]_[block]_summary.csv`: Summary data for each block

## Configuration

Default experiment parameters can be modified in the `ExperimentConfig` class:

- Screen settings (size, position)
- Timing parameters (sampling rate, trial delays)
- Visual elements (cursor size, target size)
- Movement thresholds

## Troubleshooting

1. **DAQ Device Issues**

   - Ensure device is properly connected
   - Check if device name matches 'Dev1' in configuration
   - Verify input/output channel configurations

2. **Display Issues**

   - Adjust screen settings in ExperimentConfig
   - Verify monitor selection (default: screen=1)

3. **Data Recording Issues**
   - Check write permissions in data directory
   - Verify Trials.xlsx format matches expected structure

## Contributing

Feel free to submit issues and enhancement requests.

## License

[Your License Here]
