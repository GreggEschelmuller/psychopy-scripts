"""
Main script for visuomotor rotation experiment.
Combines all conditions (target/align, extension/flexion) into a single script.
"""

from psychopy import visual, core
import numpy as np
import pandas as pd
import src.lib as lib
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """Configuration class for experiment parameters"""
    cursor_size: float = 0.5
    target_size: float = 1.5
    home_range_upper: int = -1600
    home_range_lower: int = -1800
    sampling_rate: int = 500
    time_limit: int = 4
    screen_size: List[int] = (3440, 1440)
    target_jitter_range: Tuple[float, float] = (-2.5, 2.5)
    trial_delay_range: Tuple[int, int] = (1500, 2500)

class VisuomotorExperiment:
    def __init__(
        self,
        participant_id: int,
        config: Optional[ExperimentConfig] = None,
        experimenter: str = "Gregg",
        study_id: str = "Wrist Visuomotor Rotation"
    ):
        self.participant_id = participant_id
        self.config = config or ExperimentConfig()
        self.experimenter = experimenter
        self.study_id = study_id
        self.current_date = datetime.now()
        
        # Initialize psychopy window and visual elements
        self.setup_psychopy()
        
        # Initialize clocks
        self.move_clock = core.Clock()
        self.home_clock = core.Clock()
        self.trial_delay_clock = core.Clock()
        self.rt_clock = core.Clock()
        self.display_clock = core.Clock()
        
    def setup_psychopy(self):
        """Initialize PsychoPy window and visual elements"""
        self.win = visual.Window(
            fullscr=True,
            monitor="testMonitor",
            units="pix",
            color="black",
            waitBlanking=False,
            screen=1,
            size=self.config.screen_size,
        )

        self.home_indicator = visual.Circle(
            self.win,
            radius=lib.cm_to_pixel(1),
            fillColor="red",
            pos=[0, 100]
        )

        self.cursor = visual.Rect(
            self.win,
            width=lib.cm_to_pixel(self.config.cursor_size),
            height=lib.cm_to_pixel(60),
            fillColor="Black",
        )

        self.target = visual.Rect(
            self.win,
            width=lib.cm_to_pixel(self.config.target_size),
            height=lib.cm_to_pixel(60),
            lineColor="green",
            fillColor=None,
            lineWidth=5,
        )

    def setup_data_directory(self, experiment_type: str) -> str:
        """Create and validate data directory for the experiment"""
        dir_path = f"data/{experiment_type}/p{str(self.participant_id)}"
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print("Created new directory for participant data.")
        elif os.path.exists(dir_path) and os.listdir(dir_path):
            response = input("Directory exists and contains data. Overwrite? (y/n): ")
            if response.lower() != 'y':
                raise Exception("User chose not to overwrite existing data.")
                
        return f"{dir_path}/p{str(self.participant_id)}"

    def configure_vibration(self, vib_condition: int) -> Tuple[bool, bool]:
        """Configure vibration output based on condition"""
        vib_map = {
            0: (False, False),  # No vibration
            1: (True, True),    # Dual vibration
            2: (True, False),   # Triceps only
            3: (False, True),   # Biceps only
        }
        return vib_map.get(vib_condition, (False, False))

    def wait_for_home_position(self, input_task) -> None:
        """Wait for cursor to reach home position"""
        current_pos = [lib.volt_to_pix(lib.get_x(input_task)[-1]), 0]
        
        while True:
            self.home_indicator.color = "red"
            self.home_indicator.draw()
            self.win.flip()
            
            pot_data = lib.get_x(input_task)
            new_pos = lib.volt_to_pix(pot_data[-1])
            current_pos = [lib.exp_filt(current_pos[0], new_pos), 0]
            
            if (self.config.home_range_lower < current_pos[0] < self.config.home_range_upper):
                self.home_indicator.color = "Yellow"
                self.home_indicator.draw()
                self.win.flip()
                break

    def run_trial(self, trial_data: pd.Series) -> Tuple[Dict, Dict]:
        """Run a single trial and return trial data"""
        current_trial = lib.generate_trial_dict()
        position_data = lib.generate_position_dict()

        # Configure IO tasks
        input_task = lib.configure_input(self.config.sampling_rate)
        output_task = lib.configure_output()
        input_task.start()
        output_task.start()

        try:
            # Setup vibration
            vib_output = self.configure_vibration(trial_data.vibration)
            
            # Wait for home position
            self.wait_for_home_position(input_task)

            # Random delay before trial start
            rand_wait = np.random.randint(*self.config.trial_delay_range)
            current_trial["trial_delay"].append(rand_wait / 1000)
            self.trial_delay_clock.reset()
            while self.trial_delay_clock.getTime() < (rand_wait / 1000):
                continue

            # Configure cursor visibility
            self.cursor.color = "white" if trial_data.full_feedback else None

            # Start vibration and show target
            output_task.write(vib_output)
            
            # Setup target
            target_jitter = np.random.uniform(*self.config.target_jitter_range)
            target_amplitude = trial_data.target_amp + target_jitter
            current_target_pos = lib.calc_target_pos(0, target_amplitude)
            
            self.home_indicator.color = "black"
            lib.set_position(current_target_pos, self.target)
            self.win.flip()

            # Movement tracking
            self.move_clock.reset()
            current_pos = [lib.volt_to_pix(lib.get_x(input_task)[-1]), 0]
            
            while True:
                current_time = self.move_clock.getTime()
                pot_data = lib.get_x(input_task)
                current_deg = lib.volt_to_deg(pot_data[-1])
                new_pos = lib.volt_to_pix(pot_data[-1])
                current_pos = [lib.exp_filt(current_pos[0], new_pos), 0]
                
                self.target.draw()
                lib.set_position(current_pos, self.cursor)
                self.win.flip()
                
                if current_pos[0] > self.config.home_range_upper:
                    position_data["move_index"].append(1)
                    position_data["elbow_pos_pix"].append(current_pos[0])
                    position_data["pot_volts"].append(pot_data[-1])
                    position_data["time"].append(current_time)
                    position_data["elbow_pos_deg"].append(current_deg)
                
                if current_time >= self.config.time_limit:
                    break

        finally:
            # Cleanup
            input_task.stop()
            output_task.stop()
            input_task.close()
            output_task.close()

        return current_trial, position_data

    def run_block(self, block_name: str) -> None:
        """Run a block of trials"""
        # Load trial data
        condition = lib.read_trial_data("Trials.xlsx", block_name)
        block_data = lib.generate_trial_dict()
        
        for i, trial in condition.iterrows():
            print(f"\nStarting trial {i+1}")
            
            current_trial, position_data = self.run_trial(trial)
            
            # Save trial data
            trial_df = pd.DataFrame(position_data)
            trial_df.to_csv(
                f"{self.file_path}_{block_name}_trial_{i+1}_position_data.csv",
                index=False
            )
            
            # Update block data
            for key in block_data.keys():
                if key in current_trial and current_trial[key]:
                    block_data[key].extend(current_trial[key])
        
        # Save block data
        block_df = pd.DataFrame(block_data)
        block_df.to_csv(f"{self.file_path}_{block_name}_summary.csv", index=False)

    def run_experiment(self, experiment_type: str, blocks: List[str]) -> None:
        """Run the full experiment"""
        self.file_path = self.setup_data_directory(experiment_type)
        
        # Save study information
        study_info = {
            "Participant ID": self.participant_id,
            "Date_Time": self.current_date.strftime("%Y-%m-%d %H:%M:%S"),
            "Study ID": self.study_id,
            "Experimenter": self.experimenter,
            "Experiment Type": experiment_type
        }
        pd.DataFrame([study_info]).to_csv(f"{self.file_path}_study_information.csv", index=False)
        
        print("Experiment setup complete.")
        input("Press enter to begin first block...")
        
        for block in blocks:
            self.run_block(block)
            
        self.win.close()

def main():
    # Example usage
    participant_id = int(input("Enter participant ID: "))
    experiment_type = input("Enter experiment type (target_ext/target_flex/align_ext/align_flex): ")
    blocks = input("Enter block names (comma-separated): ").split(",")
    
    experiment = VisuomotorExperiment(participant_id)
    experiment.run_experiment(experiment_type, blocks)

if __name__ == "__main__":
    main() 