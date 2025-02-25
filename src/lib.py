"""
Helper functions for visuomotor rotation experiment.
Handles DAQ configuration, unit conversions, and data management.
"""

from typing import Dict, List, Tuple, Any, Union
import numpy as np
import pandas as pd
import nidaqmx
from nidaqmx.constants import AcquisitionType
from psychopy.visual import BaseVisualStim

# Hardware constants
DAQ_DEVICE = "Dev1"
DAQ_VOLTAGE_MIN = 0
DAQ_VOLTAGE_MAX = 5
DAQ_AI_CHANNEL = f"{DAQ_DEVICE}/ai1"
DAQ_DO_CHANNELS = [f"{DAQ_DEVICE}/port0/line{i}" for i in range(2)]

# Display constants
DISPLAY_WIDTH_CM = 79.722  # 797.22 mm
DISPLAY_WIDTH_PX = 3440
PIXELS_PER_CM = DISPLAY_WIDTH_PX / DISPLAY_WIDTH_CM

# Voltage conversion constants
VOLTAGE_OFFSET = 12072
VOLTAGE_SCALE = -3263.8
DEGREE_SLOPE = -69.366
DEGREE_INTERCEPT = 364.26

def configure_input(sampling_rate: int) -> nidaqmx.Task:
    """Configure DAQ input task for continuous voltage reading.
    
    Args:
        sampling_rate: Sampling frequency in Hz
    
    Returns:
        Configured DAQ task for analog input
    """
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(DAQ_AI_CHANNEL, min_val=DAQ_VOLTAGE_MIN, max_val=DAQ_VOLTAGE_MAX)
    task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
    return task

def configure_output() -> nidaqmx.Task:
    """Configure DAQ output task for vibration control.
    
    Returns:
        Configured DAQ task for digital output
    """
    task = nidaqmx.Task()
    for channel in DAQ_DO_CHANNELS:
        task.do_channels.add_do_chan(channel)
    return task

def generate_trial_dict() -> Dict[str, List]:
    """Generate empty dictionary for storing trial data.
    
    Returns:
        Dictionary with empty lists for all trial metrics
    """
    return {
        "trial_num": [],
        "move_times": [],
        "elbow_start_volts": [],
        "elbow_start_pix": [],
        "elbow_start_cm": [],
        "elbow_start_deg": [],
        "elbow_end_volts": [],
        "elbow_end_pix": [],
        "elbow_end_cm": [],
        "elbow_end_deg": [],
        "cursor_end_pix": [],
        "curs_end_cm": [],
        "curs_end_deg": [],
        "mean_velocity": [],
        "error": [],
        "block": [],
        "trial_delay": [],
        "target_cm": [],
        "target_deg": [],
        "target_pix": [],
        "rt": [],
    }

def generate_position_dict() -> Dict[str, List]:
    """Generate empty dictionary for storing position data.
    
    Returns:
        Dictionary with empty lists for position metrics
    """
    return {
        "move_index": [],
        "elbow_pos_pix": [],
        "elbow_pos_deg": [],
        "pot_volts": [],
        "time": [],
    }

def cm_to_pixel(cm: float) -> float:
    """Convert centimeters to pixels for the display.
    
    Args:
        cm: Measurement in centimeters
    
    Returns:
        Equivalent measurement in pixels
    """
    return cm * PIXELS_PER_CM

def pixel_to_cm(px: float) -> float:
    """Convert pixels to centimeters for the display.
    
    Args:
        px: Measurement in pixels
    
    Returns:
        Equivalent measurement in centimeters
    """
    return px / PIXELS_PER_CM

def pixel_to_volt(px: float) -> float:
    """Convert pixel position to voltage value.
    
    Args:
        px: Position in pixels
    
    Returns:
        Equivalent voltage value
    """
    return (px - VOLTAGE_OFFSET) / VOLTAGE_SCALE

def volt_to_pix(volts: float) -> float:
    """Convert voltage to pixel position.
    
    Args:
        volts: Voltage value
    
    Returns:
        Equivalent position in pixels
    """
    return (volts * VOLTAGE_SCALE) + VOLTAGE_OFFSET

def volt_to_deg(volts: float) -> float:
    """Convert voltage to degrees.
    
    Args:
        volts: Voltage value
    
    Returns:
        Angle in degrees
    """
    return (DEGREE_SLOPE * volts) + DEGREE_INTERCEPT

def pixel_to_deg(px: float) -> float:
    """Convert pixel position to degrees.
    
    Args:
        px: Position in pixels
    
    Returns:
        Angle in degrees
    """
    return volt_to_deg(pixel_to_volt(px))

def cm_to_deg(cm: float) -> float:
    """Convert centimeter measurement to degrees.
    
    Args:
        cm: Measurement in centimeters
    
    Returns:
        Angle in degrees
    """
    return pixel_to_deg(cm_to_pixel(cm))

def read_trial_data(file_name: str, sheet: Union[str, int] = 0) -> pd.DataFrame:
    """Read trial configuration from Excel file.
    
    Args:
        file_name: Path to Excel file
        sheet: Sheet name or index
    
    Returns:
        DataFrame containing trial configuration
    """
    return pd.read_excel(file_name, sheet_name=sheet, engine="openpyxl")

def exp_filt(pos0: float, pos1: float, alpha: float = 0.5) -> float:
    """Apply exponential filter to position data.
    
    Args:
        pos0: Previous position
        pos1: Current position
        alpha: Filter coefficient (0-1)
    
    Returns:
        Filtered position
    """
    return (pos0 * alpha) + (pos1 * (1 - alpha))

def get_x(task: nidaqmx.Task) -> List[float]:
    """Get latest data from DAQ input task.
    
    Args:
        task: Configured DAQ input task
    
    Returns:
        List of voltage readings
    """
    while True:
        data = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
        if len(data) > 0:
            return data

def set_position(pos: Tuple[float, float], obj: BaseVisualStim) -> None:
    """Set position of a PsychoPy visual stimulus and draw it.
    
    Args:
        pos: (x, y) position in pixels
        obj: PsychoPy visual stimulus object
    """
    obj.pos = pos
    obj.draw()

def calc_target_pos(angle: float, amp: float = 8) -> Tuple[float, float]:
    """Calculate target position based on angle and amplitude.
    
    Args:
        angle: Target angle in degrees
        amp: Target amplitude in cm
    
    Returns:
        (x, y) position in pixels
    """
    magnitude = cm_to_pixel(amp)
    rad = np.radians(angle)
    return (
        np.cos(rad) * magnitude,
        np.sin(rad) * magnitude
    )

def calc_amplitude(pos: Tuple[float, float]) -> float:
    """Calculate amplitude of position relative to origin.
    
    Args:
        pos: (x, y) position
    
    Returns:
        Euclidean distance from origin
    """
    return np.sqrt(np.dot(pos, pos))