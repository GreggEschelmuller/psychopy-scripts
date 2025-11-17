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

def configure_input(
    sampling_rate: int, 
    ai_channel: str, 
    min_v: float, 
    max_v: float
) -> nidaqmx.Task:
    """Configure DAQ input task for continuous voltage reading.
    
    Args:
        sampling_rate: Sampling frequency in Hz
        ai_channel: DAQ analog input channel string (e.g., "Dev1/ai1")
        min_v: Minimum expected voltage
        max_v: Maximum expected voltage
    
    Returns:
        Configured DAQ task for analog input
    """
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(ai_channel, min_val=min_v, max_val=max_v)
    task.timing.cfg_samp_clk_timing(sampling_rate, sample_mode=AcquisitionType.CONTINUOUS)
    return task

def configure_output(do_channels: List[str]) -> nidaqmx.Task:
    """Configure DAQ output task for vibration control.
    
    Args:
        do_channels: List of DAQ digital output channel strings
    
    Returns:
        Configured DAQ task for digital output
    """
    task = nidaqmx.Task()
    for channel in do_channels:
        task.do_channels.add_do_chan(channel)
    return task

def generate_trial_dict() -> Dict[str, List]:
    """Generate empty dictionary for storing trial data.
    
    Returns:
        Dictionary with empty lists for all trial metrics
    """
    return {
        "trial_num": [], "move_times": [], "elbow_start_volts": [],
        "elbow_start_pix": [], "elbow_start_cm": [], "elbow_start_deg": [],
        "elbow_end_volts": [], "elbow_end_pix": [], "elbow_end_cm": [],
        "elbow_end_deg": [], "cursor_end_pix": [], "curs_end_cm": [],
        "curs_end_deg": [], "mean_velocity": [], "error": [], "block": [],
        "trial_delay": [], "target_cm": [], "target_deg": [], "target_pix": [],
        "rt": [],
    }

def generate_position_dict() -> Dict[str, List]:
    """Generate empty dictionary for storing position data.
    
    Returns:
        Dictionary with empty lists for position metrics
    """
    return {
        "move_index": [], "elbow_pos_pix": [], "elbow_pos_deg": [],
        "pot_volts": [], "time": [],
    }

def cm_to_pixel(cm: float, pixels_per_cm: float) -> float:
    """Convert centimeters to pixels for the display.
    
    Args:
        cm: Measurement in centimeters
        pixels_per_cm: Conversion factor for the specific display
    
    Returns:
        Equivalent measurement in pixels
    """
    return cm * pixels_per_cm

def pixel_to_cm(px: float, pixels_per_cm: float) -> float:
    """Convert pixels to centimeters for the display.
    
    Args:
        px: Measurement in pixels
        pixels_per_cm: Conversion factor for the specific display
    
    Returns:
        Equivalent measurement in centimeters
    """
    return px / pixels_per_cm

def pixel_to_volt(px: float, voltage_offset: float, voltage_scale: float) -> float:
    """Convert pixel position to voltage value.
    
    Args:
        px: Position in pixels
        voltage_offset: Calibration offset
        voltage_scale: Calibration scale factor
    
    Returns:
        Equivalent voltage value
    """
    return (px - voltage_offset) / voltage_scale

def volt_to_pix(volts: float, voltage_scale: float, voltage_offset: float) -> float:
    """Convert voltage to pixel position.
    
    Args:
        volts: Voltage value
        voltage_scale: Calibration scale factor
        voltage_offset: Calibration offset
    
    Returns:
        Equivalent position in pixels
    """
    return (volts * voltage_scale) + voltage_offset

def volt_to_deg(volts: float, degree_slope: float, degree_intercept: float) -> float:
    """Convert voltage to degrees.
    
    Args:
        volts: Voltage value
        degree_slope: Calibration slope
        degree_intercept: Calibration intercept
    
    Returns:
        Angle in degrees
    """
    return (degree_slope * volts) + degree_intercept

def pixel_to_deg(
    px: float, 
    voltage_offset: float, 
    voltage_scale: float, 
    degree_slope: float, 
    degree_intercept: float
) -> float:
    """Convert pixel position to degrees (chained conversion).
    
    Args:
        px: Position in pixels
        ... all calibration constants
    
    Returns:
        Angle in degrees
    """
    volts = pixel_to_volt(px, voltage_offset, voltage_scale)
    return volt_to_deg(volts, degree_slope, degree_intercept)

def cm_to_deg(
    cm: float, 
    pixels_per_cm: float,
    voltage_offset: float, 
    voltage_scale: float, 
    degree_slope: float, 
    degree_intercept: float
) -> float:
    """Convert centimeter measurement to degrees (chained conversion).
    
    Args:
        cm: Measurement in centimeters
        ... all calibration constants
    
    Returns:
        Angle in degrees
    """
    px = cm_to_pixel(cm, pixels_per_cm)
    return pixel_to_deg(px, voltage_offset, voltage_scale, degree_slope, degree_intercept)

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

def calc_target_pos(angle: float, amp: float, pixels_per_cm: float) -> Tuple[float, float]:
    """Calculate target position based on angle and amplitude.
    
    Args:
        angle: Target angle in degrees
        amp: Target amplitude in cm
        pixels_per_cm: Conversion factor for the display
    
    Returns:
        (x, y) position in pixels
    """
    magnitude = cm_to_pixel(amp, pixels_per_cm)
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