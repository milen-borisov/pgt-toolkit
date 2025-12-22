# Import external modules
import math
from typing import Tuple

# Predefine types
Ellipse = Tuple[Tuple[float, float], Tuple[float, float], float]


# --------------------------------------------
# Math Utils Class
# --------------------------------------------
class MathUtils:
    """A static class for managing helpful math operations."""

    # --- Circle functions ---
    @staticmethod
    def calc_contour_circularity(area, length):
        """
        Calculates the circularity of a contour.
        Returns a value between 0 and 1, where a value of 1.0 indicates a perfect circle.
        """
        # Check for a zero-length contour to prevent a division-by-zero error.
        if length == 0:
            return 0

        # Return the circularity value.
        return (4 * math.pi * area) / (length**2)

    # --- Ellipse functions ---
    @staticmethod
    def scale_ellipse(original_ellipse, scale_factor: float = 0.5) -> Ellipse:
        """
        Creates a new ellipse with the same center and angle, but with axes scaled
        by the specified factor.

        Args:
            original_ellipse: The ellipse tuple returned by cv2.fitEllipse().
                            Format: ((center_x, center_y), (axis_a, axis_b), angle)
            scale_factor: The factor by which to scale the axes lengths (default is 0.5).

        Returns:
            The new, scaled ellipse tuple.
        """
        # Unpack the components
        center_pt, axis_lengths, angle = original_ellipse

        # Scale the axes lengths (which is the second element, index 1)
        scaled_axes = (axis_lengths[0] * scale_factor, axis_lengths[1] * scale_factor)

        # Reassemble and return the new ellipse tuple
        return (center_pt, scaled_axes, angle)
    
    # --- Interpolation functions ---
    @staticmethod
    def interpolate_with_thresholds(value, boundary_value_down, boundary_value_up, out_thresh_value_down, out_thresh_value_up):
        """
        Performs linear interpolation between two threshold values, 
        clamping the result at the boundaries.

        :param v: The input value to be interpolated.
        :param boundary_value_down: The lower boundary for the input value.
        :param boundary_value_up: The upper boundary for the input value.
        :param thresh_value_down: The output value corresponding to boundary_value_down.
        :param thresh_value_up: The output value corresponding to boundary_value_up.
        :return: The interpolated/clamped output value.
        """
        
        # --- 1. Clamping Below the Lower Boundary ---
        # If the input value is less than or equal to the lower boundary, return the lower threshold value.
        if value <= boundary_value_down:
            return out_thresh_value_down

        # --- 2. Clamping Above the Upper Boundary ---
        # If the input value is greater than or equal to the upper boundary, return the upper threshold value.
        if value >= boundary_value_up:
            return out_thresh_value_up

        # --- 3. Linear Interpolation (Between Boundaries) ---
        # If the input value is strictly between the boundaries, perform linear interpolation.
        
        # Calculate the range of the input values
        input_range = boundary_value_up - boundary_value_down
        
        # Calculate the range of the output values (thresholds)
        output_range = out_thresh_value_up - out_thresh_value_down
        
        # Calculate the normalized position of 'v' within the input range (0 to 1)
        # This tells us how far 'v' is from the lower boundary as a fraction of the total range.
        normalized_position = (value - boundary_value_down) / input_range
        
        # Use the normalized position to find the corresponding point in the output range.
        interpolated_value = out_thresh_value_down + (normalized_position * output_range)
        
        return interpolated_value
