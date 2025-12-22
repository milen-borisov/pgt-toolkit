# Import external modules
import math
import numpy as np
from typing import Optional, Tuple, List, Union

# Predefine types
# Define LineSegment as the expected NumPy array output from cv2.HoughLinesP: (N, 1, 4)
LineSegmentArray = np.ndarray # Represents the (N, 1, 4) structure
# Define LineCoords for a single segment's coordinates: (x1, y1, x2, y2)
LineCoords = Tuple[float, float, float, float] 


# --------------------------------------------
# LineUtils class
# --------------------------------------------
class LineUtils:
    # --- Single Line Segment Calculations ---
    @staticmethod
    def _calculate_line_angle(line_coords: LineCoords, norm_0_180: bool = False) -> Optional[float]:
        """
        Calculates the angle (in degrees) for a line segment vector P1 -> P2.
        
        Note: The default norm_0_180 is changed to False to return [0, 360) 
              as per common vector practice, but the docstring is updated 
              to match your previous request.

        Args:
            line_coords: A tuple containing (x1, y1, x2, y2) coordinates.
            norm_0_180: If True, normalizes the angle to the [0, 180) range 
                        (for line orientation).
                        If False, returns the angle in the [0, 360) range 
                        (for vector direction).

        Returns:
            The angle of the line segment as a float (in degrees), or None 
            if input is invalid or the line has zero length.
        """
        # Check the input length
        if len(line_coords) != 4:
            return None

        # Fetch the coordinates
        x1, y1, x2, y2 = line_coords

        # Calculate delta components for vector P1 -> P2 (x2 - x1, y2 - y1)
        delta_x = x2 - x1
        delta_y = y2 - y1 
        
        # Check for zero length line segment
        if delta_x == 0 and delta_y == 0:
            # Returning None is usually better than 0.0 for a point, 
            # as it prevents false positives in angle distributions.
            return None 
        
        # Calculate angle using atan2(delta_y, delta_x).
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)

        # 1. Normalize the angle to the [0, 360) range (P1 -> P2 vector direction)
        # The modulo operator (%) handles the conversion from (-180, 180] to [0, 360)
        angle_360 = angle_deg % 360

        # 2. Normalize to the [0, 180) range if requested (line orientation)
        if norm_0_180:
            return angle_360 % 180

        # Default return is the [0, 360) vector direction
        return angle_360

    @staticmethod
    def _calculate_line_length(line_coords: LineCoords) -> Optional[float]:
        """
        Calculates the Euclidean length of a line segment given its start and end coordinates.

        Args:
            line_coords: A tuple containing (x1, y1, x2, y2) coordinates.

        Returns:
            The length of the line segment as a float, or None if input is invalid.
        """
        if len(line_coords) != 4:
            # Consider raising a ValueError for critical input errors instead of returning None
            return None

        x1, y1, x2, y2 = line_coords

        # Calculate the difference vector: (dx, dy) = (x2-x1, y2-y1)
        dx = x2 - x1
        dy = y2 - y1

        # Calculate the length using the distance formula (Euclidean norm)
        # L = sqrt(dx^2 + dy^2). math.hypot is the most robust way to calculate this.
        length = math.hypot(dx, dy)

        return length


    # --- Bulk Array Calculations (Optimized) ---
    @staticmethod
    def _calculate_line_angles(lines: Optional[LineSegmentArray], norm_0_180: bool = False) -> np.ndarray:
        """
        Calculates the angle (in degrees, [0, 360) or [0, 180)) for each line segment 
        in the provided HoughLinesP result array (N, 1, 4).
        
        This uses NumPy vectorized operations for much better performance.

        Args:
            lines: A NumPy array (LineSegmentArray) of line segments (N, 1, 4), or None.
            norm_0_180: Normalize the calculated angle to the [0, 180) range.

        Returns:
            np.ndarray: A NumPy array of angles (in degrees) for all segments.
        """
        if lines is None or lines.size == 0:
            return np.array([], dtype=np.float32)

        # Reshape the (N, 1, 4) array to a flat (N, 4) array
        coords = lines.reshape(-1, 4)
        
        # Extract columns: [x1, y1, x2, y2]
        x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]

        # Vectorized calculation for P1 -> P2
        delta_x = x2 - x1
        delta_y = y2 - y1 
        
        # Calculate angle (vectorized)
        angle_rad = np.arctan2(delta_y, delta_x)
        angle_deg = np.rad2deg(angle_rad)

        # 1. Normalize to [0, 360)
        # NumPy's % operator handles negative numbers for positive modulus correctly
        angles_360 = angle_deg % 360

        # 2. Normalize to [0, 180) if requested
        if norm_0_180:
            return angles_360 % 180
            
        return angles_360

    @staticmethod
    def _calculate_line_lengths(lines: Optional[LineSegmentArray]) -> np.ndarray:
        """
        Calculates the Euclidean length for a NumPy array of line segments,
        typically the output of cv2.HoughLinesP with shape (N, 1, 4).

        This uses NumPy vectorized operations for much better performance.

        Args:
            lines: A NumPy array (LineSegmentArray) of line segments (N, 1, 4), 
                or None if no lines were detected.

        Returns:
            np.ndarray: A NumPy array of calculated lengths (float).
        """
        if lines is None or lines.size == 0:
            return np.array([], dtype=np.float32)

        # Reshape the (N, 1, 4) array to a flat (N, 4) array
        coords = lines.reshape(-1, 4)
        
        # Extract columns: [x1, y1, x2, y2]
        x1, y1, x2, y2 = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]

        # Vectorized calculation for delta components
        delta_x = x2 - x1
        delta_y = y2 - y1

        # Calculate the length using NumPy's hypot, which is vectorized and robust
        lengths = np.hypot(delta_x, delta_y)

        return lengths
    

    # --- Dominant Orientation Calculation (Custom Clustered Logic) ---
    @staticmethod
    def find_dominant_angle(lines: Optional[LineSegmentArray], round_result: bool = True) -> Optional[float]:
        """
        Calculates the dominant line orientation by anchoring the reference to the longest 
        line segment and aligning all other perpendicular segments relative to it.

        Logic:
        1. Find the angle (A_max) of the line with maximum length (W_max).
        2. Normalize A_max to the [45, 135] reference range.
        3. Shift all other individual angles (+/- 90 degrees) to fall within 
           [A_max - 45, A_max + 45] (i.e., aligning all perpendicular lines to the A_max axis).
        4. Calculate the weighted mean of all transformed angles.
        5. Return the result normalized to [0, 90).

        Args:
            lines: A NumPy array of line segments (N, 1, 4).
            round_result: If True (default), rounds the final angle to the nearest integer.
                          If False, returns the precise float value.

        Returns:
            The dominant angle correction as a float in [0, 90) degrees, or None.
        """
        # 0. Fetch angles (in 0 to 180 deg) and lengths/weights of the lines
        angles_180 = LineUtils._calculate_line_angles(lines, norm_0_180=True)
        lengths = LineUtils._calculate_line_lengths(lines)

        mask = lengths > 0.0
        if not mask.any(): return None 
        
        angles = angles_180[mask]
        weights = lengths[mask]

        # 1. Find the angle of the line with maximum weight/length
        index_angle_max = np.argmax(weights)
        angle_max = angles[index_angle_max]

        # 2. Move A_max to the [45, 135] reference range
        angle_max_ref = angle_max
        
        # Shift until within the range [45, 135]
        while angle_max_ref < 45.0:
            angle_max_ref += 90.0
        while angle_max_ref > 135.0:
            angle_max_ref -= 90.0
        
        # 3. Next, for each other angle, apply the alignment shift
        transformed_angles = angles.copy()
        
        threshold_low = angle_max_ref - 45.0 # e.g., if ref is 90, low is 45
        threshold_high = angle_max_ref + 45.0 # e.g., if ref is 90, high is 135
        
        # Shift left (angle > A_max + 45): brings angles from the 180-side (e.g., 170 -> 80)
        # Note: We must ensure transformed_angles is in a working range for comparison. 
        # Since A_max_ref is in [45, 135], the comparison range is continuous [0, 180].

        # Shift left: If angle is too far clockwise past the ref axis
        mask_shift_left = transformed_angles > threshold_high
        transformed_angles[mask_shift_left] -= 90.0

        # Shift right: If angle is too far counter-clockwise before the ref axis
        mask_shift_right = transformed_angles < threshold_low
        transformed_angles[mask_shift_right] += 90.0

        # 4. Calculate the final linear weighted average
        # All angles are now aligned around the [A_max - 45, A_max + 45] axis.
        final_angle_unnorm = np.sum(transformed_angles * weights) / np.sum(weights)

        # 5. Normalize the result to the desired [0, 90) range
        result_angle = final_angle_unnorm % 90.0
        
        # Handle cases where the unnormalized angle is negative (e.g., -10)
        if result_angle < 0:
            result_angle += 90.0
        
        # 6. Return the result, optionally rounded
        if round_result:
            return float(np.round(result_angle))
        else:
            return float(result_angle)