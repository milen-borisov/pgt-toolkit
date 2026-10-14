"""
Defines core data structures and utilities for working with photogrammetry coded targets.

It primarily defines:
1. TargetData class: A structure for holding key data (ID, type, position, size, geometry)
   of a single detected photogrammetry coded target.

2. TargetDataUtils class: Utility functionality for processing TargetData objects.

This module is part of the PGT-Toolkit project.
"""

# Import external modules
import os
from typing import Tuple, List

# Import internal modules
from targets.target_type import TargetTypeUtils
from utils.image import ImageMeta


# --------------------------------------------
# Target Data Class
# --------------------------------------------
class TargetData:
    # --- Class initialization functions ---
    def __init__(
        self,
        image_meta: ImageMeta,
        target_type: str,
        target_id: int | str | None,
        target_id_raw: str | None,
        target_geometry_type: str,
        target_geometry_data: any,
        image_threshold_mode: str | None,
        image_threshold_value: float | None,
        id_threshold_mode: str | None = None,
        id_threshold_value: float | None = None,
        id_threshold_is_alternative: bool = None,
    ):
        # Store metadata extracted from ImageData
        self.image_meta = image_meta

        # Type of the target; e.g. "cct_8-bit_white", "hrct_4-dots_white", ...
        self.type = target_type

        # Target ID
        self.id = self._get_id(target_id)
        self.id_raw = target_id_raw

        # Target center point coordinates and size
        if self.is_cct_target() or self.is_hrct_target():
            if target_geometry_type == "ellipse":
                target_center_pt, ellipse_axis_lengths, _angle = target_geometry_data

                # Target center point
                self.center_pt_x, self.center_pt_y = target_center_pt

                # Target size
                self.size = max(ellipse_axis_lengths) / 2.0
            else:
                raise ValueError(
                    f"Not supported target geometry type '{target_geometry_type}' for CCT and HRCT target."
                )
        else:
            raise ValueError(f"Not supported target type '{self.type}'.")

        # Geometric type of the target; e.g. "ellipse", "rectangle", ..
        self.geometry_type = target_geometry_type

        # Geometry data of the target
        self.geometry_data = target_geometry_data

        # Image thresholding parameters used for target detection in the image
        self.image_threshold_modes = [image_threshold_mode]
        self.image_threshold_values = [image_threshold_value]

        # ID thresholding parameters used for target ID decoding in the found target
        self.ids_raw = [self.id_raw]
        self.id_threshold_modes = [id_threshold_mode]
        self.id_threshold_values = [id_threshold_value]    
        self.id_threshold_is_alternatives = [id_threshold_is_alternative]

    # --- Get methods ---
    def get_orientated_center_pt(self) -> tuple[float, float]:
        """
        Transforms detected coordinates on rotated preview (OpenCV top-left 0,0)
        to Metashape physical sensor orientation (bottom-left 0,0).
        """
        x_alg, y_alg = self.center_pt_x, self.center_pt_y

        if not self.image_meta or self.image_meta.orientation == 1 or not self.image_meta.orientation:
            return x_alg, y_alg

        h_canvas = self.image_meta.height  # 6000 px for portrait
        ori = self.image_meta.orientation

        # Orientation 8 (90 deg CCW): 
        # X_meta = H_canvas - Y_alg
        # Y_meta = X_alg
        if ori == 8:
            return h_canvas - y_alg, x_alg

        # Orientation 6 (90 deg CW):
        elif ori == 6:
            w_canvas = self.image_meta.width
            return y_alg, w_canvas - x_alg

        # Orientation 3 (180 deg):
        elif ori == 3:
            w_canvas = self.image_meta.width
            return w_canvas - x_alg, h_canvas - y_alg

        return x_alg, y_alg
    
    def get_image_name(self) -> str:
        """
        Extracts and returns the base image name without the file extension.

        For example: 'path/to/test.jpg' -> 'test'.

        Returns:
            str: The image stem (name) without the extension.
        """
        # Uses os.path.splitext to safely split the file name into name and extension
        image_stem, _ = os.path.splitext(self.image_meta.file_name)
        return image_stem

    def _get_id(self, id: int | str | None) -> int | str | None:
        """
        Standardizes and validates a potential ID value, converting numeric strings
        to integers for easier processing.

        Args:
            id (int | str | None): The ID value to process. Can be an integer,
                                   a string, or None.

        Returns:
            int | str | None: The processed ID:
                              - int, if input was an int or a numeric string.
                              - str, if input was a non-numeric string.
                              - None, if input was None or an unsupported type.
        """
        if isinstance(id, int) or id is None:  # ID is int or None
            return id
        elif isinstance(id, str):  # ID is string
            if id.isdigit():
                return int(id)
            else:
                return id
        else:
            return None

    # --- Checking target type and state methods ---
    def is_cct_target(self):
        return TargetTypeUtils.is_cct_target(self.type)

    def is_hrct_target(self):
        return TargetTypeUtils.is_hrct_target(self.type)

    def is_decoded(self) -> bool:
        """
        Checks if the target has been successfully decoded.
        A target is considered decoded if its ID is an integer and is between 1 and 99

        This implementation uses early exits for clarity and explicit type checking.
        """
        # 1. Check if ID is None (returns False if None)
        if self.id is None:
            return False

        # 2. Check if ID is an integer (returns False if not an int, e.g., a string)
        if not isinstance(self.id, int):
            return False

        # 3. If we reach this point, we know self.id is a non-None integer.
        # Now, return True only if the ID is positive and less than 99.
        return self.id > 0 and self.id < 99

    # --- String representation for easy debugging ---
    def __repr__(self):
        """String representation for easy debugging."""
        return (
            f"Target: Image={self.image_meta.file_name}, "
            f"Type={self.type}, "
            f"ID={self.id}, " 
            f"ID_raw={self.id_raw}, "
            f"Center=({self.center_pt_x:.1f}, {self.center_pt_y:.1f}), "
            f"Size={self.size:.2f}, "
            f"Geom. Type={self.geometry_type}, Geom. Data={self.geometry_data}, "
            f"Img. Thresh. Mode={self.image_threshold_modes}, "
            f"Img. Thresh. Value={self.image_threshold_values}, "
            f"ID Thresh. Mode={self.id_threshold_modes}, "
            f"ID Thresh. Value={self.id_threshold_values}, "
            f"ID Thresh. Is Alt={self.id_threshold_is_alternatives}"
        )

    def short_repr(self):
        """Short string representation for easy debugging."""
        return (
            f"{self.image_meta.file_name}, {self.type}, "
            f"ID={self.id}, Center=({self.center_pt_x:.1f}, {self.center_pt_y:.1f}), "
            f"Size={self.size:.2f}"
        )


# --------------------------------------------
# Target Data Utilities Class
# --------------------------------------------
class TargetDataUtils:
    """
    Utility methods for processing TargetData objects.
    """

    # --- Join targets function ---
    @staticmethod
    def join_targets(
        targets1: List[TargetData],
        targets2: List[TargetData],
        coord_tolerance: float = 1.0,
        join_decoded_target_ids: bool = True,
    ) -> List[TargetData]:
        """
        Joins two lists of targets and removes duplicates based on proximity and ID status.

        Deduplication Rules:
        1. Two targets are duplicates if they are geometrically close (diff < tolerance) AND their IDs are equal.
        2. If targets are close but have different IDs:
           a. Keep the target with the decoded ID (ID > 0) and discard the un-decoded one.
           b. If both are decoded (but different IDs) OR both are un-decoded, keep both.
        """
        # 0. Define the separator character for joined IDs
        id_join_separator = '-' # :SETTING:

        # 1. Join the two lists
        combined_targets = targets1 + targets2
        cleaned_targets: List[TargetData] = []

        for new_target in combined_targets:
            is_handled = False  # True if the new target is a duplicate OR if it replaced an existing one

            # Check the new target against all targets already accepted into the cleaned list
            for ii, existing_target in enumerate(cleaned_targets):

                # Check for geometrical proximity
                x_diff = abs(new_target.center_pt_x - existing_target.center_pt_x)
                y_diff = abs(new_target.center_pt_y - existing_target.center_pt_y)
                is_close = (x_diff < coord_tolerance) and (y_diff < coord_tolerance)

                if is_close:

                    # Case A: Same ID (Definite Duplicate - keep the first one found)
                    if new_target.id == existing_target.id:
                        is_handled = True

                        cleaned_targets[ii].image_threshold_modes.append(new_target.image_threshold_modes[0])
                        cleaned_targets[ii].image_threshold_values.append(new_target.image_threshold_values[0])
                        cleaned_targets[ii].ids_raw.append(new_target.ids_raw[0])
                        if new_target.is_hrct_target() and existing_target.is_hrct_target():
                            cleaned_targets[ii].id_threshold_modes.append(new_target.id_threshold_modes[0])
                            cleaned_targets[ii].id_threshold_values.append(new_target.id_threshold_values[0])
                            cleaned_targets[ii].id_threshold_is_alternatives.append(new_target.id_threshold_is_alternatives[0])  

                        break  # Skip the new_target

                    # Case B: Different IDs - Conflict Resolution
                    else:
                        new_is_decoded = new_target.is_decoded()
                        existing_is_decoded_or_joined = existing_target.is_decoded()
                        if id_join_separator in str(existing_target.id):
                            existing_is_decoded_or_joined = True # the existing target is a joined between two decoded targets

                        if new_is_decoded and not existing_is_decoded_or_joined:
                            # New target is decoded and existing is not -> REPLACE existing with new
                            cleaned_targets[ii] = new_target
                            is_handled = True
                            break  # Handled by replacement, skip adding new_target later

                        elif not new_is_decoded and existing_is_decoded_or_joined:
                            # Existing target is decoded and new is not -> SKIP the new one
                            is_handled = True
                            break  # Skip the new_target

                        elif new_is_decoded and existing_is_decoded_or_joined:
                            if join_decoded_target_ids:
                                # Case C: Both Decoded and have different IDs

                                if TargetDataUtils.is_zero_in_target_id(new_target.id):
                                    # If the new target ID contains '0' in its two-digit representation, skip joining
                                    is_handled = True
                                    break  # Skip the new_target

                                if TargetDataUtils.is_zero_in_target_id(existing_target.id):
                                    # If the existing target ID contains '0' in its two-digit representation -> REPLACE existing with new that doesn't contain '0'
                                    cleaned_targets[ii] = new_target
                                    is_handled = True
                                    break  # Handled by replacement, skip adding new_target later                                
                                    
                                cleaned_targets[ii].id = (f"{cleaned_targets[ii].id}{id_join_separator}{new_target.id}")
                                cleaned_targets[ii].image_threshold_modes.append(new_target.image_threshold_modes[0])
                                cleaned_targets[ii].image_threshold_values.append(new_target.image_threshold_values[0])
                                cleaned_targets[ii].ids_raw.append(new_target.ids_raw[0])
                                if new_target.is_hrct_target() and existing_target.is_hrct_target():
                                    cleaned_targets[ii].id_threshold_modes.append(new_target.id_threshold_modes[0])
                                    cleaned_targets[ii].id_threshold_values.append(new_target.id_threshold_values[0])
                                    cleaned_targets[ii].id_threshold_is_alternatives.append(new_target.id_threshold_is_alternatives[0])

                                is_handled = True
                                break  # Handled by replacement, skip adding new_target later
                        else:
                            # Both targets are undecoded -> keep the existing one and append the new one's data
                            cleaned_targets[ii].image_threshold_modes.append(new_target.image_threshold_modes[0])
                            cleaned_targets[ii].image_threshold_values.append(new_target.image_threshold_values[0])
                            cleaned_targets[ii].ids_raw.append(new_target.ids_raw[0])
                            if new_target.is_hrct_target() and existing_target.is_hrct_target():
                                cleaned_targets[ii].id_threshold_modes.append(new_target.id_threshold_modes[0])
                                cleaned_targets[ii].id_threshold_values.append(new_target.id_threshold_values[0])
                                cleaned_targets[ii].id_threshold_is_alternatives.append(new_target.id_threshold_is_alternatives[0])

                            # Existing target is undecoded and new is also -> SKIP the new one
                            is_handled = True
                            break  # Skip the new_target

            if not is_handled:
                cleaned_targets.append(new_target)

        return cleaned_targets

    # --------------------------------------------
    @staticmethod  
    def is_zero_in_target_id(target_id: int | str | None) -> bool:
        """
        Checks if a single or two-digit integer's string representation (padded 
        to two digits with a leading zero if necessary) contains the digit '0'.
        """
        # ID is None
        if target_id is None:
            return False
        
        # ID is int
        if isinstance(target_id, int):
            if target_id < 0 or target_id > 99:
                return False
            else:
                # Key Change: Use f-string formatting to ensure a two-digit string (e.g., 9 becomes "09")
                target_id_str = f"{target_id:02d}"
                return '0' in target_id_str
        
        # ID is string
        if isinstance(target_id, str):
            if not target_id.isdigit():
                return False
            else:
                # Key Change: Use f-string formatting to ensure a two-digit string (e.g., 9 becomes "09")
                target_id_str = f"{target_id:02d}"
                return '0' in target_id_str