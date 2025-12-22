"""
Base abstract class for all targets decoder subclasses.

The module is part of the PGT-Toolkit project.
"""

# Import external modules
import sys
import cv2
import math
import numpy as np
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union
from scipy.ndimage import map_coordinates

# Import internal modules that are from the pgt_toolkit folder
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))
from app.settings import settings
from app.logger import logger as log
from utils.image import ImageUtils, Image, ImageBinary
from utils.image import ImageUtils as TEST
from utils.math import MathUtils, Ellipse
from utils.bit import BitUtils
from typing import Optional, List
from targets.target_data import TargetData, TargetDataUtils
from targets.target_type import TargetTypeUtils

# Predefine types
Contour = np.ndarray
Hierarchy = np.ndarray


# --------------------------------------------
# Base abstract class of all targets decoder subclasses
# --------------------------------------------
class BTDecoder(ABC):

    # --- Initialization and cleanup functions ---
    def __init__(self):
        # Initialize with empty image data
        self._clear_input_data()

    def _init_input_data(self, target_type: str, image: Image, image_file_name: str):
        # Target type
        self.target_type = target_type
        self.target_type_name, self.target_code_number, self.target_color = (
            TargetTypeUtils.get_metadata(target_type)
        )

        # Image
        h, w = ImageUtils.get_image_dimensions(image)
        self._image = image  # remember the input image
        self._image_file_name = image_file_name
        self.image_area = h * w
        self.image_perimeter = 2 * (h + w)
        log.info("Image dimensions: %d x %d pixels (HxW)", h, w)

    def _clear_input_data(self):
        # Safely delete the debug image
        if self._dbg_image_is_created():
            del self._dbg_image

        # image file name
        self._image_file_name = ""

        # Reset numerical state variables using standard assignment (removed ': int' hints)
        self.image_area = 0
        self.image_perimeter = 0

        # Target properties data
        self.target_type = ""
        self.target_type_name = ""
        self.target_code_number = 0
        self.target_color = ""

    def _create_dbg_image(self):
        if not self._dbg_image_is_created():
            self._dbg_image = self._image.copy()  # use it for debugging and testing

    def _dbg_image_is_created(self) -> bool:
        if hasattr(self, "_dbg_image"):
            return True
        else:
            return False

    # --- Load settings  ---
    @staticmethod
    @abstractmethod
    def _load_settings():
        pass

    # --- Target finding functions ---
    def find_targets(
        self,
        target_type: str,
        image: Image,
        image_file_name: str,
        threshold_modes: Optional[List[str]],
        threshold_values: Optional[List[int]],
    ) -> List[TargetData]:
        """
        Finds and decodes targets in an image.

        Args:
            target_type: A string with the target type.
            image: The input OpenCV image (numpy.ndarray) containing targets.
            image_file_name: The file name of the image (used for exporting to CSV).
            threshold_modes: The threshold modes (used for detecting contours). One of ["adaptive", "otsu", "max"]
            threshold_values: The threshold values (used for detecting contours).
        Returns:
            A list with found targets, where each element contains TargetData object.
            Returns an empty list if no targets are found.
        """
        try:
            # Load settings
            self._load_settings()

            # Initialize image data
            self._init_input_data(target_type, image, image_file_name)

            # Find targets (using 'adaptive', 'otsu' or 'max' threshold modes)
            found_targets = []
            if threshold_modes is not None:
                for threshold_mode in threshold_modes:
                    found_targets_curr = self._find_targets(
                        image, threshold_mode=threshold_mode, threshold_value=None
                    )

                    found_targets = TargetDataUtils.join_targets(
                        found_targets,
                        found_targets_curr,
                        coord_tolerance=1,  # in pixels #:SETTING:
                        join_decoded_target_ids=True,
                    )

            # Find targets (using [e.g. 10, 245] threshold values)
            if threshold_values is not None:
                for threshold_value in threshold_values:
                    found_targets_curr = self._find_targets(
                        image, threshold_mode=None, threshold_value=threshold_value
                    )
                    found_targets = TargetDataUtils.join_targets(
                        found_targets,
                        found_targets_curr,
                        coord_tolerance=1,  # in pixels #:SETTING:
                        join_decoded_target_ids=True,
                    )

        finally:
            # Clear the image data
            self._clear_input_data()

        # Return
        return found_targets

    def _find_targets(
        self,
        image: Image,
        threshold_mode: Optional[str] = None,
        threshold_value: Optional[float] = None,
    ) -> List[TargetData]:
        """
        This is the entry point for the main logic for finding and
        decoding a target in an image.
        """
        # Remember the start time
        start_time = time.time()
        
        # Preprocess (Grayscale, Thresholding) the image for better targets detection; Returns a binary image
        self._image_threshold_mode = threshold_mode  # the threshold mode for the whole image
        processed_image_bin, self._image_threshold_value = (
            ImageUtils.preprocess_to_binary_image(
                image,
                remove_noise=False,
                # noise_kernel_size=3, # not used for remove_noise=False
                threshold_mode=threshold_mode,
                threshold_value=threshold_value,
            )
        )
        # TEST.plot_image(processed_image_bin)

        # Find all contours in the image
        decoder_str = f"{self.target_type_name.lower()}_decoder"  # e.g. "cct_decoder", "hrct_decoder"
        finding_contours_use_hierarchy_on = settings.get(
            f"{decoder_str}.finding_contours_use_hierarchy_on"
        )
        if finding_contours_use_hierarchy_on:
            all_contours, hierarchy = cv2.findContours(
                processed_image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
        else:
            all_contours, _hierarchy = cv2.findContours(
                processed_image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
            hierarchy = None

        log.info_separator_h2()
        log.info(
            f"Found {len(all_contours)} contours in the image "
            f"using threshold mode '{self._image_threshold_mode}' with value {self._image_threshold_value}."
        )
        # TEST.plot_contours(image, all_contours, 'green', 1)
        # TEST.plot_contours(ImageUtils.convert_image_to_color(processed_image_bin), all_contours, 'green', 1)

        # Filter candidate contours for target contour
        candidate_target_contours = self._filter_candidate_target_contours(
            all_contours, hierarchy
        )
        log.info(
            f"Left {len(candidate_target_contours)} contours that are candidates for target contour."
        )
        # TEST.plot_contours(image, candidate_target_contours, 'green', 2)
        # TEST.plot_contours(ImageUtils.convert_image_to_color(processed_image_bin), candidate_target_contours, 'green', 1)

        # Identify the actual targets from the candidate contours
        found_targets = []
        for contour in candidate_target_contours:
            target = self._extract_target_data(contour, image)
            if target is None:
                continue

            found_targets.append(target)

        if self._dbg_image_is_created():
            TEST.plot_image(self._dbg_image)

        # Log the elapsed time
        log.info(f"Elapsed time: {time.time() - start_time:.2f} s for finding targets using threshold mode '{self._image_threshold_mode}")

        # Return the found targets
        return found_targets

    # --- Undecoded targets functions ---
    def process_undecoded_targets(
        self, targets: List[TargetData], undecoded_targets_unique_index_ref: List[int]
    ) -> List[TargetData]:
        """
        Default implementation: Returns targets unchanged.
        Subclasses can optionally override this.
        """
        return targets

    # --- Target recognition and decoding functions ---
    @abstractmethod
    def _extract_target_data(self, contour: Contour, image: Image) -> TargetData | None:
        """
        Validates a candidate contour and extracts the target's decoded data.

        This is an **abstract method** that serves as the **recognition and decoding plug-in**
        for a specific target type. Concrete subclasses must implement the full logic
        required to:
        1.  Validate if the input contour is from the specific target type (e.g., checking
            shape, size, and internal pattern).
        2.  Decode the target ID and package the final structured target data for returning.

        Args:
            contour: The candidate contour (NumPy array of points) found during image segmentation.
            image: The source image (NumPy array) used for pixel sampling and content analysis.

        Returns:
            TargetData: The decoded target data containing its geometric parameters and ID.
            None: If the contour is not recognized as a target of this specific target type.
        """
        pass

    # --- Contour filtering functions ---
    @abstractmethod
    def _filter_candidate_target_contours(
        self, contours: List[Contour], hierarchy: Optional[Hierarchy]
    ) -> List[Contour]:
        """
        Abstract method that must be implemented by all subclasses.

        Filters the input list of contours based on a set of criteria to identify candidates
        for target contours.
        """
        pass

    def _filter_candidate_target_contours__shape_circular(
        self, contours: List[Contour]
    ) -> List[Contour]:
        """
        Filters the input list of contours based on a set of criteria to identify candidates
        for target contours, which are close to circular shape.

        This method applies filters for contour area, length, and circularity
        to refine the list of potential circular target contour. For now,
        it is used for CCT and HRCT targets.

        Args:
            contours (List[Contour]): A list of contours to filter.

        Returns:
            List[Contour]: A list of filtered contours.
        """
        # Get the decoder text in the settings.yaml file
        decoder_str = f"{self.target_type_name.lower()}_decoder"  # e.g. "cct_decoder", "hrct_decoder"

        # Get some settings for filtering contours
        circularity_threshold = settings.get(
            f"{decoder_str}.target_center_circularity_threshold"
        )
        log.info(
            f"Filtering contours with circularity threshold of {circularity_threshold:.2f}."
        )

        max_length_ratio = settings.get(
            f"{decoder_str}.target_center_contour_max_perimeter_ratio"
        )
        max_length = max_length_ratio * self.image_perimeter
        min_length = settings.get(f"{decoder_str}.target_center_contour_min_length")
        log.info(
            f"Filtering contours with length between {min_length:.1f} and {max_length:.1f} pixels."
        )

        max_area_ratio = settings.get(
            f"{decoder_str}.target_center_contour_max_area_ratio"
        )
        max_area = max_area_ratio * self.image_area
        min_area = settings.get(f"{decoder_str}.target_center_contour_min_area")
        log.info(
            f"Filtering contours with area between {min_area:.1f} and {max_area:.1f} pixels^2."
        )

        # Filter the counters to identify potential target contours
        filtered_contours = self._filter_contours_by_size_and_circularity(
            contours, min_length, max_length, min_area, max_area, circularity_threshold
        )

        return filtered_contours

    @staticmethod
    def _filter_contours_by_size_and_circularity(
        contours: List[Contour],
        contour_min_length,
        contour_max_length,
        contour_min_area,
        contour_max_area,
        contour_circularity_threshold,
    ) -> List[Contour]:
        """
        Filters a list of input contours based on predefined size constraints (length and area)
        and a minimum **circularity threshold**.

        Contour circularity is a measure of how closely the shape resembles a perfect circle
        (where a value of 1.0 indicates a perfect circle). Contours failing any of the criteria
        are discarded.

        Args:
            contours (List[Contour]): A list of OpenCV contours.
            contour_min_length (float): Minimum perimeter length allowed (in pixels).
            contour_max_length (float): Maximum perimeter length allowed (in pixels).
            contour_min_area (float): Minimum area allowed (in square pixels).
            contour_max_area (float): Maximum area allowed (in square pixels).
            contour_circularity_threshold (float): Minimum circularity value (0.0 to 1.0) required.

        Returns:
            List[Contour]: The list of contours that successfully passed all filtering criteria.
        """
        filtered_contours = []
        for contour in contours:
            # Get contour area and length
            contour_area = cv2.contourArea(contour, False)  # in pixels
            contour_length = cv2.arcLength(contour, True)  # in pixels

            # Filter out too small or too large contours based on length and area
            if (
                contour_length < contour_min_length
                or contour_length > contour_max_length
            ):
                continue
            if contour_area < contour_min_area or contour_area > contour_max_area:
                continue

            # Filter out non-circular contours based on a circularity threshold
            contour_circularity = MathUtils.calc_contour_circularity(
                contour_area, contour_length
            )
            if contour_circularity < contour_circularity_threshold:
                continue

            filtered_contours.append(contour)

        return filtered_contours

    # --- Image normalization functions ---
    def _get_normalized_square_image_for_ellipse(
        self, ellipse: Ellipse, image: Image, fill_color: Tuple[int, int, int] = (255, 255, 255),
        _recursion_depth: int = 0
    ) -> Image:
        """
        Extracts and normalizes the Region of Interest (ROI) around a fitted ellipse. 
        
        This function operates in two stages:
        1. **Boundary Check & Padding (Recursive Stage):** If the calculated ROI extends 
        beyond the image boundaries, the function recursively calls itself with a new, 
        larger, padded version of the input image and an adjusted ellipse position.
        2. **Extraction & Transformation (Final Stage):** Once the ellipse is safely 
        within the boundaries of the source image (or the padded image), the function 
        clips the ROI (implicitly via slicing the already-bounded ROI), calculates the 
        affine transformation matrix (M), and warps the ROI into a normalized square 
        image (a perfect circle).
        
        This ensures that ellipses near the edge are not clipped but are instead fully 
        captured against a background of 'fill_color'.

        Args:
            ellipse (Ellipse): The parameters of the fitted ellipse (center, axes, angle)
                as typically returned by cv2.fitEllipse.
            image (Image): The source image (OpenCV format) from which the ROI is extracted.
            fill_color (Tuple[int, int, int]): The BGR color used to pad the image 
                when the ROI extends outside its boundaries. Default is white (255, 255, 255).
            _recursion_depth (int): Internal counter to prevent infinite recursion. 
                                Should not be set by the caller.
        Returns:
            np.ndarray: The normalized square image (ROI) with the ellipse transformed
                        into a perfect circle, with dimensions equal to 'norm_size_int' x 'norm_size_int'.
            None: If the ellipse is invalid, or if the affine matrix cannot be estimated.
        """
        # Define a maximum depth to explicitly guarantee termination
        MAX_RECURSION_DEPTH = 1

        # --- 1. Setup: ---
        # Get the target center point and coordinate
        ellipse_center_pt, ellipse_axis_lengths, ellipse_angle = ellipse
        ellipse_center_pt_x_float, ellipse_center_pt_y_float = ellipse_center_pt

        # Get the size of the major axis of the ellipse; i.e. the size of the side of the normalized square image
        norm_size_float = max(ellipse_axis_lengths)
        norm_size_int = round(norm_size_float)  # This will be the size of the output normalized square image
        half_norm_size_float = (norm_size_float / 2.0)  # Calculate the half-side length needed for the bounding box

        # Get the corner points of the minimum bounded box/rectangle for the whole ellipse (used later for src_points)
        ellipse_min_rect_pts_float = cv2.boxPoints(ellipse)

        # Calculate the initial, un-clipped integer bounds of the ellipse's region of interest (ROI)
        # Note. These bounds guarantee the entire ellipse is included (if not clipped)
        roi_y_min_initial_int = int(math.floor(ellipse_center_pt_y_float - half_norm_size_float))
        roi_y_max_initial_int = int(math.ceil(ellipse_center_pt_y_float + half_norm_size_float))
        roi_x_min_initial_int = int(math.floor(ellipse_center_pt_x_float - half_norm_size_float))
        roi_x_max_initial_int = int(math.ceil(ellipse_center_pt_x_float + half_norm_size_float))

        # Calculate the size of the initial bounding box
        roi_height_initial_int = roi_y_max_initial_int - roi_y_min_initial_int
        roi_width_initial_int = roi_x_max_initial_int - roi_x_min_initial_int

        # Determine the larger dimension to ensure the square is big enough
        # This will be the base size of our final square ROI of the ellipse
        roi_size_final_int = max(roi_height_initial_int, roi_width_initial_int)

        # Calculate the half-side of the new integer square (e.g., 35/2 = 17.5)
        half_roi_size_final_int = roi_size_final_int / 2.0


        # Recenter the final square bounds on the floating-point center
        # This step ensures the square is centered, even after initial rounding/sizing
        roi_y_min_final_int = int(math.floor(ellipse_center_pt_y_float - half_roi_size_final_int))
        roi_y_max_final_int = int(math.ceil(ellipse_center_pt_y_float + half_roi_size_final_int))
        roi_x_min_final_int = int(math.floor(ellipse_center_pt_x_float - half_roi_size_final_int))
        roi_x_max_final_int = int(math.ceil(ellipse_center_pt_x_float + half_roi_size_final_int))

        # Final Validity and Clipping Check # Ensure min < max after re-centering
        if roi_y_min_final_int >= roi_y_max_final_int or roi_x_min_final_int >= roi_x_max_final_int:
            return None

        # --- 2. Check for padding: ---
        # Get the image dimension
        image_height, image_width = ImageUtils.get_image_dimensions(image)

        # Get pad values
        pad_top = max(0, -roi_y_min_final_int)
        pad_bottom = max(0, roi_y_max_final_int - image_height) 
        pad_left = max(0, -roi_x_min_final_int)
        pad_right = max(0, roi_x_max_final_int - image_width)
        
        # Check if the padding is necessary
        is_padding_needed = (pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0)

        if is_padding_needed and (_recursion_depth < MAX_RECURSION_DEPTH):
            # 2.1. Pad the image (i.e. extend the image border with the necessary 'white' pixels)
            image_padded = cv2.copyMakeBorder(
                image, 
                pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, value=fill_color)
            # TEST.plot_image(image_padded)
            # log.info(f"Pad the image (top, bottom, left, right) = {pad_top, pad_bottom, pad_left, pad_right}")

            # 2.2. Update the ellipse 
            # The center shifts by the padding added to the top (y) and left (x) sides.
            ellipse_center_pt_x_float_padded = ellipse_center_pt_x_float + pad_left
            ellipse_center_pt_y_float_padded = ellipse_center_pt_y_float + pad_top

            # Create the new ellipse tuple for reference (center, axes, angle)
            ellipse_padded = (
                (ellipse_center_pt_x_float_padded, ellipse_center_pt_y_float_padded),
                ellipse_axis_lengths, ellipse_angle
            )

            # 2.3. :TRICKY: call again the same method but for the padded image
            return self._get_normalized_square_image_for_ellipse(
                ellipse_padded, image_padded, fill_color, _recursion_depth + 1
            )
        elif is_padding_needed and (_recursion_depth >= MAX_RECURSION_DEPTH):
            # Optional: Handle case where padding is needed but we hit the max depth (shouldn't happen here)
            log.warning("Max recursion depth reached in padding logic in '_get_normalized_square_image_for_ellipse' function.")
            return None
        
        # --- 3. Affine Transformation: ---
        # Get the source points
        src_points = self._get_affine_src_points(
            ellipse_center_pt_x_float,
            ellipse_center_pt_y_float,
            ellipse_min_rect_pts_float,
            roi_x_min_final_int,
            roi_y_min_final_int,
        )

        # Get the destination points (normalized square, 0 to norm_size_int)
        dst_points = self._get_affine_dst_points(norm_size_int)

        # Use the standard OpenCV function for least-squares affine estimation (N > 3 points)
        M, _mask = cv2.estimateAffine2D(src_points, dst_points)

        # Check if the matrix M was successfully estimated
        if M is None:
            return None

        # Get the target ROI image from the input image
        target_roi_image = image[
            roi_y_min_final_int:roi_y_max_final_int,
            roi_x_min_final_int:roi_x_max_final_int,
        ]
        # self._create_dbg_image(); TEST.draw_rectangle(self._dbg_image, (target_roi_x_min, target_roi_y_min), (target_roi_x_max, target_roi_y_max), 'blue', 1);

        # Perform affine transformation on the target ROI to make the fitted target ellipse a perfect circle
        target_normalized_image = cv2.warpAffine(
            target_roi_image, M, dsize=(norm_size_int, norm_size_int)
        )
        # TEST.plot_image(target_normalized_image)


        # --- 4. Return: ---
        return target_normalized_image

    @staticmethod
    def _get_affine_src_points(
        center_x: float,
        center_y: float,
        min_rect_pts: np.ndarray,  # float coordinates
        roi_x_min: int,
        roi_y_min: int,
    ) -> np.ndarray:
        """
        Computes the source (src) points for the affine transformation,
        translated to the local ROI coordinate system (origin at roi_x_min/y_min).
        """
        # Subtracting the ROI min bounds translates global coordinates to the local ROI space.
        src_points = np.float32(
            [
                # 4 corner points of the min rotated rect
                [min_rect_pts[0][0] - roi_x_min, min_rect_pts[0][1] - roi_y_min],
                [min_rect_pts[1][0] - roi_x_min, min_rect_pts[1][1] - roi_y_min],
                [min_rect_pts[2][0] - roi_x_min, min_rect_pts[2][1] - roi_y_min],
                [min_rect_pts[3][0] - roi_x_min, min_rect_pts[3][1] - roi_y_min],
                # Center point
                [center_x - roi_x_min, center_y - roi_y_min],
            ]
        )
        return src_points

    @staticmethod
    def _get_affine_dst_points(square_side_size_int: int) -> np.ndarray:
        """
        Computes the destination (dst) points for the affine transformation.
        Defines the target of the transformation as an upright square of size
        (length x length), starting at the origin (0, 0).
        """
        # Defines the target as an upright square starting at local (0,0) and center.
        size_float = float(square_side_size_int)
        half_size_float = size_float / 2.0
        dst_points = np.float32(
            [
                [0.0, 0.0],
                [size_float, 0.0],
                [size_float, size_float],
                [0.0, size_float],
                [half_size_float, half_size_float],
            ]
        )
        return dst_points

    # --- Helper sampling image functions ---
    @staticmethod
    def _read_pixel_values_on_circle(
        image_bin: ImageBinary,
        circle_center_x: float,
        circle_center_y: float,
        circle_radius: float,
        num_sample_points: int,
    ) -> np.ndarray:
        """
        Calculates coordinates and reads the pixel values from a binary image along
        a circular path using **bilinear interpolation**.

        This method is crucial for sampling target features and converting the continuous
        pixel data back into a sequence of binary values (0 or 1).

        Args:
            image_bin (np.ndarray): The input image array, which MUST be strictly binary
                                    (containing only 0s and 1s).
            circle_center_x (float): The x-coordinate (column) of the circle's center.
            circle_center_y (float): The y-coordinate (row) of the circle's center.
            circle_radius (float): The radius of the sampling circle.
            num_sample_points (int): The total number of points to sample evenly
                                    around the circle.

        Returns:
            np.ndarray: A 1D array of sampled binary values (0 or 1) along the circle,
                        with length equal to `num_sample_points`.

        Raises:
            ValueError: If the input image contains pixel values other than 0 or 1.
            RuntimeError: If any calculated sampling coordinate falls outside the
                        image boundaries.
        """
        # --- 1. Validation and Input Check ---

        # Check if the image contains only 0 and 1
        # We use 'set' conversion for a quick check, as np.unique can be slow on large arrays.
        allowed_values = {0, 1}
        unique_values = set(np.unique(image_bin))

        if not (unique_values.issubset(allowed_values) and len(unique_values) > 0):
            raise ValueError(
                "The input 'binary_image' must be strictly binary, containing only the values 0 and 1. "
                f"Found unique values: ({unique_values})."
            )

        # --- 2. Calculate Coordinates and Interpolate ---

        # Calculate Angular Positions
        angles = np.linspace(0, 2 * np.pi, num_sample_points, endpoint=False)

        # Calculate Sample Coordinates
        sample_x = circle_radius * np.cos(angles) + circle_center_x
        sample_y = circle_radius * np.sin(angles) + circle_center_y

        # Validation and Boundary Check
        image_height, image_width = ImageUtils.get_image_dimensions(image_bin)
        if (
            np.any(sample_x < 0)
            or np.any(sample_x >= image_width - 1)
            or np.any(sample_y < 0)
            or np.any(sample_y >= image_height - 1)
        ):
            # Check against W-1/H-1 because map_coordinates requires coordinates
            # to be within the valid interpolation range.
            raise RuntimeError(f"Sampling coordinates fall outside the image boundary.")

        # Sample Pixel Values using Interpolation
        # map_coordinates takes a list of coordinate arrays: [y_coords, x_coords]
        coordinates = np.vstack([sample_y, sample_x])
        sampled_values_float = map_coordinates(
            input=image_bin,
            coordinates=coordinates,
            order=1,  # Bilinear interpolation
            mode="constant",
            cval=0.0,
        )

        # --- 3. Convert to Binary Output ---

        # Since the input was guaranteed binary (0 or 1), interpolated values are [0, 1].
        # Rounding at 0.5 restores the intended binary value.
        sampled_values_output = np.round(sampled_values_float).astype(np.uint8)

        return sampled_values_output


# --------------------------------------------
# Main execution block, if this script is executed directly
# Note. Use only for for development-time testing
# --------------------------------------------
if __name__ == "__main__":
    # Import tests.dev module
    from tests.dev import run_dev_test_find_hrct_targets, run_dev_test_find_cct_targets

    # Run dev test
    # run_dev_test_find_cct_targets()
    run_dev_test_find_hrct_targets()
