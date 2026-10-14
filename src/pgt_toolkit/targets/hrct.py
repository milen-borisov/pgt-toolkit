"""
Implements the core logic (decoding, generation, and drawing)
for Human Readable Coded Targets (HRCT) -- a novel target type,
developed in-house by the PGT team.

The module is part of the PGT-Toolkit project.
"""

# Import external modules
import os
import sys
import cv2
import math
import svgwrite
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Union

# Import internal modules that are from the current project
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))
from app.settings import settings
from app.logger import logger as log
from utils.image import ImageUtils, Image, ImageBinary
from utils.image import ImageUtils as TEST
from utils.math import MathUtils
from utils.color import ColorUtils
from utils.bit import BitUtils
from utils.string import StringUtils
from utils.list import ListUtils
from targets.target_data import TargetData, TargetTypeUtils
from targets.bt_drawer import BTDrawer
from targets.bt_decoder import BTDecoder, Contour, Hierarchy
from targets.bt_generator import BTGenerator


# --------------------------------------------
# Classes for drawing HRCT target
# --------------------------------------------


### Class for keeping settings for HRCT targets drawer ###
class HRCTDrawerSettings:
    @classmethod
    def load(cls):
        # Colors
        cls.TARGET_CENTER_LINES_COLOR = ColorUtils.standardize_color_input(
            settings.get("hrct_drawer.target_center_lines_color")
        )
        cls.TARGET_ID_TEXT_COLOR = ColorUtils.standardize_color_input(
            settings.get("hrct_drawer.target_id_text_color")
        )


### Class for drawing HRCT target in image ###
class HRCTDrawer(BTDrawer):
    @staticmethod
    def _load_settings():
        HRCTDrawerSettings.load()

    @staticmethod
    def draw_target(target: TargetData, image: Image) -> Image:
        """
        Draws the detected HRCT target on the input image.

        The drawing visually marks the target's location and geometric fit, including:
        1.  **Concentric Ellipses:** Drawing the outer, inner, and center ellipses,
            calculated by scaling the fitted outer ellipse. The thickness is dynamically
            scaled based on the image size to maintain visibility.
        2.  **Target ID:** Placing the target's decoded ID (or undecoded reference)
            as text near the center point.

        @Note: This function modifies the input image (in-place).

        Args:
            target (TargetData): The object containing the target's geometric (fitted ellipse)
                                and identifier information.
            image (Image): The source image (OpenCV format) to draw upon.

        Returns:
           Image: The modified input image with the target drawn.
        """
        # Load settings
        HRCTDrawer._load_settings()

        # Get length and scale factor
        (h, w) = ImageUtils.get_image_dimensions(image)
        image_length = (h + w) / 2.0
        scale_factor = image_length / 2000

        # Draw the fitted ellipses
        target_outer_ellipse = target.geometry_data
        scale_factor_ellipse_in = (
            HRCTDecoderSettings.TARGET_CENTER_RADIUS_IN
            / HRCTDecoderSettings.TARGET_CENTER_RADIUS_OUT
        )
        target_inner_ellipse = MathUtils.scale_ellipse(
            target_outer_ellipse, scale_factor_ellipse_in
        )
        target_center_ellipse = MathUtils.scale_ellipse(target_outer_ellipse, 3 / 20)

        ellipse_color = HRCTDrawerSettings.TARGET_CENTER_LINES_COLOR
        ellipse_thickness = round(1 * scale_factor)
        cv2.ellipse(
            image,
            target_outer_ellipse,
            color=ellipse_color,
            thickness=ellipse_thickness,
        )
        cv2.ellipse(
            image,
            target_inner_ellipse,
            color=ellipse_color,
            thickness=ellipse_thickness,
        )
        cv2.ellipse(
            image,
            target_center_ellipse,
            color=ellipse_color,
            thickness=ellipse_thickness,
        )

        # Draw a text for target_id
        text = str(target.id)
        text_color = HRCTDrawerSettings.TARGET_ID_TEXT_COLOR
        text_position = (
            int(target.center_pt_x + 0.7 * target.size),
            int(target.center_pt_y + 1.4 * target.size),
        )
        text_thickness = round(1 * scale_factor)
        text_font_scale = 1.2 * scale_factor
        cv2.putText(
            image,
            text,
            text_position,
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=text_font_scale,
            color=text_color,
            thickness=text_thickness,
        )

        return image


# --------------------------------------------
# Classes for decoding HRCT targets
# --------------------------------------------


### Class for keeping settings for HRCT targets decoder ###
class HRCTDecoderSettings:
    @classmethod
    def load(cls):
        # Target geometry settings
        cls.TARGET_SQUARE_SIZE = settings.get(
            "hrct_decoder.target_square_size"
        )  # e.g. 18.0
        cls.TARGET_SQUARE_AREA = cls.TARGET_SQUARE_SIZE**2

        cls.TARGET_CENTER_RADIUS_OUT = settings.get(
            "hrct_decoder.target_center_radius_out"
        )  # e.g. 3.0
        cls.TARGET_CENTER_RADIUS_IN = settings.get(
            "hrct_decoder.target_center_radius_in"
        )  # e.g. 2.55

        cls.DM9_GAP_SIZE = settings.get("hrct_decoder.dm9_gap_size")  # e.g. 0.45
        cls.DM9_DOT_SHAPE_SIZE = settings.get(
            "hrct_decoder.dm9_dot_shape_size"
        )  # e.g. 1.8
        cls.DM9_OUTER_MARGIN_EXTRA = settings.get(
            "hrct_decoder.dm9_outer_margin_extra"
        )  # e.g. 0.0

        cls.DM9_CELL_SIZE = cls.DM9_DOT_SHAPE_SIZE + cls.DM9_GAP_SIZE
        cls.DM9_SQUARE_SIZE = 3 * cls.DM9_CELL_SIZE

        cls.DM9_DOT_SHAPE_TYPE = settings.get("hrct_decoder.dm9_dot_shape_type")

        cls.DM9_CELL_AREA = cls.DM9_CELL_SIZE**2
        cls.DM9_DOT_SHAPE_AREA = cls.get_dot_shape_area()
        cls.DM9_DOT_SHAPE_2_CELL_AREA_RATIO = cls.DM9_DOT_SHAPE_AREA / cls.DM9_CELL_AREA
        cls.DM9_DOT_SHAPES_2_TARGET_AREA_RATIO = (cls.DM9_DOT_SHAPE_AREA * 9) / cls.TARGET_SQUARE_AREA

        # Target center classifications settings
        cls.TARGET_SCALED_IMAGE_SIZE = int(
            settings.get("hrct_decoder.target_scaled_image_size")
        )  # e.g. 240 pixels
        cls.TARGET_CENTER_NUMB_SAMPLE_POINTS = settings.get(
            "hrct_decoder.target_center_numb_sample_points"
        )  # e.g. 72

        # Decoding settings
        cls.DECODING_TARGET_ID_MORPH_CLOSE_KERNEL_SIZE = settings.get(
            "hrct_decoder.decoding_target_id_morph_close_kernel_size"
        )  # e.g. 9
        cls.DECODING_TARGET_ID_X_CHAR = settings.get(
            "hrct_decoder.decoding_target_id_x_char"
        )  # e.g. 'x'
        cls.DECODING_TARGET_ID_USE_ZERO_ON = settings.get(
            "hrct_decoder.decoding_target_id_use_zero_on"
        ) # e.g. 'True' 

        cls.DECODING_SCALDED_DM9_IMAGE_SIZE = settings.get(
            "hrct_decoder.decoding_scalded_dm9_image_size"
        )  # e.g. 120
        cls.DECODING_ROTATION_ANGLE_CLIP_TOLERANCE = settings.get(
            "hrct_decoder.decoding_rotation_angle_clip_tolerance"
        )  # e.g. 0.95

        cls.DECODING_DIGIT_BLACK_RATIO_THRESHOLD_TOLERANCE = settings.get(
            "hrct_decoder.decoding_digit_black_ratio_threshold_tolerance"
        )  # e.g. 0.3

        # Undecoded targets
        cls.UNDECODED_TARGETS_RETURN_ON = settings.get(
            f"hrct_decoder.undecoded_targets_return_on"
        )
        cls.UNDECODED_TARGETS_UNIQUE_NAME_ON = settings.get(
            f"hrct_decoder.undecoded_targets_unique_name_on"
        )

    # --- Helper methods ---
    @staticmethod
    def get_dot_shape_area():
        dm9_dot_shape_type = HRCTDecoderSettings.DM9_DOT_SHAPE_TYPE
        if dm9_dot_shape_type == "dot":
            return math.pi * HRCTDecoderSettings.DM9_DOT_SHAPE_SIZE**2 / 4.0
        elif dm9_dot_shape_type == "square":
            return HRCTDecoderSettings.DM9_DOT_SHAPE_SIZE**2
        else:
            raise ValueError(
                f"Invalid dot shape type: '{dm9_dot_shape_type}'. Expected 'dot' or 'square'"
            )


### Class for keeping settings for HRCT digit matrix ###
class HRCTDigitMatrixSettings:
    # Template  9-dots digit matrix settings
    DM9_0 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    DM9_1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    DM9_2 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

    DM9_3 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    DM9_4 = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])

    DM9_5 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])

    DM9_6 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])

    DM9_7 = np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]])

    DM9_8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    DM9_9 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    HRCT_DM9_MAP = {
        0: DM9_0,
        1: DM9_1,
        2: DM9_2,
        3: DM9_3,
        4: DM9_4,
        5: DM9_5,
        6: DM9_6,
        7: DM9_7,
        8: DM9_8,
        9: DM9_9,
    }

    # --- Helper functions ---
    @staticmethod
    def get_digits(n: int) -> tuple[int, int]:
        """
        Extracts the two digits of an integer n (0 <= n <= 99).
        Returns (tens_digit, units_digit).
        """
        if not (0 <= n <= 99):
            raise ValueError("Input must be an integer between 0 and 99.")

        # Tens digit: Integer division by 10
        tens_digit = n // 10

        # Units digit: Modulo 10
        units_digit = n % 10

        return tens_digit, units_digit

    @staticmethod
    def get_9dots_digit_matrix(digit: int):
        return HRCTDigitMatrixSettings.HRCT_DM9_MAP.get(digit)

    @staticmethod
    def get_digit_from_9dots_digit_matrix(digit_matrix) -> Optional[int]:
        if HRCTDecoderSettings.DECODING_TARGET_ID_USE_ZERO_ON:
            if np.array_equal(digit_matrix, HRCTDigitMatrixSettings.DM9_0):
                return 0
            
        if np.array_equal(digit_matrix, HRCTDigitMatrixSettings.DM9_1):
            return 1
        elif np.array_equal(digit_matrix, HRCTDigitMatrixSettings.DM9_2):
            return 2
        elif np.array_equal(digit_matrix, HRCTDigitMatrixSettings.DM9_3):
            return 3
        elif np.array_equal(digit_matrix, HRCTDigitMatrixSettings.DM9_4):
            return 4
        elif np.array_equal(digit_matrix, HRCTDigitMatrixSettings.DM9_5):
            return 5
        elif np.array_equal(digit_matrix, HRCTDigitMatrixSettings.DM9_6):
            return 6
        elif np.array_equal(digit_matrix, HRCTDigitMatrixSettings.DM9_7):
            return 7
        elif np.array_equal(digit_matrix, HRCTDigitMatrixSettings.DM9_8):
            return 8
        elif np.array_equal(digit_matrix, HRCTDigitMatrixSettings.DM9_9):
            return 9
        else:
            return None  # Invalid digit


### Class for decoding HRCT targets ###
class HRCTDecoder(BTDecoder):
    """
    A subclass that implements the main logic for finding and decoding HRCT target in an image.
    """

    # --- Settings ---
    @staticmethod
    def _load_settings():
        HRCTDecoderSettings.load()

    # --- Process undecoded targets functions ---
    def process_undecoded_targets(
        self, targets: List[TargetData], undecoded_targets_unique_index_ref: List[int]
    ) -> List[TargetData]:
        """
        Processes the list of detected targets, specifically managing the targets that
        could not be decoded (undecoded targets).

        This method implements the logic controlled by the HRCT settings:
        1.  **Retention Check:** Checks `UNDECODED_TARGETS_RETURN_ON` to decide
            whether to keep or discard the undecoded targets.
        2.  **Unique Naming (If Retained):** If retained (`undecoded_targets_return_on` is True)
            and `UNDECODED_TARGETS_UNIQUE_NAME_ON` is enabled, it calls
            `_set_unique_name_on_undecoded_targets` to assign a unique, temporary ID
            (based on their index reference and the placeholder character `x_char`).
        3.  **Removal (If Discarded):** If retained targets are set to be removed,
            it calls `_remove_undecoded_targets` to filter them out of the final list.

        Args:
            targets (List[TargetData]): The complete list of targets found and processed
                                        (including both decoded targets with integer IDs
                                        and undecoded targets with a placeholder ID).
            undecoded_targets_unique_index_ref (List[int]): A reference list of the unique
                                                            index assigned to the
                                                            current undecoded target.

        Returns:
            List[TargetData]: The filtered and potentially renamed list of targets
                            to be exported or used in subsequent steps.
        """
        # Get settings
        undecoded_targets_return_on = HRCTDecoderSettings.UNDECODED_TARGETS_RETURN_ON
        undecoded_targets_unique_name_on = (HRCTDecoderSettings.UNDECODED_TARGETS_UNIQUE_NAME_ON)
        # x_char = HRCTDecoderSettings.DECODING_TARGET_ID_X_CHAR

        # Process the found undecoded targets
        res_targets = targets
        if undecoded_targets_return_on:
            if undecoded_targets_unique_name_on:
                res_targets = HRCTDecoder._set_unique_name_on_undecoded_targets(
                    targets, undecoded_targets_unique_index_ref
                )
        else:
            res_targets = HRCTDecoder._remove_undecoded_targets(targets)

        # Return
        return res_targets

    @staticmethod
    def _remove_undecoded_targets(
        targets: List[TargetData]
    ) -> List[TargetData]:
        res_targets = [target for target in targets if target.is_decoded()]
        return res_targets

    @staticmethod
    def _set_unique_name_on_undecoded_targets(
        targets: List[TargetData],
        undecoded_targets_unique_index_ref: List[int],
    ) -> List[TargetData]:
        for target in targets:
            if not target.is_decoded():
                target.id = f"_{undecoded_targets_unique_index_ref[0]}_{target.id}" #:SETTING:
                undecoded_targets_unique_index_ref[0] += 1

        return targets

    # --- Contour filtering functions ---
    def _filter_candidate_target_contours(
        self, contours: List[Contour], hierarchy: Optional[Hierarchy]
    ) -> List[Contour]:
        """
        Applies a two-stage filtering process to a list of raw contours to isolate
        only those shapes that are candidates for the HRCT target center.

        The two stages are:
        1.  **Hierarchy/Parent Filtering (Structural Check):** If hierarchy information
            is available, the contours are filtered using `_filter_parent_contours`. This
            mandates that a candidate target center contour must have **at least one child
            contour nested inside it** (which corresponds to the white parts in the HRCT pattern).
        2.  **Shape and Circularity Filtering (Geometric Check):** The remaining contours
            are filtered based on geometric criteria (size, length, and circularity)
            using `_filter_candidate_target_contours__shape_circular`.

        Args:
            contours (List[Contour]): The initial list of contours found in the image.
            hierarchy (Optional[Hierarchy]): The contour hierarchy array, if available (e.g., from cv2.findContours).

        Returns:
            List[Contour]: The list of contours that passed both filtering stages and are
                            considered viable HRCT target centers.
        """
        filtered_contours = contours
        # TEST.plot_contours(self._image, filtered_contours, 'red', 1)
        if hierarchy is not None:
            log.info(f"Filtering contours with 'hierarchy' filter.")
            filtered_contours = self._filter_parent_contours(
                contours, hierarchy
            )  # The HRCT center contour must have one or more child contours in it.
            log.info(
                f"Left {len(filtered_contours)} contours after filtering with 'hierarchy' filter."
            )
        # TEST.plot_contours(self._image, filtered_contours, 'red', 1)

        filtered_contours = self._filter_candidate_target_contours__shape_circular(
            filtered_contours
        )
        return filtered_contours

    def _filter_parent_contours(
        self, contours: List[Contour], hierarchy: Hierarchy
    ) -> List[Contour]:
        """
        Filters a list of contours, keeping only those that have one or more
        child contours nested inside them (i.e., parent contours).

        Args:
            all_contours: The list of contours returned by cv2.findContours.
            hierarchy: The hierarchy array returned by cv2.findContours
                    (must be the inner array, typically hierarchy[0]).

        Returns:
            A new list containing only the parent contours.
        """

        # Check if hierarchy is in the expected (1, N, 4) format and flatten it
        if hierarchy.ndim == 3 and hierarchy.shape[0] == 1:
            hierarchy = hierarchy[0]

        if hierarchy.shape[0] != len(contours):
            log.error("Error: Contour and hierarchy list sizes do not match.")
            return []

        # Initialize the parent_contours
        parent_contours = []

        # Iterate through the contours and their corresponding hierarchy entries
        for i, contour in enumerate(contours):
            # The third index (hierarchy[i][2]) is the index of the first child.
            # If it is NOT -1, the current contour (contours[i]) is a parent
            # because it has at least one nested contour (child).
            if hierarchy[i][2] != -1:
                parent_contours.append(contour)

        return parent_contours

    # --- Target recognition and decoding functions ---
    def _extract_target_data(
        self, target_center_contour: Contour, image: Image
    ) -> Optional[TargetData]:
        """
        Tries to extract the full target data, including the decoded ID, from a given HRCT
        (High Resolution Coded Target) center contour.

        This is a multi-stage process critical for decoding HRCTs:
        1.  **Initial Ellipse Fitting and Normalization:** The input contour is fit to an
            ellipse. The ellipse's region is then affine-warped (normalized)
            to create a scaled, upright, and binary image of the center feature, compensating
            for perspective and rotation.
        2.  **Center Feature Validation:** The normalized center image is validated using
            `_is_hrct_target_center` to confirm it contains the distinctive HRCT cross shape.
        3.  **Rotation Angle Determination:** The dominant orientation of the cross shape
            is calculated to find the necessary rotation angle (`rot_target_angle`)
            required to orient the target image such that the **upper-right quadrant is white**,
            which is the standard orientation for decoding.
        4.  **Full Target Extraction & Final Preparation:** The outer ellipse, encompassing
            the entire target (center and digit matrices), is calculated. The full target
            area is then normalized, scaled, processed into a binary image, and finally
            **rotated** by `rot_target_angle`. The image is clipped and rescaled for
            standardized input to the decoder.
        5.  **Decoding:** The `_read_hrct_target_id` method is called on the final,
            standardized binary image to decode the unique **Target ID**.
        6.  **Packaging:** The decoded ID and the geometric data (center ellipse) are
            compiled into a **TargetData** object.

        Args:
            target_center_contour (Contour): The raw contour of the target's central
                                                feature (the black dot/crosshair).
            image (Image): The source image (OpenCV format) containing the target.

        Returns:
            Optional[TargetData]: The fully populated TargetData object if extraction
                                and decoding were successful, otherwise 'None'.
        """
        # --------------------------------------------------------------------------
        # --- 1.1. Initial Validation: If the input contour is HRCT target center ---

        # Fit ellipses for the target center area (1x)
        target_center_ellipse = cv2.fitEllipse(target_center_contour)

        # Get normalized scaled image (a normalized squared image of the target in perfect upright circle)
        norm_target_center_image_bin = self._get_normalized_square_image_for_ellipse(
            target_center_ellipse, image
        )
        if norm_target_center_image_bin is None:
            return None
        # TEST.plot_image(norm_target_center_image_bin)

        # Scale the normalized (affined transformed) target image to normalized square image with fixed size
        scaled_image_size = (
            HRCTDecoderSettings.TARGET_SCALED_IMAGE_SIZE
        )  # e.g. 240 pixels
        scaled_target_center_image_bin = cv2.resize(
            norm_target_center_image_bin,
            (
                scaled_image_size,
                scaled_image_size,
            ),  # tuple of integers for square image (width, height)
            interpolation=cv2.INTER_LANCZOS4,
            # fx and fy arguments are not needed when dsize is explicitly provided
        )
        # TEST.plot_image(scaled_target_center_image_bin)

        # Process (Grayscale, Thresholding, Erosion) the target image; return a binary image
        target_center_threshold_mode = self._image_threshold_mode
        target_center_threshold_value = self._image_threshold_value
        if target_center_threshold_mode == "adaptive":
            target_center_threshold_mode = "otsu" #:TRICKY: Use Otsu for adaptive thresholding in this step
            target_center_threshold_value = None

        proc_target_center_image_bin, _threshold = (
            ImageUtils.preprocess_to_binary_image(
                scaled_target_center_image_bin,
                remove_noise=True,
                noise_kernel_size=3,  #:SETTING:
                threshold_mode=target_center_threshold_mode,
                threshold_value=target_center_threshold_value,
            )
        )
        # TEST.plot_image(proc_target_center_image_bin)

        # Check if the target center image contains HRCT target center
        if not self._is_hrct_target_center(proc_target_center_image_bin):
            # TEST.plot_image(proc_target_center_image_bin)
            return None
        # TEST.plot_image(proc_target_center_image_bin)
        # TEST.plot_histogram_from_image(scaled_target_center_image_bin, title="HRCT Target Center Image Histogram")

        # --------------------------------------------------------------------------
        # --- 1.2. Determine the target rotation angle to make the upper-right part white ---

        # Clip the image to remain only the cross shape of the target center in the image
        clip_ratio = (1 / math.sqrt(2)) * (
            HRCTDecoderSettings.TARGET_CENTER_RADIUS_IN
            / HRCTDecoderSettings.TARGET_CENTER_RADIUS_OUT
        )  # i.e. leave only the center cross in the image
        clip_ratio = (
            clip_ratio * HRCTDecoderSettings.DECODING_ROTATION_ANGLE_CLIP_TOLERANCE
        )
        cross_image_bin = ImageUtils.clip_center_square_by_ratio(
            proc_target_center_image_bin, clip_ratio
        )
        # TEST.plot_image(cross_image_bin)

        # Calculate the rotation angle of the target center
        rot_target_angle = ImageUtils.find_dominant_angle(cross_image_bin, round_result = False)
        if rot_target_angle is None:
            log.warning("Could not extract the rotation angle of HRCT target center.")
            return None  # No target found
        # log.info(f"Target rotation angle: {rot_target_angle}")

        # Rotate the target image
        rotated_target_cross_image_bin_tmp = ImageUtils.rotate_image_to_angle(
            cross_image_bin, rot_target_angle
        )
        # TEST.plot_image(rotated_target_cross_image_bin_tmp)

        # Check is the upper-right part of the target center is white
        sample_image_bin = ImageUtils.clip_offset_square_by_ratio(
            rotated_target_cross_image_bin_tmp,
            clip_ratio=1/8,
            center_offset_ratio=(3/12.0, -3/12.0),
        )  #:SETTING:
        # TEST.plot_image(sample_image_bin)
        if not ImageUtils.is_image_almost_white(
            sample_image_bin, min_white_ratio=0.8
        ):  #:SETTING:
            # The upper-right part is black, rotate with +90 degrees to make it white
            rot_target_angle += 90

        # --------------------------------------------------------------------------
        # --- 1.3. Fetch the target binary image (normalized, scale, rotated, clipped) ---

        # Get the target outer ellipse (that contains the target center and the target digit matrices)
        scale_factor_outer_ellipse = math.sqrt(2) * (
            (HRCTDecoderSettings.TARGET_SQUARE_SIZE / 2)
            / HRCTDecoderSettings.TARGET_CENTER_RADIUS_OUT
        )
        # scale_factor_outer_ellipse = HRCTDecoder._adjust_scale_factor_outer_ellipse(target_center_ellipse, scale_factor_outer_ellipse) #:TRICKY: Not used for now
        target_outer_ellipse = MathUtils.scale_ellipse(
            target_center_ellipse, scale_factor_outer_ellipse
        )
        # cv2.ellipse(image, target_center_ellipse, color=(0, 255, 0), thickness=1); TEST.plot_image(image)
        # cv2.ellipse(image, target_outer_ellipse, color=(0, 255, 0), thickness=1); TEST.plot_image(image)

        # Get normalized scaled image (a normalized squared image of the target in perfect upright circle)
        norm_target_image = self._get_normalized_square_image_for_ellipse(
            target_outer_ellipse, image
        )
        if norm_target_image is None:
            return None
        # TEST.plot_image(norm_target_image)

        # Scale the normalized (affined transformed) target image to normalized square image with fixed size
        scaled_image_size = (
            HRCTDecoderSettings.TARGET_SCALED_IMAGE_SIZE
        )  # e.g. 240 pixels
        scaled_target_image = cv2.resize(
            norm_target_image,
            (
                scaled_image_size,
                scaled_image_size,
            ),  # tuple of integers for square image (width, height)
            interpolation=cv2.INTER_LANCZOS4,
            # fx and fy arguments are not needed when dsize is explicitly provided
        )
        # TEST.plot_image(scaled_target_image)

        # # Process (Grayscale) the target image; return a gray (0-255) image
        gray_target_image = ImageUtils.convert_image_to_gray(scaled_target_image)
        # TEST.plot_image(gray_target_image)

        # Rotate the target processed image to make white the upper-right part of the target center
        rot_target_image_gray = ImageUtils.rotate_image_to_angle(
            gray_target_image, rot_target_angle
        )
        # log.info(f"Rotated target image by {rot_target_angle} degrees."); TEST.plot_image(rot_target_image_gray);

        # Clip the target image
        _target_center_pt, ellipse_axis_lengths, _angle = target_outer_ellipse
        shorter_axis_length = min(ellipse_axis_lengths)
        correction_scale_coeff = MathUtils.interpolate_with_thresholds(
            #:SETTING:TRICKY: the small targets (in pixels size) need larger clipping area
            value=shorter_axis_length,
            boundary_value_down=0,  
            boundary_value_up=100, 
            out_thresh_value_down=0.975,  
            out_thresh_value_up=1.0,
        )
        clip_ratio = correction_scale_coeff * 1 / math.sqrt(2) #:TODO:SETTING:
        target_image_gray = ImageUtils.clip_center_square_by_ratio(
            rot_target_image_gray, clip_ratio
        )
        # TEST.plot_image(target_image_gray, title=f"correction_scale_coeff = {correction_scale_coeff}")

        # Scale the square target image to standard size
        target_image_gray = cv2.resize(
            target_image_gray,
            (
                scaled_image_size,
                scaled_image_size,
            ),  # tuple of integers for square image (width, height)
            interpolation=cv2.INTER_LANCZOS4,
            # fx and fy arguments are not needed when dsize is explicitly provided
        )
        # TEST.plot_image(target_image_gray)


        # --------------------------------------------------------------------------
        # --- 2. Decode the Target ID ---

        # Read target ID
        (target_id, target_id_raw, id_threshold_mode, id_threshold_value, id_threshold_is_alternative) = self._read_hrct_target_id(
            target_image_gray, self.target_code_number, self.target_color
        )

        # --------------------------------------------------------------------------
        # --- 3. Package Decoded Target Data ---

        # Create TargetData object for returning
        target_data = TargetData(
            self._image_meta,
            self.target_type,
            target_id,
            target_id_raw,
            "ellipse",
            target_center_ellipse,
            self._image_threshold_mode,
            self._image_threshold_value,
            id_threshold_mode,
            id_threshold_value,
            id_threshold_is_alternative
        )

        # --------------------------------------------------------------------------
        # --- 4. Return Final Target Data for Returning ---

        # Return
        return target_data

    def _is_hrct_target_center(self, image_bin: ImageBinary) -> bool:
        """
        Validates if a standardized, binary image contains the distinctive
        **HRCT target center** (a black ring with a white cross-like center feature).

        This validation is performed by three main checks on the normalized target image:
        1.  **Pixel Ratio Check (Inner Circle):** Ensures the ratio of white to total
            pixels in a large inner circle is balanced (e.g., between 1/3 and 2/3).
        2.  **Alternating Pattern Check (Mid-Radius):** Verifies the 4-part alternating
            symmetry characteristic of the HRCT crosshair structure.
        3.  **Pixel Ratio Check (Small-Radius):** Ensures the pixel values sampled
            very close to the center are not overwhelmingly one color (e.g.,
            at least 33.33% of the sampled points must be of the minority color).

        Args:
            image_bin (ImageBinary): The preprocessed, scaled, normalized, and binary
                                    (0 or 1) image of the target's center region.

        Returns:
            bool: **True** if the image contains the expected HRCT target center pattern,
                **False** otherwise.
        """
        # --------------------------------------------------------------------------
        # --- 1. Setup Constants and Dimensions ---
        # --------------------------------------------------------------------------

        # Get image dimensions
        image_size = ImageUtils.get_square_image_size(image_bin)

        # Center coordinates (e.g., 33 / 2.0 = 16.5)
        target_center_x = image_size / 2.0
        target_center_y = image_size / 2.0

        # Target max radius is half the image side
        target_center_r_out_size = image_size / 2.0
        target_center_r_in_size = target_center_r_out_size * (
            HRCTDecoderSettings.TARGET_CENTER_RADIUS_IN
            / HRCTDecoderSettings.TARGET_CENTER_RADIUS_OUT
        )

        # --------------------------------------------------------------------------
        # --- 2. Check for equal amount of black an white pixels in the inner circle ---
        # --------------------------------------------------------------------------
        # @TEST
        # image_test = ImageUtils.convert_image_to_color(image_bin)
        # ImageUtils.draw_center_circle(image_test, int(target_center_r_in_size*(0.95)))
        # TEST.plot_image(image_test)

        (total_numb_pixels, numb_white_pixels, _numb_black_pixels) = (ImageUtils.analyze_inner_circle(image_bin, target_center_r_in_size * 0.92))  #:SETTING:
        white_pixels_ratio = numb_white_pixels / total_numb_pixels
        if not (1 / 3 < white_pixels_ratio < 2 / 3):  #:SETTING:
            return False  # Skip the almost white or black target center

        # --------------------------------------------------------------------------
        # --- 3.1. Check for alternating symmetry pattern near the ring (Big-Radius) ---
        # --------------------------------------------------------------------------
        # Define the radius of the sampling circle
        sampling_circle_r = (9/12.) * target_center_r_in_size  # :SETTING:        

        # @TEST
        # image_tmp = ImageUtils.convert_image_to_color(image_bin)
        # ImageUtils.draw_center_circle(image_tmp, int(sampling_circle_r))
        # TEST.plot_image(image_tmp)

        # Check for the alternating pattern on the sampling circle
        is_pattern_found = self._check_target_center_for_alternating_pattern(
            image_bin, sampling_circle_r, target_center_x, target_center_y
        )

        if not is_pattern_found:
            return False
        

        # --------------------------------------------------------------------------
        # --- 3.2. Check for alternating symmetry pattern near the center (Small-Radius) ---
        # --------------------------------------------------------------------------
        # Define the radius of the sampling circle
        sampling_circle_r = (5/12.) * target_center_r_in_size  # :SETTING: 

        # @TEST
        # image_tmp = ImageUtils.convert_image_to_color(image_bin)
        # ImageUtils.draw_center_circle(image_tmp, int(sampling_circle_r))
        # TEST.plot_image(image_tmp)

        # Check for the alternating pattern on the sampling circle
        is_pattern_found = self._check_target_center_for_alternating_pattern(
            image_bin, sampling_circle_r, target_center_x, target_center_y
        )

        if not is_pattern_found:
            return False

        # --------------------------------------------------------------------------
        # --- 3.3. Check pixel balance near the center (Small-Radius) ---
        # --------------------------------------------------------------------------
        # Define the radius of the sampling circle
        sampling_circle_r_small = (4/12.) * target_center_r_in_size  # :SETTING:
        numb_sample_points = HRCTDecoderSettings.TARGET_CENTER_NUMB_SAMPLE_POINTS      
        
        # @TEST
        # image_tmp = ImageUtils.convert_image_to_color(image_bin)
        # ImageUtils.draw_center_circle(image_tmp, int(sampling_circle_r_small))
        # TEST.plot_image(image_tmp)

        # Read pixel values on the sample circle
        sampling_values_small = self._read_pixel_values_on_circle(
            image_bin=image_bin,
            circle_center_x=target_center_x,
            circle_center_y=target_center_y,
            circle_radius=sampling_circle_r_small,
            num_sample_points=numb_sample_points,
        )

        # Check if the sampled points are not overwhelmingly one color
        numb_white_samples = sum(sampling_values_small)
        numb_black_samples = numb_sample_points - numb_white_samples

        # The minority color count must be greater than or equal to 33.33% of the total samples
        min_required_samples = numb_sample_points * 1/3 # :SETTING: 
        
        # We check the minimum of the two counts (white and black)
        min_samples = min(numb_white_samples, numb_black_samples)
        
        if min_samples < min_required_samples:
            return False # The center is too skewed towards one color
        
        # Return 
        # If execution reaches this point, all three validation checks have passed:
        # 1. Inner Circle Pixel Ratio (Section 2)
        # 2. Alternating Symmetry Pattern (Section 3.1 and 3.2)
        # 3. Small Radius Pixel Balance (Section 3.3)
        return True
    
    def _check_target_center_for_alternating_pattern(
        self, 
        image_bin : ImageBinary, 
        sampling_circle_r: float, 
        target_center_x: float, 
        target_center_y: float
    ) -> bool:
        """
        Validates if a standardized, binary image contains the distinctive
        **HRCT target center** (a black ring with a white cross-like center feature)
        by checking for the 4-part alternating symmetry pattern.

        This validation is performed by checking the pixel values sampled
        on a circle of given radius around the target center.

        Args:
            image_bin (ImageBinary): The preprocessed, scaled, normalized, and binary
                                    (0 or 1) image of the target's center region.
            sampling_circle_r (float): The radius of the sampling circle.
            target_center_x (float): The x-coordinate of the target center.
            target_center_y (float): The y-coordinate of the target center.
        Returns:
            bool: **True** if the image contains the expected HRCT target center pattern,
                  **False** otherwise.
        """
        # @TEST
        # image_tmp = ImageUtils.convert_image_to_color(image_bin)
        # ImageUtils.draw_center_circle(image_tmp, int(sampling_circle_r))
        # TEST.plot_image(image_tmp)

        # Read pixel values on the sample circle
        numb_sample_points = HRCTDecoderSettings.TARGET_CENTER_NUMB_SAMPLE_POINTS
        sampling_values = self._read_pixel_values_on_circle(
            image_bin=image_bin,
            circle_center_x=target_center_x,
            circle_center_y=target_center_y,
            circle_radius=sampling_circle_r,
            num_sample_points=numb_sample_points,
        )

        # Analyze the sampled pixels values for the 4-part alternating pattern (9 +/- 1 segments)
        # e.g. [1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0] or [0 0 1 1 1 1 0 0 0 1 1 1 1 1 0 0] are good alternating patterns
        tolerance = 1 if numb_sample_points <= 18 else 2  #:SETTING:
        is_pattern_found = BitUtils.check_alternating_pattern(
            sampling_values, num_parts=4, tolerance=tolerance
        )

        # @TEST
        # TEST.plot_image(image_bin)
        # log.info(f"\nPattern found {is_pattern_found} in {sampling_values} with tolerance {tolerance}")
        
        return is_pattern_found
        

    def _read_hrct_target_id(
        self,
        target_image_gray: Image,
        code_dots_number: int,
        target_dots_color: str,
    ) -> Tuple[str, str, Union[str, None], int, bool]:
        """
        Reads and decodes the unique ID from the four digit matrices of the normalized,
        rotated, and binary HRCT target image.

        The HRCT ID is a two-digit number (d10 d01) encoded across four separate
        digit matrices (DM) positioned in the corners around the target center:
        - Upper-Right (UR): Encodes d10
        - Bottom-Left (BL): Encodes d10
        - Bottom-Right (BR): Encodes d01
        - Upper-Left (UL): Encodes d01

        The decoding process is as follows:
        1.  **Matrix Clipping:** The image is clipped four times, once for each digit matrix,
            based on predefined settings for size, gap, and offset (`HRCTDecoderSettings`).
        2.  **Digit Reading:** The `_read_digit` method is called for each of the four
            clipped images to determine the corresponding digit (0-9). If a digit cannot
            be read, a placeholder character ('X') is returned.
        3.  **Consistency Check and Consolidation:**
            - The two readings for **d10** (UR and BL) are compared. If they match, that
            digit is confirmed. If they conflict, the resulting d10 is 'X' (undecoded).
            If only one reading is successful, it is used for d10.
            - The same logic is applied to consolidate the two readings for **d01** (BR and UL).
        4.  **Final ID Construction:**
            - If both d10 and d01 are successfully determined (not 'X'), the final
            target ID is the concatenated two-digit string (e.g., "12").
            - If either d10 or d01 is 'X', the raw four-digit string (e.g., "1X2X")
            from the individual matrix readings is returned. This allows subsequent
            processing steps to manage partially or completely undecoded targets.

        Args:
            target_image_gray (Image): The normalized, scaled, rotated, and gray-scale
                                        image containing the full HRCT pattern.
            code_dots_number (int): The number of code dots in each digit matrix (currently
                                    only 9 is supported).
            target_dots_color (str): The color of the dots in the target (e.g., 'black').

        Returns:
            Tuple[str, Union[str, None], int, bool]: A tuple containing four elements:
                1. **str**: The decoded HRCT target ID. This is either a two-digit ID (e.g., "12")
                   if fully decoded, or a four-digit string containing 'X' placeholder
                   characters if decoding was partial or complete failure (e.g., "1X2X").
                2. **str**: The raw HRCT target ID read from the four digit matrices
                   before consolidation (e.g., "1X2X").
                3. **Union[str, None]**: The threshold mode used (e.g., 'mean') or **None**
                   if an alternative, explicit threshold value was used.
                4. **int**: The threshold value (0-255) used to convert the image to binary.
                5. **bool**: A flag, **True** if alternative thresholding was used, **False** otherwise.
        """
        # --------------------------------------------------------------------------
        # --- 1. Check the input arguments ---
        
        # Check the number of dots
        if code_dots_number != 9:
            raise ValueError(
                f"Unsupported number '{code_dots_number}' for code dots of HRCT target. "
                f"For now, only 9 dots HRCT target is supported."
            )
        

        # --------------------------------------------------------------------------
        # --- 2. Preprocess the target image to binary (main thresholding pass) ---

        # Preprocess the target image to binary
        main_threshold_mode = "mean"  #:TRICKY: Use 'mean' as thresholding mode at this step
        main_target_image_bin, main_threshold_value = ImageUtils.preprocess_to_binary_image(
            target_image_gray,
            remove_noise=True,
            noise_kernel_size=3,  #:SETTING:
            threshold_mode=main_threshold_mode,
            threshold_value=None,
        )

        # Read the target ID in one pass
        main_target_id, main_target_id_raw = self._read_hrct_target_id__one_pass(
            main_target_image_bin, code_dots_number, target_dots_color
        )   
        # log.info(f"TEST: Target ID_raw = {main_target_id_raw}; #using threshold value {main_threshold_value} of the main pass ('{main_threshold_mode}').")
        # TEST.plot_histogram_from_image(target_image_gray, main_threshold_value, title="HRCT Target Image Histogram")

        # If the target is decoded properly then return it
        x_char = HRCTDecoderSettings.DECODING_TARGET_ID_X_CHAR  
        if main_target_id.count(x_char) == 0:
            # Fully decoded target ID (e.g., "12")
            # log.info(f"Found HRCT Target: ID = {main_target_id}; #using threshold value {main_threshold_value}.") # Log the found target

            # Return the successfully decoded target ID using main thresholding
            return(main_target_id, main_target_id_raw, main_threshold_mode, main_threshold_value, False) # False = Indicate that main thresholding was used
        

        # --------------------------------------------------------------------------
        # --- 3. Preprocess the target image to binary (alternative thresholding passes) ---
        # Partially or fully undecoded target ID (e.g., "1X2X" or "XX")
        # Try with alternative thresholding values
        alt_threshold_values = ListUtils.generate_alternating_sequence(
            start_value=main_threshold_value,
            step_value=5, #:SETTING:
            bound_down_value=0,
            bound_up_value=255,
        )

        prev_target_id_raw_down = main_target_id_raw # contains the previous decoded digits in the main pass or previous alternative passes.
        prev_target_id_raw_up = main_target_id_raw # contains the previous decoded digits in the main pass or previous alternative passes.
        count_empty_alternative_passes_down = 0 
        count_empty_alternative_passes_up = 0
        flag_break_alternative_decoding_down = False
        flag_break_alternative_decoding_up = False
        for alt_threshold_value in alt_threshold_values:
            if flag_break_alternative_decoding_down and flag_break_alternative_decoding_up:
                # Both downward and upward alternative decoding searches have been stopped.
                break

            # Preprocess the target image to binary
            alt_target_image_bin, _alt_threshold_value = ImageUtils.preprocess_to_binary_image(
                target_image_gray,
                remove_noise=True,
                noise_kernel_size=3,  #:SETTING:
                threshold_mode=None,
                threshold_value=alt_threshold_value,
            )

            # Read the target ID in an alternative pass
            alt_target_id, alt_target_id_raw = self._read_hrct_target_id__one_pass(
                alt_target_image_bin, code_dots_number, target_dots_color
            )  
            # log.info(f"TEST: Target ID_raw = {alt_target_id_raw}; #using threshold value {alt_threshold_value} of an alternative pass.")
            

            if alt_threshold_value < main_threshold_value:
                if alt_target_id_raw == "0000" or alt_target_id == x_char*4: #e.g. "xxxx"
                    count_empty_alternative_passes_down += 1
                else:
                    count_empty_alternative_passes_down = 0
            else:
                if alt_target_id_raw == "0000" or alt_target_id == x_char*4: #e.g. "xxxx"
                    count_empty_alternative_passes_up += 1
                else:
                    count_empty_alternative_passes_up = 0

            # Check if too many empty alternative passes have occurred in a row
            if count_empty_alternative_passes_down >= 5: #:SETTING:
                flag_break_alternative_decoding_down = True
                continue

            if count_empty_alternative_passes_up >= 5: #:SETTING:
                flag_break_alternative_decoding_up = True
                continue            

            # Check if the alternative passes could successfully decode some of the undecoded digits
            if alt_target_id == "00" or alt_target_id == "0":
                # The alternative passes could not decode properly the target ID when the ID = 0 is reached.
                # Break the alternative decoding search
                if alt_threshold_value < main_threshold_value:
                    flag_break_alternative_decoding_down = True
                else: 
                     flag_break_alternative_decoding_down = True
                continue
            
            # Check for conflicts in the digit decoding between the current alternative pass and the previous passes
            if alt_threshold_value < main_threshold_value:
                prev_target_id_raw = prev_target_id_raw_down
            else:
                prev_target_id_raw = prev_target_id_raw_up

            flag_break_alternative_decoding = False
            for ii in range(len(prev_target_id_raw)):
                # Check if the current alternative pass decodes with different value a digit, 
                # that was previously decoded in the main pass or previous alternative passes.
                if (prev_target_id_raw[ii] != x_char # the digit is decoded in the previous pass
                    and alt_target_id_raw[ii] != x_char # the digit is decoded in the current alternative pass
                    and prev_target_id_raw[ii] != alt_target_id_raw[ii]): # the digits are different
                        # Conflict in the digit decoding. 
                        # Break the alternative decoding search
                        flag_break_alternative_decoding = True 
                        break
                
                # Update the digit of the curr_target_id_raw with the newly decoded digits from the alt_target_id_raw
                if prev_target_id_raw[ii] == x_char and alt_target_id_raw[ii] != x_char:
                    if alt_threshold_value < main_threshold_value:
                        prev_target_id_raw_down = StringUtils.replace_char_at_index(prev_target_id_raw_down, alt_target_id_raw, ii)
                    else:
                        prev_target_id_raw_up = StringUtils.replace_char_at_index(prev_target_id_raw_up, alt_target_id_raw, ii)

            # Stop the search with alternative passes.
            if flag_break_alternative_decoding:
                # Break the alternative decoding search
                if alt_threshold_value < main_threshold_value:
                    flag_break_alternative_decoding_down = True
                else: 
                     flag_break_alternative_decoding_up = True
                continue

            # Check if the target is decoded properly
            if alt_target_id.count(x_char) == 0:
                # Log the found and decoded target
                log.info(f"Found HRCT Target: ID = {alt_target_id}; #using threshold value {alt_threshold_value} of an alternative pass.")

                # Return the successfully decoded target ID using alternative thresholding
                return (alt_target_id, alt_target_id_raw, None, alt_threshold_value, True)  # True = Indicate that alternative thresholding was used

            
        # --------------------------------------------------------------------------
        # --- 4. Return the partially or fully undecoded target ID (main thresholding pass) ---

        # Return the target ID from the main pass (partially or fully undecoded)
        log.info(f"Found HRCT Target: ID = {main_target_id}; #using threshold value {main_threshold_value}.")
        return (main_target_id, main_target_id_raw, main_threshold_mode, main_threshold_value, False) # False = Indicate that main thresholding was used


    def _read_hrct_target_id__one_pass(
        self, target_image_bin: ImageBinary, code_dots_number: int, target_dots_color: str
    ) -> Tuple[str, str]:
        """
        Reads and decodes the unique ID from the four digit matrices of the normalized,
        rotated, and binary HRCT target image in one pass.  

        Args:
            target_image_bin (ImageBinary): The normalized, scaled, rotated, and binary 
                                            image containing the full HRCT pattern.     
            code_dots_number (int): The number of code dots in each digit matrix (currently
                                            only 9 is supported).
            target_dots_color (str): The color of the dots in the target (e.g., 'black').   
        Returns:
            str: The two-digit HRCT target ID (e.g., "12"), or a four-digit string
                containing placeholder 'X' characters if decoding failed (e.g., "1X2X"). 
            str: The raw four-digit string from the individual matrix readings (e.g., "1X2X").   
        """
        # Calculate the ratios #:TODO: calc the clip_ratio and offset_ratio_val in the HRCTDecoderSettings class
        target_square_size = HRCTDecoderSettings.TARGET_SQUARE_SIZE
        digit_matrix_size = HRCTDecoderSettings.DM9_SQUARE_SIZE
        gap_size = HRCTDecoderSettings.DM9_GAP_SIZE
        outer_margin_extra = HRCTDecoderSettings.DM9_OUTER_MARGIN_EXTRA
        clip_ratio = digit_matrix_size / target_square_size
        offset_ratio_val = (
            (target_square_size - outer_margin_extra - digit_matrix_size - gap_size) / 2
        ) / target_square_size


        # Clean the target image (variant-1)
        # Doesn't work, because the cleaning removes some dots as well  
        # TEST.plot_image(target_image_bin)
        # area_threshold_ratio = HRCTDecoderSettings.DM9_DOT_SHAPES_2_TARGET_AREA_RATIO #:SETTING:
        # target_image_bin = ImageUtils.clean_big_dark_areas_on_boundaries(
        #     target_image_bin, area_threshold_ratio=area_threshold_ratio
        # )
        # TEST.plot_image(target_image_bin)


        # Clean the target image (variant-2)
        # TEST.plot_image(target_image_bin)
        target_image_bin = ImageUtils.remove_long_black_lines(
            target_image_bin,
            ratio_threshold=2/15,  # :SETTING: Remove all black lines that are longer than 2/15 of the image size
            boundary_ratio=1/20,  # :SETTING: and are close (1/20) to the image boundaries.
        )
        kernel_morph_close = np.ones((3, 3), np.uint8) #:SETTING:
        target_image_bin = cv2.morphologyEx(
            target_image_bin, cv2.MORPH_CLOSE, kernel_morph_close
        )
        # TEST.plot_image(target_image_bin)


        # Read the upper-right coded digit
        # TEST.plot_image(target_image_bin)
        code_image_bin_ur = ImageUtils.clip_offset_square_by_ratio(
            target_image_bin,
            clip_ratio,
            center_offset_ratio=(offset_ratio_val, -offset_ratio_val),
        )
        digit_ur = self._read_digit(
            code_image_bin_ur, code_dots_number, target_dots_color
        )
        # log.info(f"Digit: {digit_ur}"); TEST.plot_image(code_image_bin_ur)

        # Read the bottom-right coded digit
        # TEST.plot_image(target_image_bin)
        code_image_bin_br = ImageUtils.clip_offset_square_by_ratio(
            target_image_bin,
            clip_ratio,
            center_offset_ratio=(offset_ratio_val, offset_ratio_val),
        )
        digit_br = self._read_digit(
            code_image_bin_br, code_dots_number, target_dots_color
        )
        # log.info(f"Digit: {digit_br}"); TEST.plot_image(code_image_bin_br)

        # Read the bottom-left coded digit
        # TEST.plot_image(target_image_bin)
        code_image_bin_bl = ImageUtils.clip_offset_square_by_ratio(
            target_image_bin,
            clip_ratio,
            center_offset_ratio=(-offset_ratio_val, offset_ratio_val),
        )
        digit_bl = self._read_digit(
            code_image_bin_bl, code_dots_number, target_dots_color
        )
        # log.info(f"Digit: {digit_bl}"); TEST.plot_image(code_image_bin_bl)

        # Read the upper-left coded digit
        # TEST.plot_image(target_image_bin)
        code_image_bin_ul = ImageUtils.clip_offset_square_by_ratio(
            target_image_bin,
            clip_ratio,
            center_offset_ratio=(-offset_ratio_val, -offset_ratio_val),
        )
        digit_ul = self._read_digit(
            code_image_bin_ul, code_dots_number, target_dots_color
        )
        # log.info(f"Digit: {digit_ul}"); TEST.plot_image(code_image_bin_ul)

        # Create target ID
        X = HRCTDecoderSettings.DECODING_TARGET_ID_X_CHAR
        ur_str = f"{digit_ur}" if not (digit_ur is None) else X
        br_str = f"{digit_br}" if not (digit_br is None) else X
        bl_str = f"{digit_bl}" if not (digit_bl is None) else X
        ul_str = f"{digit_ul}" if not (digit_ul is None) else X

    
        # Create target ID
        d10_1 = ur_str
        d10_2 = bl_str
        d01_1 = br_str
        d01_2 = ul_str

        if d10_1 != X and d10_2 != X:
            if d10_1 == d10_2:
                d10 = d10_1
            else:
                if d10_1 == "0":
                    d10 = d10_2
                elif d10_2 == "0":
                    d10 = d10_1
                else:
                    d10 = X
        elif d10_1 != X and d10_2 == X:
            d10 = d10_1
        elif d10_1 == X and d10_2 != X:
            d10 = d10_2
        elif d10_1 == X and d10_2 == X:
            d10 = X
        else:
            pass  # not possible

        if d01_1 != X and d01_2 != X:
            if d01_1 == d01_2:
                d01 = d01_1
            else:
                if d01_1 == "0":
                    d01 = d01_2
                elif d01_2 == "0":
                    d01 = d01_1
                else:
                    d01 = X
        elif d01_1 != X and d01_2 == X:
            d01 = d01_1
        elif d01_1 == X and d01_2 != X:
            d01 = d01_2
        elif d01_1 == X and d01_2 == X:
                d01 = X
        else:
            pass  # not possible

        if d10 != X and d01 != X:
            target_id = d10 + d01
        else:
            target_id = d10_1 + d01_1 + d10_2 + d01_2

        # Create the target raw string
        target_id_raw = d10_1 + d01_1 + d10_2 + d01_2

        # Return
        return (target_id, target_id_raw)

    def _read_digit(
        self,
        digit_matrix_image_bin: ImageBinary,
        dots_number: int,
        target_dots_color: str,
    ) -> int | None:
        """
        Reads and decodes the numerical digit (0-9) encoded within a single
        digit matrix image extracted from an HRCT target.

        This process involves:
        1.  **Image Cleaning:** Removes long, dark lines near the boundaries and applies
            morphological closing to smooth the dot shapes and close small gaps.
        2.  **Standard Scaling:** Rescales the cleaned digit matrix image to a fixed,
            standardized size (`DECODING_SCALDED_DM9_IMAGE_SIZE`) for consistent processing.
        3.  **Dot Contours Cleaning (Grid Alignment):** Uses contour analysis (`clean_contours_to_grid`)
            to filter out non-dot features within the matrix area, ensuring that only
            circular, appropriately sized dots aligned to the implicit grid remain. This
            step inverts and reinverts the image to operate on the dots (dark features) as contours.
        4.  **Matrix Sampling:** Divides the standardized, cleaned image into a grid (e.g., 3x3 for
            9 dots) and samples the pixels within each cell. It calculates the **black pixel ratio** in each cell, comparing it against empirically defined thresholds (`black_ratio_down_threshold`,
            `black_ratio_up_threshold`) to determine if a dot is present (1) or absent (0).
        5.  **Digit Decoding:** The resulting binary matrix (e.g., a 3x3 array of 0s and 1s) is
            passed to `get_digit_from_9dots_digit_matrix` to look up the corresponding
            decoded integer digit.

        Args:
            digit_matrix_image_bin (np.ndarray): The binary image containing a single
                                                digit matrix (e.g., the UL matrix).
            dots_number (int): The number of dots/cells in the matrix (e.g., 9 for a 3x3 grid).
            target_dots_color (str): The expected color of the dots.

        Returns:
            int | None: The decoded integer digit (0-9) if successful, or **None** if the
                        pattern does not match any known digit code.
        """
        # Set some variables for cleaning the digit matrix image
        kernel_size_morph_close = (
            HRCTDecoderSettings.DECODING_TARGET_ID_MORPH_CLOSE_KERNEL_SIZE
        )
        kernel_morph_close = np.ones(
            (kernel_size_morph_close, kernel_size_morph_close), np.uint8
        )

        # Clean the digit matrix image
        # TEST.plot_image(digit_matrix_image_bin)
        cln_digit_matrix_image_bin = ImageUtils.remove_long_black_lines(
            digit_matrix_image_bin,
            ratio_threshold=2/5,  # :SETTING: Remove all black lines that are longer than 2/5 of the image size
            boundary_ratio=1/10,  # :SETTING: and are close (1/10) to the image boundaries.
        )
        cln_digit_matrix_image_bin = cv2.morphologyEx(
            cln_digit_matrix_image_bin, cv2.MORPH_CLOSE, kernel_morph_close
        )
        # TEST.plot_image(cln_digit_matrix_image_bin)

        # Scale the input code image to standard size (the size should mod 2 and 3)
        scaled_image_size = int(HRCTDecoderSettings.DECODING_SCALDED_DM9_IMAGE_SIZE)
        std_digit_matrix_image_bin = cv2.resize(
            cln_digit_matrix_image_bin,
            (
                scaled_image_size,
                scaled_image_size,
            ),  # tuple of integers for square image (width, height)
            interpolation=cv2.INTER_LANCZOS4,
            # fx and fy arguments are not needed when dsize is explicitly provided
        )
        # TEST.plot_image(std_digit_matrix_image_bin)

        # Clean the binary image (i.e. clear the part of dot that are not in the dot's cell)
        # The other options for cleaning are: 'clean_contours_to_grid__centroid_based' and 
        # 'clean_contours_to_grid__proportion_based' function
        cln_std_digit_matrix_image_bin = ImageUtils.clean_contours_to_grid__hybrid_size_based(
            ImageUtils.invert_binary_image(std_digit_matrix_image_bin),
            num_cells=dots_number,
            min_area_threshold=5,  #:SETTING:
            min_circularity_threshold=1/10,  #:SETTING:
        )
        cln_std_digit_matrix_image_bin = ImageUtils.invert_binary_image(
            cln_std_digit_matrix_image_bin
        )
        # TEST.plot_image(cln_std_digit_matrix_image_bin)

        # Read the coded digit matrix
        ratio = HRCTDecoderSettings.DM9_DOT_SHAPE_2_CELL_AREA_RATIO
        ratio_tolerance = (
            HRCTDecoderSettings.DECODING_DIGIT_BLACK_RATIO_THRESHOLD_TOLERANCE
        )
        black_ratio_down_threshold = max(0.03, ratio - ratio_tolerance)  #:SETTING
        black_ratio_up_threshold = min(0.97, ratio + ratio_tolerance)  #:SETTING
        digit_matrix = ImageUtils.split_image_to_binary_matrix(
            cln_std_digit_matrix_image_bin,
            dots_number,
            black_ratio_down_threshold,
            black_ratio_up_threshold,
        )

        # Read the digit from the code digit matrix
        digit = HRCTDigitMatrixSettings.get_digit_from_9dots_digit_matrix(digit_matrix)
        # log.info(f"Black ration range: [{black_ratio_down_threshold}, {black_ratio_up_threshold}]")
        # log.info(f"Digit: {digit}\n{digit_matrix}"); TEST.plot_image(cln_std_digit_matrix_image_bin)

        # Return
        return digit

    @staticmethod
    def _adjust_scale_factor_outer_ellipse(
        target_center_ellipse: tuple, initial_scale_factor: float
    ) -> float:
        """
        Adjusts the outer ellipse scale factor based on the detected size of the
        central ellipse. This compensates for detection bias (e.g., shrinkage on small
        targets) by applying a piece-wise linear adjustment coefficient (adj_coeff).

        The adjustment logic is:
        - If size_px > 500, adj_coeff = 1.0 (no adjustment).
        - If size_px < 50, adj_coeff = 0.95 (maximum adjustment); i.e. remove 1 pixel
        - Linearly interpolates between 50 and 500 pixels.

        Args:
            target_center_ellipse: A tuple representing the detected ellipse
                                ((center_x, center_y), (axis_a, axis_b), angle).
            initial_scale_factor: The base scale factor to be adjusted.

        Returns:
            The adjusted scale factor for the outer ellipse.
        """
        # Constants for the piece-wise linear interpolation range and limits
        SIZE_UPPER_BOUND = 500
        SIZE_LOWER_BOUND = 50
        ADJ_COEFF_MIN = 0.975
        ADJ_COEFF_MAX = 1.0

        # Extract axis lengths from the ellipse tuple (ignoring center and angle)
        # The longest axis determines the size for scaling adjustment
        _, ellipse_axis_lengths, _ = target_center_ellipse
        target_size_px = max(ellipse_axis_lengths) * initial_scale_factor

        adj_coeff: float
        if target_size_px >= SIZE_UPPER_BOUND:
            # Large targets: No adjustment needed (adj_coeff = 1.0)
            adj_coeff = ADJ_COEFF_MAX
        elif target_size_px <= SIZE_LOWER_BOUND:
            # Small targets: Apply maximum adjustment
            adj_coeff = ADJ_COEFF_MIN
        else:
            size_range = SIZE_UPPER_BOUND - SIZE_LOWER_BOUND  # 450
            coeff_range = ADJ_COEFF_MAX - ADJ_COEFF_MIN  # 0.05

            adj_coeff = (
                ADJ_COEFF_MIN
                + coeff_range * (target_size_px - SIZE_LOWER_BOUND) / size_range
            )

        # Apply the calculated adjustment
        adjusted_scale_factor = initial_scale_factor * adj_coeff
        return adjusted_scale_factor


# --------------------------------------------
# Classes for generating HRCT targets
# --------------------------------------------


### Class for keeping settings for HRCT targets generation
class HRCTGeneratorSettings:
    @classmethod
    def load(cls):
        cls.LENGTH_UNIT = settings.get("hrct_generator.length_unit")  # e.g. cm

        cls.TARGET_SQUARE_SIZE = settings.get(
            "hrct_generator.target_square_size"
        )  # e.g. 18.0 cm
        cls.TARGET_CENTER_RADIUS_OUT = settings.get(
            "hrct_generator.target_center_radius_out"
        )  # e.g. 3.0 cm
        cls.TARGET_CENTER_RADIUS_IN = settings.get(
            "hrct_generator.target_center_radius_in"
        )  # e.g. 2.55 cm

        cls.DM9_DOT_SHAPE_SIZE = settings.get(
            "hrct_generator.dm9_dot_shape_size"
        )  # e.g. 1.8 cm
        cls.DM9_GAP_SIZE = settings.get("hrct_generator.dm9_gap_size")  # e.g. 0.45 cm
        cls.DM9_OUTER_MARGIN_EXTRA = settings.get(
            "hrct_generator.dm9_outer_margin_extra"
        )  # e.g. 0.0 cm
        cls.DM9_SQUARE_SIZE = 3 * (cls.DM9_DOT_SHAPE_SIZE + cls.DM9_GAP_SIZE)  # e.g. cm

        cls.DM9_DOT_SHAPE_TYPE = settings.get(
            "hrct_generator.dm9_dot_shape_type"
        )  # e.g. 'dot' or 'square'
        cls.TARGET_BOUNDARY_THICKNESS = settings.get(
            "hrct_generator.target_boundary_thickness"
        )  # e.g. 0.025 cm

        cls.DRAW_TARGET_ID_ON = settings.get(
            "hrct_generator.draw_target_id_on"
        )  # e.g. True
        cls.DRAW_TARGET_ID_FONT_SCALE = settings.get(
            "hrct_generator.draw_target_id_font_scale"
        )  # e.g. True
        cls.DRAW_TARGET_ID_MARGIN_OUT = settings.get(
            "hrct_generator.draw_target_id_margin_out"
        )  # e.g. 2 cm


### Class for generating HRCT targets in printable format ###
class HRCTGenerator(BTGenerator):
    # --- Settings ---
    @staticmethod
    def _load_settings():
        HRCTGeneratorSettings.load()

    # --- Generate targets images ---
    @staticmethod
    def generate_targets(
        target_type: str,
        res_folder_path: str,
        res_image_type: str,
        res_image_size_px: int,
        res_max_images_to_generate: int,
        progress_callback=None,
    ) -> int:
        """
        Generates a set of HRCT (High Resolution Coded Target) images based on the
        specified target type, image format, and quantity.

        This is a static factory method that handles the overall generation workflow:
        1.  **Validation:** Ensures the requested `target_type` (e.g., 'HRCT9') uses
            the supported number of code dots (9) and color ('black'). Unsupported
            requests are ignored.
        2.  **Quantity Determination:** Calculates the total number of target IDs to
            generate. If `res_max_images_to_generate` is -1, it defaults to generating
            all possible unique target images ($10^2 = 100$ IDs for 9-dot matrices).
        3.  **Output Format Dispatch:** Selects the appropriate internal generation
            function (`_generate_target_image_black`, `_generate_target_image_black_svg`)
            and the corresponding save function (`ImageUtils.save_image`,
            `ImageUtils.save_svg`, or `ImageUtils.save_pdf`) based on the requested
            `res_image_type` ('png', 'svg', 'pdf', etc.).
        4.  **Iteration and Saving:** Iterates through the required `target_id` range,
            generates the image content for the current ID, and saves it to the specified
            `res_folder_path` with the file name matching the ID (e.g., "12.png").
        5.  **Progress Tracking:** Emits progress updates via the optional `progress_callback`.

        Args:
            target_type (str): The specific type of HRCT to generate (e.g., 'HRCT_9dots_black').
            res_folder_path (str): The directory path where the generated files will be saved.
            res_image_type (str): The desired output image format ('png', 'svg', 'pdf', etc.).
            res_image_size_px (int): The target size in pixels for the generated raster
                                    images (used only for non-vector formats like PNG).
            res_max_images_to_generate (int): The maximum number of unique target IDs to
                                            generate. Use -1 to generate all possible IDs.
            progress_callback (Optional): A callback function for reporting generation progress.

        Returns:
            int: The total number of unique target images successfully generated and saved.
        """
        # Load settings
        HRCTGenerator._load_settings()

        # Get target type metadata
        _target_type_name, code_dots_number, target_dots_color = (
            TargetTypeUtils.get_metadata(target_type)
        )

        # Validate the color
        if target_dots_color != "black":
            log.warning(
                f"HRCT targets generation only supports 'black' coding dots. "
                f"Ignoring request for '{target_dots_color}'."
            )
            return 0

        # Validate the code_bit_number
        if code_dots_number != 9:
            log.warning(
                f"Invalid number of coding dots '{code_dots_number}' for HRCT generation. "
                f"The supported number should be '9'. Request ignored."
            )
            return 0

        # Get number of ids for generation
        max_ids_number = res_max_images_to_generate
        if res_max_images_to_generate == -1:
            max_ids_number = (
                code_dots_number + 1
            ) ** 2  # i.e. generate all possible unique target images

        # Determine the correct generate and save functions
        if res_image_type.lower() in ImageUtils.SUPPORT_RASTER_IMAGE_TYPES:
            generate_target_image__func__ = HRCTGenerator._generate_target_image_black
            save_image__func__ = ImageUtils.save_image
        elif res_image_type.lower() == "svg":
            generate_target_image__func__ = (
                HRCTGenerator._generate_target_image_black_svg
            )
            save_image__func__ = ImageUtils.save_svg
        elif res_image_type.lower() == "pdf":
            generate_target_image__func__ = (
                HRCTGenerator._generate_target_image_black_svg
            )
            save_image__func__ = ImageUtils.save_pdf
        else:
            log.warning(f"Unsupported image type: {res_image_type}. Request ignored.")
            return 0

        # Generate targets images and save them
        for target_id in range(0, max_ids_number):
            if progress_callback:
                progress_callback.emit(100 * target_id / max_ids_number)

            # Generate target image
            target_image_content = generate_target_image__func__(
                target_id, code_dots_number, res_image_size_px
            )
            # TEST.plot_image(target_image)

            # Save target image
            target_image_path = os.path.join(
                res_folder_path, f"{target_id}.{res_image_type}"
            )
            save_image__func__(target_image_path, target_image_content)

        # Return the number of the generated target images
        return max_ids_number

    # --- Raster image generation (PNG, JPEG, ...) ---
    @staticmethod
    def _generate_target_image_black(
        target_id: int, code_dots_number: int, image_size: int
    ) -> Image:
        """
        Returns the generated HRCT target image (NumPy array) coded with the input target_id.
        """

        # --- 1. Setup Variables ---

        # Colors
        color_black = ColorUtils.COLOR_BLACK
        color_white = ColorUtils.COLOR_WHITE

        # Target id
        (tens_digit, units_digit) = HRCTDigitMatrixSettings.get_digits(target_id)
        tens_digit_matrix = HRCTDigitMatrixSettings.get_9dots_digit_matrix(tens_digit)
        units_digit_matrix = HRCTDigitMatrixSettings.get_9dots_digit_matrix(units_digit)

        # Create square blank white image
        image = ImageUtils.create_blank_image(image_size, image_size, color=color_white)

        # --- 2. Draw the boundary square ---
        boundary_thickness = HRCTGeneratorSettings.TARGET_BOUNDARY_THICKNESS
        if boundary_thickness > 0:
            boundary_thickness = int(
                np.round(
                    image_size
                    * (
                        HRCTGeneratorSettings.TARGET_BOUNDARY_THICKNESS
                        / HRCTGeneratorSettings.TARGET_SQUARE_SIZE
                    )
                )
            )
            HRCTGenerator._draw_boundary_square_lines(
                image, color_black, thickness=boundary_thickness
            )

        # --- 3. Draw target center ---
        target_center_r_out = image_size * (
            HRCTGeneratorSettings.TARGET_CENTER_RADIUS_OUT
            / HRCTGeneratorSettings.TARGET_SQUARE_SIZE
        )
        target_center_r_in = image_size * (
            HRCTGeneratorSettings.TARGET_CENTER_RADIUS_IN
            / HRCTGeneratorSettings.TARGET_SQUARE_SIZE
        )
        HRCTGenerator._draw_target_center_cross(
            image, target_center_r_out, target_center_r_in, color_black, color_white
        )

        # --- 4. Draws the four digit matrices ---
        # Check the code_dots_number
        try:
            M = math.isqrt(code_dots_number)
        except ValueError:
            raise ValueError(
                f"Error: code_dots_number ({code_dots_number}) is not appropriated for a perfect square."
            )

        # Set size in pixels
        outer_margin_default = image_size * (
            (HRCTGeneratorSettings.DM9_GAP_SIZE / 2.0)
            / HRCTGeneratorSettings.TARGET_SQUARE_SIZE
        )
        outer_margin_extra = image_size * (
            HRCTGeneratorSettings.DM9_OUTER_MARGIN_EXTRA
            / HRCTGeneratorSettings.TARGET_SQUARE_SIZE
        )
        outer_margin = outer_margin_default + outer_margin_extra
        matrix_size = image_size * (
            HRCTGeneratorSettings.DM9_SQUARE_SIZE
            / HRCTGeneratorSettings.TARGET_SQUARE_SIZE
        )
        dot_shape_size = image_size * (
            HRCTGeneratorSettings.DM9_DOT_SHAPE_SIZE
            / HRCTGeneratorSettings.TARGET_SQUARE_SIZE
        )
        shape_type = HRCTGeneratorSettings.DM9_DOT_SHAPE_TYPE  # 'dot' or 'square'

        # Draw the M x M matrix in all four corners

        # Upper Left (ul)
        HRCTGenerator._draw_digit_matrix_at_corner(
            image,
            units_digit_matrix,
            "ul",
            color_black,
            outer_margin,
            matrix_size,
            dot_shape_size,
            shape_type,
        )

        # Upper Right (ur)
        HRCTGenerator._draw_digit_matrix_at_corner(
            image,
            tens_digit_matrix,
            "ur",
            color_black,
            outer_margin,
            matrix_size,
            dot_shape_size,
            shape_type,
        )

        # Bottom Left (bl)
        HRCTGenerator._draw_digit_matrix_at_corner(
            image,
            tens_digit_matrix,
            "bl",
            color_black,
            outer_margin,
            matrix_size,
            dot_shape_size,
            shape_type,
        )

        # Bottom Right (br)
        HRCTGenerator._draw_digit_matrix_at_corner(
            image,
            units_digit_matrix,
            "br",
            color_black,
            outer_margin,
            matrix_size,
            dot_shape_size,
            shape_type,
        )

        # --- 5. Draw the target ID ---
        if HRCTGeneratorSettings.DRAW_TARGET_ID_ON:
            HRCTGenerator._draw_target_id_text(image, target_id, color_black)

        # --- 6. Return the target image ---
        return image

    @staticmethod
    def _draw_boundary_square_lines(image: Image, color, thickness):
        image_size = ImageUtils.get_square_image_size(image)
        cv2.rectangle(
            image,
            (0, 0),  # top_left
            (image_size - 1, image_size - 1),  # bottom_right
            color,
            thickness,
        )

    @staticmethod
    def _draw_target_center_cross(
        image: Image,
        ring_r_out: float,
        ring_r_in: float,
        color_black: tuple,
        color_white: tuple,
    ):
        """
        Draws the central black ring with the coded black cross inside the white area
        using the specified radius ratios.
        """
        # Calculate center point
        image_size = ImageUtils.get_square_image_size(image)
        center_x = center_y = int(image_size / 2)
        center = (center_x, center_y)

        # Outer radius of the black ring (outer_radius_ratio of the half of the image size)
        ring_r_out = int(ring_r_out)

        # Inner radius of the black ring (inner_radius_ratio of the outer radius)
        ring_r_in = int(ring_r_in)

        # Draw the outer black circle (filled)
        cv2.circle(image, center, ring_r_out, color_black, -1)

        # # Draw the inner white circle (filled) to create the black ring
        # cv2.circle(image, center, ring_r_in, color_white, -1)

        # Now, modify the white central circle (radius_ring_inner) to create the black cross
        axes_quadrant = (ring_r_in, ring_r_in)

        # Draw (Top Right, 0 to 90 degrees) - White
        cv2.ellipse(
            img=image,
            center=center,
            axes=axes_quadrant,
            angle=0,
            startAngle=90,
            endAngle=180,
            color=color_white,
            thickness=-1,
        )

        # Draw Quadrant 3 (Bottom Left) - White (The opposite one)
        cv2.ellipse(
            img=image,
            center=center,
            axes=axes_quadrant,
            angle=0,
            startAngle=270,
            endAngle=360,
            color=color_white,
            thickness=-1,
        )

    @staticmethod
    def _draw_digit_matrix_at_corner(
        image: Image,
        digit_matrix: list[list[int]],
        corner: str,
        dot_color: tuple,
        outer_margin: float,
        matrix_size: float,
        dot_shape_size: float,
        shape_type: str = "dot",  # New argument for dot or square
    ):
        """
        Draws an M x M matrix of shapes (dot or square) in one of the four corners
        ('ul', 'ur', 'bl', 'br') of the image, based on the provided digit_matrix.

        Args:
            image: The NumPy array image to draw on.
            digit_matrix: A list of lists (M x M matrix) where 1 means draw dot, 0 means skip.
            corner: The position for the matrix ('ul', 'ur', 'bl', 'br').
            dot_color: The BGR color tuple for the shapes.
            outer_margin: The outer margin (matrix to image edge).
            matrix_size: The matrix side length.
            dot_fill_ratio: Ratio of the dot diameter to the matrix cell size. (0.0 to 1.0)
            shape_type: Specifies the shape to draw: 'dot' (circle) or 'square' (filled rectangle).
        """

        # Calculate M (matrix size) from the input matrix
        M = len(digit_matrix)
        if M == 0 or any(len(row) != M for row in digit_matrix):
            raise ValueError(
                "Error: The digit matrix must be a non-empty square matrix (M x M)."
            )

        # Calculate dimensions
        image_size = ImageUtils.get_square_image_size(image)
        outer_margin = int(outer_margin)
        matrix_size = int(matrix_size)

        # Cell size (Dots Center-to-Center distance)
        dot_cell_size = matrix_size / M

        # Shape Size calculation
        shape_size = int(dot_shape_size)  # :OLD: int(dot_cell_size * dot_fill_ratio)

        # For 'dot', shape_size is the diameter, radius is half the size.
        dot_radius = max(1, int(shape_size / 2))

        # For 'square', shape_size is the side length.
        square_half_side = max(1, int(shape_size / 2))

        # Determine the starting coordinates for the matrix based on the corner
        if corner == "ul":  # Upper Left
            start_x = outer_margin
            start_y = outer_margin
        elif corner == "ur":  # Upper Right
            start_x = image_size - outer_margin - matrix_size
            start_y = outer_margin
        elif corner == "bl":  # Bottom Left
            start_x = outer_margin
            start_y = image_size - outer_margin - matrix_size
        elif corner == "br":  # Bottom Right
            start_x = image_size - outer_margin - matrix_size
            start_y = image_size - outer_margin - matrix_size
        else:
            raise ValueError(f"Error: Invalid corner position '{corner}'.")

        # Draw the M x M matrix
        for j in range(M):  # Rows (Y-coordinate, 0 to M-1)
            for i in range(M):  # Columns (X-coordinate, 0 to M-1)

                # Check the digit matrix value before drawing
                if digit_matrix[j][i] == 1:
                    # Calculate the center of the current shape
                    center_x_shape = int(start_x + dot_cell_size * (i + 0.5))
                    center_y_shape = int(start_y + dot_cell_size * (j + 0.5))

                    if shape_type == "dot":
                        # Draw the filled circle (dot)
                        cv2.circle(
                            image,
                            (center_x_shape, center_y_shape),
                            dot_radius,
                            dot_color,
                            -1,
                        )

                    elif shape_type == "square":
                        # Calculate corner points for the square
                        top_left = (
                            center_x_shape - square_half_side,
                            center_y_shape - square_half_side,
                        )
                        bottom_right = (
                            center_x_shape + square_half_side,
                            center_y_shape + square_half_side,
                        )

                        # Draw the filled rectangle (square)
                        cv2.rectangle(image, top_left, bottom_right, dot_color, -1)

                    else:
                        raise ValueError(
                            f"Error: Invalid shape_type '{shape_type}'. Must be 'dot' or 'square'."
                        )

    @staticmethod
    def _draw_target_id_text(image: Image, target_id: int, color_black):
        # 1. Process input variable
        image_size = ImageUtils.get_square_image_size(image)
        text = f"{target_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 2. Determine Font Scale based on image size to keep text proportional.
        # We aim for the text to take up about 90% of the image width.
        target_width_ratio = 0.1 * HRCTGeneratorSettings.DRAW_TARGET_ID_FONT_SCALE

        # Use a reference scale (e.g., 1.0) and thickness (e.g., 1) to measure text size.
        ref_scale = 1.0
        ref_thickness = 1
        (ref_width, ref_height), baseline = cv2.getTextSize(
            text, font, ref_scale, ref_thickness
        )

        # Calculate the required font_scale to make the text width equal to target_width_ratio * size
        target_text_width = image_size * target_width_ratio

        # Calculate font_scale to achieve the target width
        if ref_width > 0:
            font_scale = ref_scale * (target_text_width / ref_width)
        else:
            # Fallback for empty string or error
            font_scale = 1.0

        # Determine thickness proportionally to the image size (e.g., 1 unit per 150 pixels of size)
        font_thickness = max(1, int(image_size / 200))

        # 3. Get the final size of the text with the calculated font_scale and thickness
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )

        # Calculate text position (bottom-center)
        org_x = (image_size - text_width) // 2
        org_y = image_size - int(
            image_size
            * (
                HRCTGeneratorSettings.DRAW_TARGET_ID_MARGIN_OUT
                / HRCTGeneratorSettings.TARGET_SQUARE_SIZE
            )
        )
        org = (org_x, org_y)

        cv2.putText(
            img=image,
            text=text,
            org=org,
            fontFace=font,
            fontScale=font_scale,
            color=color_black,
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

    # --- Vector image generation (SVG, PDF) ---
    @staticmethod
    def _generate_target_image_black_svg(
        target_id: int,
        code_dots_number: int,
        res_image_size_px: int = None,  # Ignored for SVG, as size is determined by real-world settings
    ) -> svgwrite.Drawing:
        """
        Returns the generated HRCT target SVG (svgwrite.Drawing object)
        coded with the input target_id.
        """

        # --- 1. Setup Variables ---
        color_black = ColorUtils.COLOR_BLACK_HEX
        color_white = ColorUtils.COLOR_WHITE_HEX

        length_unit = HRCTGeneratorSettings.LENGTH_UNIT  # e.g. cm
        image_size = HRCTGeneratorSettings.TARGET_SQUARE_SIZE

        # Target id matrices
        (tens_digit, units_digit) = HRCTDigitMatrixSettings.get_digits(target_id)
        tens_digit_matrix = HRCTDigitMatrixSettings.get_9dots_digit_matrix(tens_digit)
        units_digit_matrix = HRCTDigitMatrixSettings.get_9dots_digit_matrix(units_digit)

        # Create SVG drawing object
        dwg = svgwrite.Drawing(
            size=(f"{image_size}{length_unit}", f"{image_size}{length_unit}"),
            viewBox=f"0 0 {image_size} {image_size}",
        )

        # Set background (whole image area) to white
        dwg.add(
            dwg.rect(insert=(0, 0), size=(image_size, image_size), fill=color_white)
        )

        # --- 2. Draw the boundary square ---
        boundary_thickness = HRCTGeneratorSettings.TARGET_BOUNDARY_THICKNESS
        if boundary_thickness > 0:
            HRCTGenerator._draw_boundary_square_lines_svg(
                dwg, color_black, thickness=boundary_thickness
            )

        # --- 3. Draw target center ---
        target_center_r_out = HRCTGeneratorSettings.TARGET_CENTER_RADIUS_OUT
        target_center_r_in = HRCTGeneratorSettings.TARGET_CENTER_RADIUS_IN
        HRCTGenerator._draw_target_center_cross_svg(
            dwg, target_center_r_out, target_center_r_in, color_black, color_white
        )

        # --- 4. Draws the four digit matrices ---
        # Check the code_dots_number
        try:
            M = math.isqrt(code_dots_number)
        except ValueError:
            raise ValueError(
                f"Error: code_dots_number ({code_dots_number}) is not appropriated for a perfect square."
            )

        # Set sizes (from settings)
        outer_margin = (
            HRCTGeneratorSettings.DM9_GAP_SIZE / 2.0
        ) + HRCTGeneratorSettings.DM9_OUTER_MARGIN_EXTRA
        matrix_size = HRCTGeneratorSettings.DM9_SQUARE_SIZE
        dot_shape_size = HRCTGeneratorSettings.DM9_DOT_SHAPE_SIZE
        shape_type = HRCTGeneratorSettings.DM9_DOT_SHAPE_TYPE  # 'dot' or 'square'

        # Draw the M x M matrix in all four corners
        # Note: The coordinates are relative to the top-left (0, 0)
        # Upper Left (ul)
        HRCTGenerator._draw_digit_matrix_at_corner_svg(
            dwg,
            units_digit_matrix,
            "ul",
            color_black,
            image_size,
            outer_margin,
            matrix_size,
            dot_shape_size,
            shape_type,
        )

        # Upper Right (ur)
        HRCTGenerator._draw_digit_matrix_at_corner_svg(
            dwg,
            tens_digit_matrix,
            "ur",
            color_black,
            image_size,
            outer_margin,
            matrix_size,
            dot_shape_size,
            shape_type,
        )

        # Bottom Left (bl)
        HRCTGenerator._draw_digit_matrix_at_corner_svg(
            dwg,
            tens_digit_matrix,
            "bl",
            color_black,
            image_size,
            outer_margin,
            matrix_size,
            dot_shape_size,
            shape_type,
        )

        # Bottom Right (br)
        HRCTGenerator._draw_digit_matrix_at_corner_svg(
            dwg,
            units_digit_matrix,
            "br",
            color_black,
            image_size,
            outer_margin,
            matrix_size,
            dot_shape_size,
            shape_type,
        )

        # --- 5. Draw the target ID ---
        if HRCTGeneratorSettings.DRAW_TARGET_ID_ON:
            HRCTGenerator._draw_target_id_text_svg(
                dwg, image_size, target_id, color_black
            )

        # --- 6. Return the SVG string ---
        return dwg.tostring()

    @staticmethod
    def _draw_boundary_square_lines_svg(
        dwg: svgwrite.Drawing, color: str, thickness: float
    ):
        # SVG's coordinates are in the physical unit (e.g. cm)
        target_size = float(dwg["viewBox"].split()[-1])

        # Draw a rectangle from (0, 0) to (size, size). Use stroke for the boundary.
        # The 'fill' is set to 'none' to avoid drawing over the contents.
        dwg.add(
            dwg.rect(
                insert=(0, 0),
                size=(target_size, target_size),
                fill="none",
                stroke=color,
                stroke_width=thickness,
            )
        )

    @staticmethod
    def _draw_target_center_cross_svg(
        dwg: svgwrite.Drawing,
        ring_r_out: float,
        ring_r_in: float,
        color_black: str,
        color_white: str,
    ):
        """
        Draws the central black ring with the coded black cross inside the white area
        using SVG circle and path elements.
        """
        target_size = float(dwg["viewBox"].split()[-1])
        center_x = center_y = target_size / 2.0
        center = (center_x, center_y)

        # 1. Draw the outer black circle (filled)
        dwg.add(dwg.circle(center=center, r=ring_r_out, fill=color_black))

        # 2. Draw the two white quadrants as closed path sectors to create the cross cutout.

        # Helper function to convert polar to cartesian coordinates (SVG angle convention: 0 is East)
        def polar_to_cartesian(cx, cy, r, angle_deg):
            angle_rad = math.radians(angle_deg)
            x = cx + r * math.cos(angle_rad)
            y = cy + r * math.sin(angle_rad)
            return x, y

        # --- Quadrant 1 / White ---
        # The segment needs to be closed by drawing a line back to the center (Z).
        start_q2_x, start_q2_y = polar_to_cartesian(
            center_x, center_y, ring_r_in, 90
        )  # Start at North (Top)
        end_q2_x, end_q2_y = polar_to_cartesian(
            center_x, center_y, ring_r_in, 180
        )  # End at West (Left)
        path_q2_d = (
            f"M {center_x},{center_y} "  # Move to Center
            f"L {start_q2_x},{start_q2_y} "  # Line to Start Point (90 deg)
            # A rx ry x-rot large-arc-flag sweep-flag x y
            f"A {ring_r_in},{ring_r_in} 0 0 1 {end_q2_x},{end_q2_y} "  # Arc to End Point (180 deg, sweep-flag=1 for CW from 90 to 180)
            f"Z"  # Close path (Line back to center)
        )
        dwg.add(dwg.path(d=path_q2_d, fill=color_white, stroke="none"))

        # --- Quadrant 2 / White ---
        start_q4_x, start_q4_y = polar_to_cartesian(
            center_x, center_y, ring_r_in, 270
        )  # Start at South (Bottom)
        end_q4_x, end_q4_y = polar_to_cartesian(
            center_x, center_y, ring_r_in, 360
        )  # End at East (Right/0 deg)
        path_q4_d = (
            f"M {center_x},{center_y} "  # Move to Center
            f"L {start_q4_x},{start_q4_y} "  # Line to Start Point (270 deg)
            # A rx ry x-rot large-arc-flag sweep-flag x y
            f"A {ring_r_in},{ring_r_in} 0 0 1 {end_q4_x},{end_q4_y} "  # Arc to End Point (360 deg, sweep-flag=1 for CW from 270 to 360)
            f"Z"  # Close path (Line back to center)
        )
        dwg.add(dwg.path(d=path_q4_d, fill=color_white, stroke="none"))

    @staticmethod
    def _draw_digit_matrix_at_corner_svg(
        dwg: svgwrite.Drawing,
        digit_matrix: list[list[int]],
        corner: str,
        dot_color: str,
        target_size: float,
        outer_margin: float,
        matrix_size: float,
        dot_shape_size: float,
        shape_type: str = "dot",
    ):
        """
        Draws an M x M matrix of shapes (dot or square) in one of the four corners
        of the SVG drawing.

        Note: All dimensions (outer_margin, matrix_size, dot_shape_size) are in the physical unit (e.g. cm).
        """
        M = len(digit_matrix)
        if M == 0 or any(len(row) != M for row in digit_matrix):
            raise ValueError(
                "Error: The digit matrix must be a non-empty square matrix (M x M)."
            )

        dot_cell_size = matrix_size / M

        # Shape Size calculation
        dot_circle_r = dot_shape_size / 2.0
        dot_square_half_side = dot_shape_size / 2.0

        # Determine the starting coordinates for the matrix based on the corner
        if corner == "ul":  # Upper Left
            start_x = outer_margin
            start_y = outer_margin
        elif corner == "ur":  # Upper Right
            start_x = target_size - outer_margin - matrix_size
            start_y = outer_margin
        elif corner == "bl":  # Bottom Left
            start_x = outer_margin
            start_y = target_size - outer_margin - matrix_size
        elif corner == "br":  # Bottom Right
            start_x = target_size - outer_margin - matrix_size
            start_y = target_size - outer_margin - matrix_size
        else:
            raise ValueError(f"Error: Invalid corner position '{corner}'.")

        # Draw the M x M matrix
        for j in range(M):  # Rows (Y-coordinate, 0 to M-1)
            for i in range(M):  # Columns (X-coordinate, 0 to M-1)

                if digit_matrix[j][i] == 1:
                    # Calculate the center of the current shape
                    center_x_shape = start_x + dot_cell_size * (i + 0.5)
                    center_y_shape = start_y + dot_cell_size * (j + 0.5)

                    if shape_type == "dot":
                        # Draw the filled circle (dot)
                        dwg.add(
                            dwg.circle(
                                center=(center_x_shape, center_y_shape),
                                r=dot_circle_r,
                                fill=dot_color,
                            )
                        )

                    elif shape_type == "square":
                        # Calculate top-left point for the rectangle
                        top_left_x = center_x_shape - dot_square_half_side
                        top_left_y = center_y_shape - dot_square_half_side

                        # Draw the filled rectangle (square)
                        dwg.add(
                            dwg.rect(
                                insert=(top_left_x, top_left_y),
                                size=(dot_shape_size, dot_shape_size),
                                fill=dot_color,
                            )
                        )

                    else:
                        raise ValueError(
                            f"Error: Invalid shape_type '{shape_type}'. Must be 'dot' or 'square'."
                        )

    @staticmethod
    def _draw_target_id_text_svg(
        dwg: svgwrite.Drawing,
        image_size: float,
        target_id: int,
        color_black: str,
    ):
        # Get text ID
        text = f"{target_id}"

        # Calculate text position (bottom-center)
        center_x = image_size / 2.0

        # Target text height
        text_height = image_size * 0.1 * HRCTGeneratorSettings.DRAW_TARGET_ID_FONT_SCALE

        # Place text near the bottom
        baseline_y = image_size - HRCTGeneratorSettings.DRAW_TARGET_ID_MARGIN_OUT
        dwg.add(
            dwg.text(
                text=text,
                insert=(center_x, baseline_y),
                fill=color_black,
                font_size=f"{text_height}",
                text_anchor="middle",
                font_family="Arial, sans-serif",
            )
        )


# --------------------------------------------
# Main execution block, if this script is executed directly
# Note. Use only for for development-time testing
# --------------------------------------------
if __name__ == "__main__":
    # Import tests.dev module
    from tests.dev import run_dev_test_find_hrct_targets
    from tests.dev import run_dev_test_generate_hrct_targets

    # Run dev test
    run_dev_test_find_hrct_targets()
    # run_dev_test_generate_hrct_targets()
