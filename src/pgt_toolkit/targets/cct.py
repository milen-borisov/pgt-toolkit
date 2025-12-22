"""
Implements the core logic (decoding, drawing and generation)
for Circular Coded Targets (CCT) with a single outer code ring.

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
from typing import List, Tuple
from PIL import Image
from typing import Optional

# Import internal modules that are from the pgt_toolkit folder
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))
from app.settings import settings
from app.logger import logger as log
from utils.image import ImageUtils
from utils.image import ImageUtils as TEST
from utils.image import Image, ImageBinary
from utils.math import MathUtils
from utils.bit import BitUtils
from utils.color import ColorUtils
from targets.target_data import TargetData, TargetTypeUtils
from targets.bt_drawer import BTDrawer
from targets.bt_decoder import BTDecoder, Contour, Hierarchy
from targets.bt_generator import BTGenerator


# --------------------------------------------
# Classes for drawing CCT target in image
# --------------------------------------------


### Class for keeping settings for CCT target in image ###
class CCTDrawerSettings:
    @classmethod
    def load(cls):
        # Colors
        cls.TARGET_CENTER_LINES_COLOR = ColorUtils.standardize_color_input(
            settings.get("cct_drawer.target_center_lines_color")
        )  # e.g. 'green'
        cls.TARGET_ID_TEXT_COLOR = ColorUtils.standardize_color_input(
            settings.get("cct_drawer.target_id_text_color")
        )  # e.g. 'red'


### Class for drawing CCT target in image ###
class CCTDrawer(BTDrawer):
    @staticmethod
    def _load_settings():
        CCTDrawerSettings.load()

    @staticmethod
    def draw_target(target: TargetData, image: Image) -> Image:
        """
        Draws the detected CCT target on the input image.

        The drawing includes:
        1.  **Concentric Ellipses:** Drawing the outer, middle, and inner rings of the
            target pattern, calculated by scaling the fitted outer ellipse. The thickness
            is scaled based on the image size.
        2.  **Target ID:** Placing the target's decoded ID (or undecoded reference)
            as text near the center point.

        @Note: This function modifies the input image (in-place).

        Args:
            target (TargetData): The object containing the target's geometric and
                                identifier information.
            image (Image): The OpenCV image array to draw upon.

        Returns:
            Image: The modified input image with the target drawn.
        """
        # Load settings
        CCTDrawer._load_settings()

        # Get length and scale factor
        (h, w) = ImageUtils.get_image_dimensions(image)
        image_length = (h + w) / 2.0
        scale_factor = image_length / 2000

        # Draw the fitted ellipses
        target_outer_ellipse = target.geometry_data
        middle_ring_scale_factor = (
            CCTDecoderSettings.TARGET_MIDDLE_RING_RADIUS_OUT
            / CCTDecoderSettings.TARGET_CENTER_RING_RADIUS_OUT
        )
        target_middle_ellipse = MathUtils.scale_ellipse(
            target_outer_ellipse, middle_ring_scale_factor
        )

        outer_ring_scale_factor = (
            CCTDecoderSettings.TARGET_OUTER_RING_RADIUS_OUT
            / CCTDecoderSettings.TARGET_CENTER_RING_RADIUS_OUT
        )
        target_center_ellipse = MathUtils.scale_ellipse(
            target_outer_ellipse, outer_ring_scale_factor
        )

        ellipse_color = CCTDrawerSettings.TARGET_CENTER_LINES_COLOR
        ellipse_thickness = round(1 * scale_factor)
        cv2.ellipse(
            image,
            target_outer_ellipse,
            color=ellipse_color,
            thickness=ellipse_thickness,
        )
        cv2.ellipse(
            image,
            target_middle_ellipse,
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
        text_color = CCTDrawerSettings.TARGET_ID_TEXT_COLOR
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
# Classes for decoding CCT targets
# --------------------------------------------


### Class for keeping settings for CCT targets decoder ###
class CCTDecoderSettings:
    @classmethod
    def load(cls):
        # Target geometry settings
        cls.TARGET_CENTER_RING_RADIUS_OUT = settings.get(
            "cct_decoder.target_center_ring_radius_out"
        )  # e.g. 1
        cls.TARGET_MIDDLE_RING_RADIUS_OUT = settings.get(
            "cct_decoder.target_middle_ring_radius_out"
        )  # e.g. 2
        cls.TARGET_OUTER_RING_RADIUS_OUT = settings.get(
            "cct_decoder.target_outer_ring_radius_out"
        )  # e.g. 3
        # cls.TARGET_SQUARE_SIZE = settings.get("cct_decoder.target_square_size") # e.g. 8

        # Target center classifications
        cls.TARGET_SCALED_IMAGE_SIZE = int(
            settings.get("cct_decoder.target_scaled_image_size")
        )  # e.g. 180 pixels
        cls.TARGET_CENTER_NUMB_SAMPLE_POINTS = settings.get(
            "cct_decoder.target_center_numb_sample_points"
        )  # e.g. 36


### Class for decoding CCT targets ###
class CCTDecoder(BTDecoder):
    """
    A subclass that implements the main logic for finding and decoding CCT target in an image.
    """

    # --- Settings ---
    @staticmethod
    def _load_settings():
        CCTDecoderSettings.load()

    # --- Contour filtering functions ---
    def _filter_candidate_target_contours(
        self, contours: List[Contour], hierarchy: Optional[Hierarchy]
    ) -> List[Contour]:
        return self._filter_candidate_target_contours__shape_circular(contours)

    # --- Target recognition and decoding functions ---
    def _extract_target_data(
        self, target_center_contour: Contour, image: Image
    ) -> Optional[TargetData]:
        """
        Tries to extract the full target data, including the decoded ID, from a given target
        center contour.

        This is a multi-step process that involves:
        1.  **Ellipses Fitting and Scaling:** Fits the center ellipse and scales it
            up to the outer ring size, and then scales it slightly further (using a
            sqrt(2) coefficient) to ensure the bounding box fully captures the area
            required for affine transformation.
        2.  **Normalization (Affine Warp):** Calls `_get_normalized_square_image_for_ellipse`
            to extract and warp the target's image region into a **perfect, upright circle** on a square background, compensating for perspective distortion and rotation.
        3.  **Image Preprocessing:** Clips, resizes the normalized image to a fixed
            pixel size (`TARGET_SCALED_IMAGE_SIZE`), and then applies gray-scaling and
            thresholding/erosion to create a standardized **binary image**.
        4.  **Initial Validation:** Checks the binary image against expected CCT patterns
            using `_is_cct_target`.
        5.  **Decoding:** Reads the bit pattern from the binary image using
            `_read_cct_target_id` to determine the unique **Target ID**.
        6.  **Packaging:** Compiles the original geometry data (the center ellipse)
            and the decoded ID into a final **TargetData** object.

        Args:
            target_center_contour (Contour): The raw contour of the target's central
                                                feature (the black dot/crosshair).
            image (Image): The source image (OpenCV format) containing the target.

        Returns:
            Optional[TargetData]: The fully populated TargetData object if decoding
                                was successful, otherwise 'None'.
        """
        # --------------------------------------------------------------------------
        # --- 1. Preprocessing, Normalization, and Initial Validation ---

        # Fit ellipses for the target center area (1x), code area (2x), and the outer target boundary (3x)
        target_center_ellipse = cv2.fitEllipse(target_center_contour)
        outer_ring_scale_factor = (
            CCTDecoderSettings.TARGET_OUTER_RING_RADIUS_OUT
            / CCTDecoderSettings.TARGET_CENTER_RING_RADIUS_OUT
        )
        target_outer_ellipse = MathUtils.scale_ellipse(
            target_center_ellipse, scale_factor=outer_ring_scale_factor
        )
        outer_ellipse_scale_coeff_big = math.sqrt(
            2
        )  #:TRICKY: make all pixels of the normalized target image to be from the original image
        target_outer_ellipse_big = MathUtils.scale_ellipse(
            target_outer_ellipse, outer_ellipse_scale_coeff_big
        )

        # Get normalized scaled image (a normalized squared image of the target in perfect upright circle)
        norm_target_image = self._get_normalized_square_image_for_ellipse(
            target_outer_ellipse_big, image
        )
        if norm_target_image is None:
            return None
        # TEST.plot_image(norm_target_image)

        # Left the target only
        norm_target_image = ImageUtils.clip_center_square_by_ratio(
            norm_target_image, 1 / outer_ellipse_scale_coeff_big
        )
        # TEST.plot_image(norm_target_image)

        # Scale the normalized (affined transformed) target image to normalized square image with fixed size
        scaled_target_image_size = (
            CCTDecoderSettings.TARGET_SCALED_IMAGE_SIZE
        )  # e.g., 180
        scaled_target_image = cv2.resize(
            norm_target_image,
            (
                scaled_target_image_size,
                scaled_target_image_size,
            ),  # tuple of integers for square image (width, height)
            interpolation=cv2.INTER_LANCZOS4,
            # fx and fy arguments are not needed when dsize is explicitly provided
        )
        # TEST.plot_image(scaled_target_image)

        # Process (Grayscale, Thresholding, Erosion) on the scaled target image; return a binary image 
        target_center_threshold_mode = self._image_threshold_mode
        target_center_threshold_value = self._image_threshold_value
        if target_center_threshold_mode == "adaptive":
            target_center_threshold_mode = "otsu" #:TRICKY: Use Otsu for adaptive thresholding in this step
            target_center_threshold_value = None
        proc_target_image_bin, _threshold = ImageUtils.preprocess_to_binary_image(
            scaled_target_image,
            remove_noise=True,
            noise_kernel_size=3, #:SETTING:
            threshold_mode=target_center_threshold_mode,
            threshold_value=target_center_threshold_value,
        )
        # TEST.plot_image(proc_target_image_bin)

        # Validate if the target image is CCT target
        if not self._is_cct_target(proc_target_image_bin):
            return None

        # --------------------------------------------------------------------------
        # --- 2. Decode the Target ID ---

        # Read target ID
        target_id = self._read_cct_target_id(
            proc_target_image_bin, self.target_code_number, self.target_color
        )

        # --------------------------------------------------------------------------
        # --- 3. Package Decoded Target Data for Returning ---

        # Create TargetData object for returning
        target_data = TargetData(
            self._image_file_name,
            self.target_type,
            target_id,
            target_id, # Raw ID is same as decoded ID for CCT
            "ellipse",
            target_center_ellipse,
            self._image_threshold_mode,
            self._image_threshold_value,
        )

        # Log the found target
        log.info(f"Found CCT Target: ID = {target_id}")

        # --------------------------------------------------------------------------
        # --- 4. Return Final Target Data ---

        # Return
        return target_data

    def _is_cct_target(self, processed_target_image_bin: ImageBinary) -> bool:
        """
        Checks if the input binary image is an image of a CCT target by sampling
        three concentric circles and checking pixel sums.
        """
        # --------------------------------------------------------------------------
        # --- 1. Setup Constants and Dimensions ---

        # Get dimensions and ensure the image is square
        image_height, image_width = ImageUtils.get_image_dimensions(processed_target_image_bin)
        if image_height != image_width:
            raise ValueError(
                f"Input image to '_is_cct_target' must be square, but dimensions are {image_height} x {image_width}."
            )
        target_diameter = image_width

        # Center coordinates (float)
        center_x = target_diameter / 2.0
        center_y = target_diameter / 2.0

        # Calculate the fundamental unit radius (R_unit); i.e. the outer radius of the target center ring
        r_unit = (target_diameter / 2.0) * (
            CCTDecoderSettings.TARGET_CENTER_RING_RADIUS_OUT
            / CCTDecoderSettings.TARGET_OUTER_RING_RADIUS_OUT
        )

        # Define the three sampling radii (in the middle of the segments)
        center_r_sf = CCTDecoderSettings.TARGET_CENTER_RING_RADIUS_OUT / 2.0
        middle_r_sf = (
            CCTDecoderSettings.TARGET_MIDDLE_RING_RADIUS_OUT
            + CCTDecoderSettings.TARGET_CENTER_RING_RADIUS_OUT
        ) / 2.0
        outer_r_sf = (
            CCTDecoderSettings.TARGET_OUTER_RING_RADIUS_OUT
            + CCTDecoderSettings.TARGET_MIDDLE_RING_RADIUS_OUT
        ) / 2.0

        center_ring_r = center_r_sf * r_unit  # e.g. 0.5 * r_unit
        middle_ring_r = middle_r_sf * r_unit  # e.g. 1.5 * r_unit
        outer_ring_r = outer_r_sf * r_unit  # e.g. 2.5 * r_unit

        # --------------------------------------------------------------------------
        # --- 2. Circular Sampling and Summation ---

        numb_sample_points = CCTDecoderSettings.TARGET_CENTER_NUMB_SAMPLE_POINTS

        # Sample the innermost ring (Expected Solid Color)
        pixels_center_ring = self._read_pixel_values_on_circle(
            image_bin=processed_target_image_bin,
            circle_center_x=center_x,
            circle_center_y=center_y,
            circle_radius=center_ring_r,
            num_sample_points=numb_sample_points,
        )

        # Sample the middle ring (Expected Solid Color)
        pixels_middle_ring = self._read_pixel_values_on_circle(
            image_bin=processed_target_image_bin,
            circle_center_x=center_x,
            circle_center_y=center_y,
            circle_radius=middle_ring_r,
            num_sample_points=numb_sample_points,
        )

        # Sample the outer ring (Expected Code Pattern)
        pixels_outer_ring = self._read_pixel_values_on_circle(
            image_bin=processed_target_image_bin,
            circle_center_x=center_x,
            circle_center_y=center_y,
            circle_radius=outer_ring_r,
            num_sample_points=numb_sample_points,
        )

        # Calculate the sum of pixel values for each ring
        sum_center_ring = np.sum(pixels_center_ring)
        sum_middle_ring = np.sum(pixels_middle_ring)
        sum_outer_ring = np.sum(pixels_outer_ring)

        # --------------------------------------------------------------------------
        # --- 3. Target Structure Validation ---

        # CCT target structure 1: White Center, Black Middle Ring, White Code Ring (Default: Black Target)
        # Expected: Center=White (1s), Middle=Black (0s), Outer=Pattern
        if (
            self.target_color == "white"
            and sum_center_ring
            == numb_sample_points  # Center must be entirely white (pixel value 1)
            and sum_middle_ring
            == 0  # Middle ring must be entirely black (pixel value 0)
            and sum_outer_ring
            >= 2  # Outer code ring must have at least 2 white pixels (i.e., not solid black)
        ):
            return True

        # CCT target structure 2: Black Center, White Middle Ring, Black Code Ring (Default: White Target)
        # Expected: Center=Black (0s), Middle=White (1s), Outer=Pattern
        elif (
            self.target_color == "black"
            and sum_center_ring == 0  # Center must be entirely black (pixel value 0)
            and sum_middle_ring
            == numb_sample_points  # Middle ring must be entirely white (pixel value 1)
            and sum_outer_ring
            <= numb_sample_points
            - 2  # Outer code ring must have at least 2 black pixels (i.e. <= 34 white)
        ):
            return True

        return False

    def _read_cct_target_id(
        self,
        target_image_bin: ImageBinary,
        code_bit_number: int,
        target_center_color: str,
    ) -> int:
        """
        Reads and decodes the unique target ID from the outer code ring of the CCT target image.

        The method performs sub-degree sampling around the code ring to minimize
        noise and angular misalignment, averaging the results from several rotations.
        """

        # --------------------------------------------------------------------------
        # --- 1. Setup Constants and Dimensions ---

        # N: The number of code bits on the outer code ring.
        BIT_COUNT = code_bit_number  # N

        # Get dimensions
        image_height, image_width = ImageUtils.get_image_dimensions(target_image_bin)

        # Guardrail: The processed image must be square
        if image_height != image_width:
            raise ValueError(
                f"Input image to '_read_target_id' must be square, but dimensions are {image_height} x {image_width}."
            )
        target_diameter = image_width

        # Center coordinates
        center_x = target_diameter / 2.0
        center_y = target_diameter / 2.0

        # Unit radius (R_unit) and sampling radius (2.5 * R_unit)
        r_unit = (target_diameter / 2.0) * (
            CCTDecoderSettings.TARGET_CENTER_RING_RADIUS_OUT
            / CCTDecoderSettings.TARGET_OUTER_RING_RADIUS_OUT
        )
        outer_r_sf = (
            CCTDecoderSettings.TARGET_OUTER_RING_RADIUS_OUT
            + CCTDecoderSettings.TARGET_MIDDLE_RING_RADIUS_OUT
        ) / 2.0

        # The code area radius is 2.5 times the unit radius, corresponding to the middle of the outer code ring.
        outer_code_ring_r = outer_r_sf * r_unit  # e.g. 2.5 * r_unit

        # The number of rotation steps to average
        rotation_step_count = int(
            360 / BIT_COUNT
        )  # e.g. 45, if N=8; @note 360/BIT_COUNT will give 1 degree for the rotation_step_deg angle

        # Initialize array to store decoded, CANONICAL bits from all rotation steps
        code_bits_matrix = np.zeros(
            (rotation_step_count, BIT_COUNT)
        )  # e.g. (45x8), if N=8

        # --------------------------------------------------------------------------
        # --- 2. Vectorized Coordinate Calculation (Rotation Sampling) ---

        # The step angle for sampling bits
        bit_step_deg = 360.0 / BIT_COUNT  # e.g. 45 degrees, if N=8

        # The sub-degree angle for rotation shift
        rotation_step_deg = bit_step_deg / rotation_step_count  # e.g. 1 degree

        # Array of bit indices to sample (0, 1, ..., N-1) * bit_step_deg
        base_bit_angles_deg = (
            np.arange(BIT_COUNT) * bit_step_deg
        )  # e.g. [ 0.  45.  90. 135. 180. 225. 270. 315.]

        # Iterate over each rotation shift (j)
        for j in range(rotation_step_count):
            # Calculate the total angular shift for the current rotation
            angular_shift_deg = (
                j * rotation_step_deg
            )  # e.g. each iterations: {0., 8., 12., ..., 352.} #45 times

            # Combine the base bit angles with the rotational shift
            total_angles_deg = (
                base_bit_angles_deg + angular_shift_deg
            )  # e.g. each iteration: {[  0.  45.  90. 135. 180. 225. 270. 315.], [  1.  46.  91. 136. 181. 226. 271. 316.], ..., [ 44.  89. 134. 179. 224. 269. 314. 359.]} #45 times
            total_angles_rad = np.deg2rad(total_angles_deg)

            # Calculate the direction vectors (cos/sin) for all sampling points
            xs = np.cos(total_angles_rad)
            ys = np.sin(total_angles_rad)

            # Calculate coordinates for the code ring (2.5 * R_unit)
            sampling_points_x = np.round(outer_code_ring_r * xs + center_x).astype(int)
            sampling_points_y = np.round(outer_code_ring_r * ys + center_y).astype(int)

            # Vectorized Pixel Sampling
            try:
                # :IMPORTANT: NumPy indexing is image[rows (Y), columns (X)]
                sampling_pixels = target_image_bin[sampling_points_y, sampling_points_x]
            except IndexError:
                log.warning(
                    f"Rotation step {j} caused sampling points outside array boundaries. Skipping."
                )
                continue  # Skip this iteration if coordinates are out of bounds

            ## Canonicalization (Rotation Minimization)
            # Convert the raw sampled array of bits (0s and 1s) to a standard Python list
            curr_code_bits = sampling_pixels.tolist()  # e.g. [0, 1, 1, 1, 0, 1, 0, 0]

            # Find ID for the current bit sequence;
            # i.e., find the minimum integer value across all bit rotations of the current bit sequence
            temp_code_id = BitUtils.bits_to_id(curr_code_bits, BIT_COUNT)  # e.g. 23

            # Convert back to the CANONICAL bit sequence (i.e. this with the minimum integer value)
            canonical_code_bits = BitUtils.int_to_bits(
                temp_code_id, BIT_COUNT
            )  # e.g. [1, 1, 1, 0, 1, 0, 0, 0]

            # Store the canonical bit sequence in the matrix
            code_bits_matrix[j, :] = canonical_code_bits

        # --------------------------------------------------------------------------
        # --- 3. Final Decoding and Binarization ---

        # Calculate the average BIT VALUE (0 or 1) for each bit column across all rotations
        # Since we stored canonical bits, the mean represents the confidence in that bit position.
        code_bits_avg = np.mean(
            code_bits_matrix, axis=0
        )  # e.g. [1.   0.97777778 0.   0.  0.97777778 0.   0.  0.]

        # Binarization: Convert mean values to final bits (0 or 1)
        final_code_bits = np.where(
            code_bits_avg > 0.5, 1, 0
        ).tolist()  # e.g. [1, 1, 1, 0, 1, 0, 0, 0] -> id=23

        # Apply inversion based on target center color (Original Logic)
        if target_center_color == "black":
            final_code_bits = BitUtils.flip_bits(final_code_bits)

        # Call the decoding function to convert the final bit sequence to the integer ID
        final_id = BitUtils.bits_to_id(final_code_bits, len(final_code_bits))

        # --------------------------------------------------------------------------
        # --- 4. Return Final Target ID ---

        return final_id


# --------------------------------------------
# Classes for generating CCT targets
# --------------------------------------------


### Class for keeping settings for CCT targets generator ###
class CCTGeneratorSettings:
    @classmethod
    def load(cls):
        # Target geometry settings
        cls.LENGTH_UNIT = settings.get("cct_generator.length_unit")  # e.g. cm

        cls.TARGET_CENTER_RING_RADIUS_OUT = settings.get(
            "cct_generator.target_center_ring_radius_out"
        )  # e.g. 1 cm
        cls.TARGET_MIDDLE_RING_RADIUS_OUT = settings.get(
            "cct_generator.target_middle_ring_radius_out"
        )  # e.g. 2 cm
        cls.TARGET_OUTER_RING_RADIUS_OUT = settings.get(
            "cct_generator.target_outer_ring_radius_out"
        )  # e.g. 3 cm
        cls.TARGET_SQUARE_SIZE = settings.get(
            "cct_generator.target_square_size"
        )  # e.g. 8 cm

        cls.TARGET_BOUNDARY_THICKNESS = settings.get(
            "cct_generator.target_boundary_thickness"
        )  # e.g. 0.025 cm

        cls.GENERATE_CENTRAL_POINT_ON = settings.get(
            "cct_generator.generate_central_point_on"
        )  # e.g. True

        cls.DRAW_TARGET_ID_ON = settings.get(
            "cct_generator.draw_target_id_on"
        )  # e.g. True
        cls.DRAW_TARGET_ID_FONT_SCALE = settings.get(
            "cct_generator.draw_target_id_font_scale"
        )  # e.g. 1
        cls.DRAW_TARGET_ID_MARGIN_OUT = settings.get(
            "cct_generator.draw_target_id_margin_out"
        )  # e.g. 0.2 cm


### Class for generating CCT targets in printable format ###
class CCTGenerator(BTGenerator):
    # --- Settings ---
    @staticmethod
    def _load_settings():
        CCTGeneratorSettings.load()

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
        Generates a set of unique CCT (Circular Coded Target) images based on the
        specified target type, image size, and maximum generation count.

        This method handles the high-level logic for CCT generation, including:
        1.  **Validation:** Checks the target type's color and bit number against
            CCT generation limits.
        2.  **ID Generation:** Iterates from 0 up to the maximum possible ID value
            (based on bit number) or `res_max_images_to_generate`, collecting a list of
            unique target ID bit sequences.
        3.  **Output Selection:** Determines the appropriate generation and saving
            functions (`_generate_target_image_black`, `_generate_target_image_black_svg`)
            based on the requested `res_image_type` (e.g., PNG, JPG, SVG, PDF).
        4.  **Image Creation & Saving:** Loops through the collected ID bit sequences,
            generates the corresponding target image content, and saves the file to
            `res_folder_path` with the filename corresponding to the decoded integer ID.

        Args:
            target_type (str): The specific CCT target type (e.g., 'CCT_12bit_black').
            res_folder_path (str): The directory path where the generated images will be saved.
            res_image_type (str): The desired output file type ('png', 'jpg', 'svg', 'pdf').
            res_image_size_px (int): The square dimension of the generated image in pixels.
            res_max_images_to_generate (int): The maximum number of unique target images to generate.
                                Use -1 to generate all possible unique IDs for the given bit number.
            progress_callback (Optional): A callback function for reporting generation progress.

        Returns:
            int: The total number of unique target images successfully generated and saved.

        @Note: Currently only supports CCT targets with a 'black' center.
        """
        # Load settings
        CCTGenerator._load_settings()

        # Get metadata (bit number, center color) from the target type
        _target_type_name, code_bit_number, target_center_color = (
            TargetTypeUtils.get_metadata(target_type)
        )

        # Validate the color
        if target_center_color != "black":
            log.warning(
                f"CCT targets generation only supports 'black' target centers. "
                f"Ignoring request for '{target_center_color}'."
            )
            return 0

        # Validate the code_bit_number
        if code_bit_number < 4 or code_bit_number > 24:
            log.warning(
                f"Invalid code bit number '{code_bit_number}' for CCT generation. "
                f"This value must be within the range [4, 24]. Request ignored."
            )
            return 0

        # Get target_id list in bit format for generation
        target_id_bits_list = []
        max_int_value = 2**code_bit_number
        max_ids_number = res_max_images_to_generate
        if max_ids_number == -1:
            max_ids_number = max_int_value  # generate all possible unique target images

        for ii in range(0, max_int_value):
            if progress_callback:
                progress_callback.emit(100 * ii / max_int_value)

            target_id_bits = BitUtils.id_to_bits(
                ii, code_bit_number
            )  # e.g. [1, 0, 1, 0] for ii=5, N=4
            if target_id_bits not in target_id_bits_list:
                target_id_bits_list.append(target_id_bits)

                if len(target_id_bits_list) >= max_ids_number:
                    break

        # Determine the correct generate and save functions
        if res_image_type.lower() in ImageUtils.SUPPORT_RASTER_IMAGE_TYPES:
            generator_target_image__func__ = CCTGenerator._generate_target_image_black
            save_image__func__ = ImageUtils.save_image
        elif res_image_type.lower() == "svg":
            generator_target_image__func__ = (
                CCTGenerator._generate_target_image_black_svg
            )
            save_image__func__ = ImageUtils.save_svg
        elif res_image_type.lower() == "pdf":
            generator_target_image__func__ = (
                CCTGenerator._generate_target_image_black_svg
            )
            save_image__func__ = ImageUtils.save_pdf
        else:
            log.warning(f"Unsupported image type: {res_image_type}. Request ignored.")
            return 0

        # Generate targets images and save them
        for target_id_bits in target_id_bits_list:
            # Generate target (raster image or SVG string/image)
            target_image_content = generator_target_image__func__(
                target_id_bits, code_bit_number, res_image_size_px
            )

            # Save target content
            target_id = BitUtils.bits_to_id(target_id_bits, len(target_id_bits))
            target_image_path = os.path.join(
                res_folder_path, f"{target_id}.{res_image_type}"
            )

            # Save the generated image
            save_image__func__(target_image_path, target_image_content)

        # Return the number of the generated target images
        return len(target_id_bits_list)

    # --- Raster image generation (PNG, JPG, ...) ---
    @staticmethod
    def _generate_target_image_black(
        target_id_bits: str, code_bit_number: int, image_size: int
    ) -> Image:
        """
        Returns the generated CCT target as an raster image for the input target_id_bits.
        """
        # --- 1.1. Setup Variables ---

        # Colors
        color_black = ColorUtils.COLOR_BLACK
        color_white = ColorUtils.COLOR_WHITE

        # The input variable
        image_size_r = image_size / 2.0

        # Calculate the unit angle in degree (i.e. the angle of one bit)
        unit_angle = 360.0 / code_bit_number

        # Calculate center point
        center_x = center_y = int(image_size_r)
        center_pnt = (center_x, center_y)

        # Create square blank white image
        image = ImageUtils.create_blank_image(image_size, image_size, color=color_white)

        # --- 1.2. Radii Calculations ---

        # Outer radius of the coding region
        outer_ring_r_out = int(
            image_size_r
            * 2
            * CCTGeneratorSettings.TARGET_OUTER_RING_RADIUS_OUT
            / CCTGeneratorSettings.TARGET_SQUARE_SIZE
        )  # e.g. 0.75*size_r
        axes = (
            outer_ring_r_out,
            outer_ring_r_out,
        )  # Used for cv2.ellipse and drawing of coded sectors

        # Radius for the first white covering circle
        middle_ring_r_out = int(
            image_size_r
            * 2
            * CCTGeneratorSettings.TARGET_MIDDLE_RING_RADIUS_OUT
            / CCTGeneratorSettings.TARGET_SQUARE_SIZE
        )  # e.g. 0.5*size_r

        # Radius for the second black covering circle
        center_ring_r_out = int(
            image_size_r
            * 2
            * CCTGeneratorSettings.TARGET_CENTER_RING_RADIUS_OUT
            / CCTGeneratorSettings.TARGET_SQUARE_SIZE
        )  # e.g. 0.25*size_r

        # Radius for center small ring point (White)
        center_small_point_r_out = max(1, int(0.03 * image_size_r))

        # Radius for center small point (Black)
        center_small_point_r_in = max(1, int(0.01 * image_size_r))

        # --- 2. Draw the boundary square ---
        boundary_thickness = CCTGeneratorSettings.TARGET_BOUNDARY_THICKNESS
        if boundary_thickness > 0:
            boundary_thickness = int(
                np.round(
                    image_size
                    * (
                        CCTGeneratorSettings.TARGET_BOUNDARY_THICKNESS
                        / CCTGeneratorSettings.TARGET_SQUARE_SIZE
                    )
                )
            )
            CCTGenerator._draw_boundary_square_lines(
                image, color_black, thickness=boundary_thickness
            )

        # --- 3. Draw CCT target ---

        # 3.1. Draw coding sectors (Pieslices)
        for i in range(code_bit_number):
            # Only draw if the CCT bit is 1 (black)
            if target_id_bits[i] == 1:
                start_angle = i * unit_angle
                end_angle = (i + 1) * unit_angle

                # cv2.ellipse is used to draw a filled sector (thickness=-1)
                cv2.ellipse(
                    img=image,
                    center=center_pnt,
                    axes=axes,
                    angle=0,  # Rotation angle
                    startAngle=start_angle,
                    endAngle=end_angle,
                    color=color_black,
                    thickness=-1,  # Fill the sector
                )

        # 3.2. Draw white inner circle (Replaces first draw.ellipse)
        cv2.circle(image, center_pnt, middle_ring_r_out, color_white, -1)

        # 3.3. Draw black inner circle (Replaces second draw.ellipse)
        cv2.circle(image, center_pnt, center_ring_r_out, color_black, -1)

        if CCTGeneratorSettings.GENERATE_CENTRAL_POINT_ON:
            # 3.4. Draw center small ring point (White) (Replaces third draw.ellipse)
            cv2.circle(image, center_pnt, center_small_point_r_out, color_white, -1)

            # 3.5. Draw center small point (Black) (Replaces fourth draw.ellipse)
            cv2.circle(image, center_pnt, center_small_point_r_in, color_black, -1)

        # --- 4. Draw Target ID Text at the Bottom ---
        if CCTGeneratorSettings.DRAW_TARGET_ID_ON:
            CCTGenerator._draw_target_id_text(
                image, target_id_bits, code_bit_number, color_black
            )

        # --- 5. Return the image ---
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
    def _draw_target_id_text(
        image: Image,
        target_id_bits: str,
        code_bit_number: int,
        color_black=ColorUtils.COLOR_BLACK,
    ):
        # 1. Process input variable
        image_size = ImageUtils.get_square_image_size(image)

        target_id = BitUtils.bits_to_id(target_id_bits, len(target_id_bits))
        target_type = TargetTypeUtils.build_cct_target_type(code_bit_number, "black")
        target_id_text = f"{target_id} : {target_type}"

        text = target_id_text
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 2. Determine Font Scale based on image size to keep text proportional.
        # We aim for the text to take up about 90% of the image width.
        target_width_ratio = 0.5 * CCTGeneratorSettings.DRAW_TARGET_ID_FONT_SCALE

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
                CCTGeneratorSettings.DRAW_TARGET_ID_MARGIN_OUT
                / CCTGeneratorSettings.TARGET_SQUARE_SIZE
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
        target_id_bits: List[int],
        code_bit_number: int,
        res_image_size: int = None,  # Ignored for SVG
    ) -> str:
        """
        Returns the generated CCT target as an SVG string (vector format).
        """
        # --- 1.2. Setup Variables ---
        color_black = ColorUtils.COLOR_BLACK_HEX
        color_white = ColorUtils.COLOR_WHITE_HEX

        # Use the physical size from settings
        length_unit = CCTGeneratorSettings.LENGTH_UNIT  # e.g. cm
        image_size = CCTGeneratorSettings.TARGET_SQUARE_SIZE
        unit_angle = 360.0 / code_bit_number
        center_x = center_y = image_size / 2.0  # The half-size radius

        # Create SVG drawing object
        dwg = svgwrite.Drawing(
            size=(f"{image_size}{length_unit}", f"{image_size}{length_unit}"),
            viewBox=f"0 0 {image_size} {image_size}",
        )

        # Set background (whole image area) to white
        dwg.add(
            dwg.rect(insert=(0, 0), size=(image_size, image_size), fill=color_white)
        )

        # --- 1.2. Radii Calculations ---
        outer_ring_r_out = CCTGeneratorSettings.TARGET_OUTER_RING_RADIUS_OUT
        middle_ring_r_out = CCTGeneratorSettings.TARGET_MIDDLE_RING_RADIUS_OUT
        center_ring_r_out = CCTGeneratorSettings.TARGET_CENTER_RING_RADIUS_OUT

        # Use ratios relative to the full square size to calculate the small points
        center_small_point_r_out = 0.015 * image_size  #:SETTING:
        center_small_point_r_in = 0.005 * image_size

        # --- 2. Draw the boundary square ---
        boundary_thickness = CCTGeneratorSettings.TARGET_BOUNDARY_THICKNESS
        if boundary_thickness > 0:
            CCTGenerator._draw_boundary_square_lines_svg(
                dwg, color_black, thickness=boundary_thickness
            )

        # --- 3. Draw CCT target (Vector Shapes) ---

        # 3.1. Draw coding sectors (Black Arcs/Paths)
        for i in range(code_bit_number):
            if target_id_bits[i] == 1:
                # Calculate path for a pie slice (arc)
                # The coordinate system is now in mm
                start_angle = i * unit_angle
                end_angle = (i + 1) * unit_angle

                # SVG Path Drawing: M = Move to (center), L = Line to (outer edge start), A = Arc, Z = Close path
                path_d = CCTGenerator._create_arc_path_svg(
                    center_x, center_y, outer_ring_r_out, start_angle, end_angle
                )
                dwg.add(dwg.path(d=path_d, fill=color_black, stroke="none"))

        # 3.2. Draw white inner circle (Covers the inner part of the arcs)
        dwg.add(
            dwg.circle(
                center=(center_x, center_y),
                r=middle_ring_r_out,
                fill=color_white,
                stroke="none",
            )
        )

        # 3.3. Draw black inner circle (Central fiducial)
        dwg.add(
            dwg.circle(
                center=(center_x, center_y),
                r=center_ring_r_out,
                fill=color_black,
                stroke="none",
            )
        )

        if CCTGeneratorSettings.GENERATE_CENTRAL_POINT_ON:
            # 3.4. Draw center small ring point (White)
            dwg.add(
                dwg.circle(
                    center=(center_x, center_y),
                    r=center_small_point_r_out,
                    fill=color_white,
                    stroke="none",
                )
            )
            # 3.5. Draw center small point (Black)
            dwg.add(
                dwg.circle(
                    center=(center_x, center_y),
                    r=center_small_point_r_in,
                    fill=color_black,
                    stroke="none",
                )
            )

        # --- 4. Draw Target ID Text at the Bottom ---
        if CCTGeneratorSettings.DRAW_TARGET_ID_ON:
            target_id_color = color_black
            CCTGenerator._draw_target_id_text_svg(
                dwg, image_size, target_id_bits, code_bit_number, target_id_color
            )

        # --- 5. Return the SVG string ---
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
    def _create_arc_path_svg(cx, cy, r, start_angle_deg, end_angle_deg):
        """Helper to create an SVG path command string for a filled arc/sector."""

        # Convert to radians (numpy is required for this)
        start_angle_rad = np.deg2rad(
            start_angle_deg - 90
        )  # Adjust by -90 for SVG's coordinate system (0 deg is East, but should be North for radial)
        end_angle_rad = np.deg2rad(end_angle_deg - 90)

        # Calculate start and end points on the outer circle
        x_start = cx + r * np.cos(start_angle_rad)
        y_start = cy + r * np.sin(start_angle_rad)
        x_end = cx + r * np.cos(end_angle_rad)
        y_end = cy + r * np.sin(end_angle_rad)

        # Determine if the arc is large (more than 180 degrees)
        large_arc_flag = "1" if (end_angle_deg - start_angle_deg) > 180 else "0"

        # A = Arc command: (rx ry x-axis-rotation large-arc-flag sweep-flag x y)
        path = [
            f"M {cx},{cy}",  # 1. Move to Center
            f"L {x_start},{y_start}",  # 2. Line to start point
            f"A {r},{r} 0 {large_arc_flag} 1 {x_end},{y_end}",  # 3. Draw Arc (sweep-flag=1 for clockwise)
            f"Z",  # 4. Close path (Line back to center)
        ]
        return " ".join(path)

    @staticmethod
    def _draw_target_id_text_svg(
        dwg: svgwrite.Drawing,
        image_size: float,
        target_id_bits: List[int],
        code_bit_number: int,
        color_black: str,
    ):
        """Renders the target ID text using SVG text elements."""
        # 0. Get target id text
        target_id = BitUtils.bits_to_id(target_id_bits, len(target_id_bits))
        target_type = TargetTypeUtils.build_cct_target_type(code_bit_number, "black")
        target_id_text = f"{target_id} : {target_type}"

        # Calculate text position (bottom-center)
        center_x = image_size / 2.0
        target_text_height = (
            image_size * 0.015
        ) * CCTGeneratorSettings.DRAW_TARGET_ID_FONT_SCALE

        # T Place text near the bottom
        baseline_y = image_size - CCTGeneratorSettings.DRAW_TARGET_ID_MARGIN_OUT
        dwg.add(
            dwg.text(
                text=target_id_text,
                insert=(center_x, baseline_y),
                fill=color_black,
                # Set font size relative to the target size in mm
                font_size=f"{target_text_height}mm",
                # Center the text horizontally
                text_anchor="middle",
                # Using a common sans-serif font
                font_family="Arial, sans-serif",
            )
        )


# --------------------------------------------
# Main execution block, if this script is executed directly
# Note. Use only for for development-time testing
# --------------------------------------------
if __name__ == "__main__":
    # Import tests.dev module
    from tests.dev import (
        run_dev_test_find_cct_targets,
        run_dev_test_generate_cct_targets,
    )

    # Run dev test
    run_dev_test_find_cct_targets()
    run_dev_test_generate_cct_targets()
