# Import external modules
import cv2
import math
import cairosvg
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, Tuple, List

# Import internal modules
from utils.color import ColorUtils
from utils.math import MathUtils
from utils.line import LineUtils, LineSegmentArray

# Predefine types
ColorInput = Union[str, Tuple[int, int, int]]

Image = np.ndarray
ImageBinary = np.ndarray




# --------------------------------------------
# Image Utilities Functions
# --------------------------------------------
class ImageUtils:
    """A static class for managing helpful image operations and functions."""

    # --- Support image types ---
    SUPPORT_RASTER_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    SUPPORT_RASTER_IMAGE_TYPES = ("jpg", "jpeg", "png", "bmp", "tiff", "tif")

    SUPPORT_VECTOR_IMAGE_TYPES = ("svg", "pdf")
    SUPPORT_GENERATED_IMAGE_TYPES = (
        SUPPORT_RASTER_IMAGE_TYPES + SUPPORT_VECTOR_IMAGE_TYPES
    )

    # --- Image file functions ---
    @staticmethod
    def file_is_image(file_path: Path) -> bool:
        """
        Checks if the given Path object points to an existing file with an image extension.

        Args:
            file_path: The pathlib.Path object representing the file.

        Returns:
            True if the path exists, is a file, and has a supported image extension (case-insensitive).
        """
        # Ensure the path exists and is a file
        is_file_and_exists = file_path.is_file()

        # Check if the file's suffix (extension) is in our supported set
        has_supported_extension = (
            file_path.suffix.lower() in ImageUtils.SUPPORT_RASTER_IMAGE_EXTENSIONS
        )

        return is_file_and_exists and has_supported_extension

    @staticmethod
    def find_all_images_path_in_folder(input_folder_path: Path) -> list[Path]:
        """
        Recursively searches the given folder path for all files that are valid images.

        Args:
            input_folder_path: The root directory to start searching from.

        Returns:
            A list of pathlib.Path objects for all valid images found.
        """
        # Use a generator expression for memory efficiency and then convert to a list
        # The rglob('*') performs the recursive search.
        image_paths_to_process = [
            file_path
            for file_path in input_folder_path.rglob("*")
            if ImageUtils.file_is_image(file_path)
        ]

        image_paths_to_process.sort()
        return image_paths_to_process

    @staticmethod
    def load_image(image_path: str) -> Image:
        """
        Loads an image from the specified path using OpenCV (cv2).

        This function reads the image file and checks if the loading was successful.
        If the image cannot be found or loaded (e.g., due to a bad path or corrupted
        file), it raises a FileNotFoundError.

        Args:
            image_path (str): The full path to the image file.

        Returns:
            np.ndarray: The image data as a NumPy array (the standard OpenCV format).

        Raises:
            FileNotFoundError: If cv2.imread is unable to load the image.
        """
        # cv2.imread returns None if the image cannot be loaded.
        image: Optional[np.ndarray] = cv2.imread(image_path)

        if image is None:
            # Raise an informative error if the image failed to load
            raise FileNotFoundError(
                f"Unable to load image at {image_path}. Check if the file exists and is a valid image format."
            )

        return image

    @staticmethod
    def save_image(image_path: str, image: Image):
        """
        Saves the image (NumPy array) to the specified file path using OpenCV (cv2).

        Args:
            image_path (str): The full path where the image should be saved.
            image (np.ndarray): The image data as a NumPy array (OpenCV format).

        Raises:
            IOError: If cv2.imwrite fails to save the image to the specified path.
        """
        # cv2.imwrite returns True on success, False on failure.
        success: bool = cv2.imwrite(image_path, image)
        if not success:
            raise IOError(
                f"Unable to save image to {image_path}. Check if the directory exists and permissions are correct."
            )

    # --- Get image properties functions ---
    @staticmethod
    def get_image_dimensions(image: Image):
        """Returns the dimensions of the image as (height, width) in pixels."""
        h = image.shape[0]  # Number of rows
        w = image.shape[1]  # Number of columns
        return (h, w)

    @staticmethod
    def get_square_image_size(image: Image) -> int:
        """
        Returns the side length (size) in pixels of a square input image.

        This function is useful for quickly retrieving the N value from an N x N image.

        Args:
            image: The square input image (NumPy array).

        Returns:
            int: The side length of the square image in pixels.

        Raises:
            ValueError: If the input array does not represent a square image.
        """
        # Check that the array has at least 2 dimensions (Height and Width)
        if image.ndim < 2:
            raise ValueError(
                "Input array must have at least 2 dimensions (height and width)."
            )

        # Extract height and width from the shape.
        # image.shape[:2] safely handles grayscale (2D) and color (3D/4D) images.
        height, width = image.shape[:2]

        if height != width:
            raise ValueError(
                f"Input image must be square (height must equal width). "
                f"Received dimensions: {height}x{width}."
            )

        return height  # Since height == width, this is the side length

    # --- Image operations  ---
    @staticmethod
    def convert_image_to_gray(image: Image):
        """
        Converts a color image (BGR) to grayscale.

        If the input image is already a 2-dimensional grayscale array, it is returned unchanged.
        If it's a 3-dimensional array, it performs the color space conversion.

        Args:
            image (Image): The input image data as a NumPy array. Can be 2D (grayscale) or 3D (color).

        Returns:
            Image: The resulting grayscale image (2-dimensional NumPy array).

        Raises:
            ValueError: If the input array has an unsupported number of dimensions.
        """
        if len(image.shape) == 3:
            # Convert from BGR (OpenCV default) to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return gray_image
        elif len(image.shape) == 2:
            # Image is already grayscale (2D array)
            return image
        else:
            # Handle cases where shape is neither 2D nor 3D (e.g., 4D array)
            raise ValueError(
                f"Unsupported image format with {len(image.shape)} dimensions for grayscale conversion."
            )

    @staticmethod
    def convert_image_to_color(image: Image) -> Optional[Image]:
        """
        Converts any valid single-channel (grayscale/binary) image into a
        three-channel BGR color image (0/255).
        If the image is already color (3 channels), it returns the image directly.

        Handles:
        1. Single-channel (2D) images with 0/1 pixel values (rescales and converts).
        2. Single-channel (2D) images with 0/255 pixel values (converts).
        3. Three-channel (3D) images (returns as is).

        Args:
            image: A NumPy array representing an image.

        Returns:
            Optional[Image]: The 3-channel BGR image, or None if the
                                  input is not a valid 2D or 3D image array.
        """

        # 1. Input Validation and Channel Check
        if image.ndim not in [2, 3]:
            raise ValueError(
                "Error: Input array must be a 2D (grayscale) or 3D (color) image."
            )

        # If already a 3-channel color image, return it immediately.
        if image.ndim == 3:
            # Check for standard BGR/RGB channels (H, W, 3)
            if image.shape[2] == 3:
                return image.copy()
            # Handle non-standard 4-channel (e.g., RGBA) by converting
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                raise ValueError(
                    f"Error: 3D image has {image.shape[2]} channels, which is not supported."
                )

        # --- Processing Single-Channel (2D) Image ---

        # Get a copy to avoid modifying the original array
        processed_image = image.copy()

        # Determine the maximum value in the image to check the range
        max_val = np.max(processed_image)

        # 2. Rescale and Type Conversion
        if max_val <= 1.0:
            # Handles 0/1 binary images, float grayscale images, etc. Rescale to 0/255.
            processed_image = (processed_image * 255).astype(np.uint8)

        elif processed_image.dtype != np.uint8:
            # Handles 0/255 images that might be int16 or float64. Convert to uint8.
            processed_image = processed_image.astype(np.uint8)

        # 3. Convert Single Channel (0/255) to BGR
        color_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

        return color_image

    @staticmethod
    def preprocess_to_binary_image(
        image: Image,
        remove_noise: bool = False,
        noise_kernel_size: int = 3,
        threshold_mode: Optional[str] = None,
        threshold_value: Optional[float] = None,
    ) -> Tuple[ImageBinary, float]:
        """
        Converts an input image (color or grayscale) into a clean, binary image (0 or 1).

        The process includes grayscale conversion, thresholding, and optional morphological filtering.

        Args:
            image: The source image as a NumPy array (e.g., color or grayscale).
            remove_noise: If True, applies morphological operations for noise removal.
            noise_kernel_size: The side length of the kernel for morphological operations (e.g., 3 for a 3x3 kernel).
            threshold_mode: Defines the threshold calculation method (ignored if threshold_value is set).
                            Options are: 'otsu' (default), 'triangle', 'max' (histogram mode), 'mean' (pixel mean).
            threshold_value: Optional fixed threshold value (0-255). If provided, this value
                             is used directly, overriding threshold_mode.

        Returns:
            Tuple[ImageBinary, float]: A tuple containing:
                                      - ImageBinary: The binary image (0 for black/background, 1 for white/foreground).
                                      - float: The actual threshold value used (0-255).

        Raises:
            ValueError: If an invalid threshold mode is provided.
        """
        if threshold_mode is None and threshold_value is None:
            raise ValueError(
                "Both threshold mode and threshold value are None. Expected one of them to be not None."
            )

        # Create a copy of the input image and process it
        processed_image = image.copy()

        # 1. Convert the image to grayscale (assumes helper exists)
        processed_image = ImageUtils.convert_image_to_gray(processed_image)

        # --- 2. Thresholding Logic ---
        calculated_threshold: float = 0.0

        if threshold_value is not None:
            # Case A: Use fixed threshold value (overrides threshold_mode)
            calculated_threshold = threshold_value
            _, processed_image = cv2.threshold(
                processed_image,
                thresh=int(calculated_threshold),
                maxval=1,
                type=cv2.THRESH_BINARY,
            )
        elif (
            threshold_mode.lower() == "adaptive"
        ):  # Note. threshold_mode is not None & threshold_value is None
            # Case B.1: Use adaptive method
            calculated_threshold = None
            processed_image = cv2.adaptiveThreshold(
                processed_image,
                1,  # Max Value
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  #:SETTING: ADAPTIVE_THRESH_GAUSSIAN_C or ADAPTIVE_THRESH_MEAN_C
                cv2.THRESH_BINARY,
                51,  # blockSize: neighborhood (51x51) or (101x101) #:SETTING:
                5,  # C: Subtract 5 from the mean #:SETTING:
            )
        elif (
            threshold_mode.lower() == "otsu"
        ):  # Note. threshold_mode is not None & threshold_value is None
            # Case B.2: Use Otsu's automatic method
            calculated_threshold, processed_image = cv2.threshold(
                processed_image,
                thresh=0,
                maxval=1,
                type=cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                # type=cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
            )
        elif (
            threshold_mode.lower() == "triangle"
        ):  # Note. threshold_mode is not None & threshold_value is None
            # Case B.3: Use Triangle automatic method
            calculated_threshold, processed_image = cv2.threshold(
                processed_image,
                thresh=0,
                maxval=1,
                type=cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
            )

        else:  # Note. threshold_mode is not None & threshold_value is None
            # Case B.3: Use calculated threshold ('max' or 'mean')
            threshold_mode = threshold_mode.lower()

            if threshold_mode == "max":
                # Calculate histogram
                hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])

                # Apply smoothing to ensure the peak is robust (not a single spike)
                hist_smoothing_kernel_size = 5  #:SETTING:
                if hist_smoothing_kernel_size > 1:
                    kernel = (
                        np.ones(hist_smoothing_kernel_size) / hist_smoothing_kernel_size
                    )
                    smoothed_hist = np.convolve(hist.flatten(), kernel, mode="same")
                    calculated_threshold = float(np.argmax(smoothed_hist))
                else:
                    calculated_threshold = float(np.argmax(hist))

            elif threshold_mode == "mean":
                # calculated_threshold is the arithmetic mean of all pixel INTENSITIES (0-255 value)
                calculated_threshold = float(np.mean(processed_image))
            else:
                raise ValueError(
                    f"Invalid threshold mode: {threshold_mode}. Must be 'otsu', 'triangle', 'max', or 'mean'."
                )

            # Apply manual thresholding using the calculated_threshold
            calculated_threshold = round(calculated_threshold)
            _, processed_image = cv2.threshold(
                processed_image,
                thresh=int(calculated_threshold),
                maxval=1,
                type=cv2.THRESH_BINARY,
            )

        # 3. Apply opening and closing to remove isolated noise pixels
        if remove_noise:
            processed_image = ImageUtils.remove_binary_islands(
                processed_image, noise_kernel_size
            )

        return processed_image, calculated_threshold

    @staticmethod
    def remove_binary_islands(
        image_bin: ImageBinary, kernel_size: int = 3
    ) -> ImageBinary:
        """
        Removes small isolated islands of both white (salt) and black (pepper)
        noise from a binary image using morphological opening and closing.

        Args:
            image_bin: The binary input image (0 or 255 values).
            kernel_size: The size of the square kernel used for morphological operations.
                         A larger kernel removes larger noise artifacts. Default is 3 (3x3).

        Returns:
            ImageBinary: The cleaned binary image.
        """
        if image_bin.ndim > 2 and image_bin.shape[2] > 1:
            raise ValueError("Input image must be grayscale or binary (2D array).")

        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd integer.")

        # Ensure the image is 8-bit single channel (typical for cv2 ops)
        if image_bin.dtype != np.uint8:
            image_bin = image_bin.astype(np.uint8)

        # 1. Define the kernel (structuring element)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 2. Remove White Islands (Salt Noise) using OPENING
        # Opening = Erosion (removes small white spots) followed by Dilation (restores main objects)
        cleaned_image = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel)

        # 3. Remove Black Islands/Holes (Pepper Noise) using CLOSING
        # Closing = Dilation (fills small black holes) followed by Erosion (restores main objects)
        cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, kernel)

        return cleaned_image

    @staticmethod
    def rotate_image_to_angle(image: Image, angle: float) -> Image:
        """
        Rotates an image by a specified angle around its center.

        This function is typically used to normalize an object's orientation by
        rotating by the negative of its calculated angle.

        Args:
            image: The input image (NumPy array).
            ion_angle: The angle in degrees to rotate the image. To make the
                       object horizontal, pass the NEGATIVE of the detected angle.

        Returns:
            Image: The rotated image, maintaining the original image dimensions.
        """
        # Get image dimensions (height, width)
        (h, w) = image.shape[:2]

        # 1. Define the center of rotation
        center = (w // 2, h // 2)

        # 2. Get the rotation matrix (scale factor 1.0)
        # NOTE: cv2.getRotationMatrix2D expects the angle in degrees.
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 3. Apply the affine transformation
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

        return rotated_image

    @staticmethod
    def invert_binary_image(image_bin: ImageBinary) -> ImageBinary:
        """
        Inverts the colors of a binary image based on its content (0/1 or 0/255).

        - If the image contains values close to 255 (typical of OpenCV 8-bit output), it uses cv2.bitwise_not (0 <-> 255).
        - If the image contains only 0s and 1s, it uses simple subtraction (0 <-> 1).

        Args:
            image_bin: The input binary image.

        Returns:
            ImageBinary: The inverted binary image.

        Raises:
            ValueError: If the input array is empty.
        """
        if image_bin.size == 0:
            raise ValueError("Input image array is empty.")

        # Check the maximum value in the image (or a sample of it)
        max_val = np.max(image_bin)

        if max_val > 1:
            # Assuming 0/255 binary format (OpenCV default for uint8)
            if image_bin.dtype == np.uint8:
                # Use cv2.bitwise_not for 0 <-> 255 flip (most efficient for uint8)
                return cv2.bitwise_not(image_bin)
            else:
                # Simple subtraction for other high-value types if cv2 isn't suitable
                return 255 - image_bin.astype(
                    np.float32
                )  # Convert to float temporarily
        else:
            # Assuming 0/1 binary format
            # Use simple subtraction: 1 - 0 = 1, 1 - 1 = 0
            return 1 - image_bin

    @staticmethod
    def remove_long_black_lines(
        image_bin: ImageBinary,
        ratio_threshold: float = 0.8,
        boundary_ratio: float = 0.1,
    ) -> ImageBinary:
        """
        Removes long, straight black lines (0s) by changing them to white (1s).

        A line is removed if its length exceeds the specified ratio of the image's
        side dimension. This change is ONLY applied to pixels in the border region
        defined by the boundary_ratio.

        Note: The input image MUST be square (H == W) and use 0/1 for black/white.

        Args:
            image_bin: The input binary image (must contain only 0s and 1s).
            ratio_threshold: The minimum length ratio (e.g., 0.8 means 80% of the dimension).
            boundary_ratio: The fraction (0.0 to 0.5) defining the width of the
                            border area where lines can be removed. Default is 0.2 (1/5).

        Returns:
            ImageBinary: The modified binary image with long black lines removed (set to 1)
                        only in the border area.

        Raises:
            ValueError: If the input image is not 2D, not square, or the ratios are invalid.
        """
        if image_bin.ndim != 2:
            raise ValueError("Input image must be a 2D grayscale array.")
        if not 0.0 < ratio_threshold < 1.0:
            raise ValueError("Ratio threshold must be strictly between 0 and 1.")

        # --- VALIDATION for boundary_ratio ---
        if not 0.0 < boundary_ratio <= 0.5:
            raise ValueError(
                "Boundary ratio must be strictly between 0 and 0.5 (to allow a center area)."
            )

        # Ensure the image is mutable and handle the 0/1 data type
        modified_image = image_bin.copy()

        H, W = modified_image.shape

        # --- CHECK: Ensure the image is square ---
        if H != W:
            raise ValueError(
                f"Input image must be square (Height {H} must equal Width {W})."
            )

        # --- Define Center/Border Boundaries using the argument ---
        border_start = int(H * boundary_ratio)
        border_end = H - border_start  # The (1 - boundary_ratio) mark

        # --- 1. Identify and Remove Long HORIZONTAL Lines ---

        min_horizontal_len = int(W * ratio_threshold)

        # Iterate over each row
        for r in range(H):
            row = modified_image[r, :]
            current_len = 0
            start_c = -1

            for c in range(W):
                is_end_of_row = c == W - 1

                if row[c] == 0:
                    if current_len == 0:
                        start_c = c
                    current_len += 1

                # Check for run termination
                if row[c] == 1 or is_end_of_row:

                    if current_len >= min_horizontal_len and start_c != -1:
                        # Found a long run. Now, apply the change only in the border area.
                        for c_idx in range(start_c, start_c + current_len):

                            # A pixel is in the center if:
                            # (r is in [border_start, border_end)) AND (c_idx is in [border_start, border_end))
                            in_center_row = border_start <= r < border_end
                            in_center_col = border_start <= c_idx < border_end

                            # Only set to white (1) if the pixel is NOT in the center square
                            if not (in_center_row and in_center_col):
                                modified_image[r, c_idx] = 1

                    # Reset run tracking
                    current_len = 0
                    start_c = -1

        # --- 2. Identify and Remove Long VERTICAL Lines ---

        min_vertical_len = int(H * ratio_threshold)

        # Iterate over each column
        for c in range(W):
            col = modified_image[:, c]
            current_len = 0
            start_r = -1

            for r in range(H):
                is_end_of_col = r == H - 1

                if col[r] == 0:
                    if current_len == 0:
                        start_r = r
                    current_len += 1

                # Check for run termination
                if col[r] == 1 or is_end_of_col:

                    if current_len >= min_vertical_len and start_r != -1:
                        # Found a long run. Now, apply the change only in the border area.
                        for r_idx in range(start_r, start_r + current_len):

                            # The current pixel is (r_idx, c)
                            # A pixel is in the center if:
                            # (r_idx is in [border_start, border_end)) AND (c is in [border_start, border_end))
                            in_center_row = border_start <= r_idx < border_end
                            in_center_col = border_start <= c < border_end

                            # Only set to white (1) if the pixel is NOT in the center square
                            if not (in_center_row and in_center_col):
                                modified_image[r_idx, c] = 1

                    # Reset run tracking
                    current_len = 0
                    start_r = -1

        return modified_image

    @staticmethod
    def clean_big_dark_areas_on_boundaries(
        image_bin: ImageBinary, area_threshold_ratio: float = 0.01
    ) -> ImageBinary:
        """
        Removes large dark regions (black areas) that touch the image boundary.

        This is useful for removing border artifacts or shadows that span the
        image edge and could interfere with target decoding. The function
        treats "dark" as pixels with value 0 and works with images in either
        0/1 or 0/255 binary formats. The output preserves the input scale
        (0/1 or 0/255) and dtype is uint8.

        Args:
            image_bin: Input binary image (2D) with dark regions as 0.
            area_threshold_ratio: If a dark connected component that touches the
                                  image boundary has area >= (area_threshold_ratio * image_area)
                                  it will be removed (set to background/white).

        Returns:
            ImageBinary: Cleaned binary image with large boundary-touching dark
                         regions removed.
        """
        if image_bin.ndim != 2:
            raise ValueError("Input image must be a 2D array.")

        H, W = image_bin.shape
        if H == 0 or W == 0:
            return image_bin

        # Preserve input scaling: 0/1 or 0/255
        max_val = np.max(image_bin)

        # Work in uint8 0/255 for connected components
        if max_val <= 1:
            img_uint8 = (image_bin.astype(np.uint8) * 255)
            # img_uint8 uses 0/255 scale; background (white) must be 255 here
            background_value = 255
            out_scale = "01"
        else:
            img_uint8 = image_bin.astype(np.uint8)
            background_value = 255
            out_scale = "0255"

        # Create mask of dark pixels (value == 0) as foreground for CC analysis
        dark_mask = (img_uint8 == 0).astype(np.uint8) * 255

        # Connected components on dark regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark_mask, connectivity=8
        )

        image_area = float(H * W)

        # Iterate over each component (skip label 0 which is background)
        for lab in range(1, num_labels):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            left = int(stats[lab, cv2.CC_STAT_LEFT])
            top = int(stats[lab, cv2.CC_STAT_TOP])
            width = int(stats[lab, cv2.CC_STAT_WIDTH])
            height = int(stats[lab, cv2.CC_STAT_HEIGHT])

            # Check if this component touches any image boundary
            touches_boundary = (
                left == 0 or top == 0 or (left + width) >= W or (top + height) >= H
            )

            if not touches_boundary:
                continue
            
            area_ratio = area / image_area
            if area_ratio >= float(area_threshold_ratio):
                # Remove this dark component by setting its pixels to background (white)
                img_uint8[labels == lab] = background_value

        # Convert back to original scale (0/1 or 0/255)
        if out_scale == "01":
            return (img_uint8 // 255).astype(np.uint8)
        else:
            return img_uint8

    # --- Image analyzing functions ---
    @staticmethod
    def analyze_inner_circle(
        image_bin: ImageBinary, radius: int
    ) -> Tuple[int, int, int]:
        """
        Analyzes the pixel values (0s and 1s) within a central circle
        of a binary image.

        Args:
            image_bin: The binary input image (NumPy array, typically dtype=uint8
                    with values 0 or 1).
            radius: The radius of the inner circle in pixels.

        Returns:
            A tuple containing: (count_of_all_pixels, count_of_ones_pixels, count_of_zeros_pixels).
            Returns (0, 0, 0) if the radius is 0 or less.
        """

        if radius <= 0:
            return 0, 0, 0

        # Ensure the input is treated as 8-bit unsigned integer (standard for binary images)
        if image_bin.dtype != np.uint8:
            # Convert to uint8 and ensure values are strictly 0 or 1
            image_bin = (image_bin > 0).astype(np.uint8)

        H, W = image_bin.shape

        # 1. Determine the center of the image
        center_y, center_x = H // 2, W // 2

        # 2. Create a coordinate grid
        Y, X = np.ogrid[:H, :W]

        # 3. Calculate the distance of every pixel from the center
        # Distance = sqrt((X - center_x)^2 + (Y - center_y)^2)
        distance_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        # 4. Create the circular mask
        # This mask is a boolean array (True for pixels inside the circle)
        circular_mask = distance_from_center <= radius

        # --- Calculations ---

        # Total pixels in the circle: Sum of the 'True' values in the mask
        count_of_all_pixels = np.sum(circular_mask)

        # Pixels with value 1: Apply the mask to the image and sum the result.
        # Since the image is binary (0 or 1), summing gives the count of 1s.
        count_of_ones_pixels = np.sum(image_bin[circular_mask])

        # Pixels with value 0: Total pixels minus count of 1s
        count_of_zeros_pixels = count_of_all_pixels - count_of_ones_pixels

        return count_of_all_pixels, count_of_ones_pixels, count_of_zeros_pixels

    @staticmethod
    def count_circular_contours(
        image_bin: np.ndarray,
        min_circularity_threshold: float = 0.8,
        min_area_threshold: int = 10,
    ) -> Tuple[int, int, List[Tuple[float, float]]]:
        """
        Counts circular contours in a binary image and returns the centers of the
        circular ones.

        The image is assumed to be almost white (1s) with black dots (0s) being the
        objects of interest.

        Args:
            image_bin: The binary input image (NumPy array, expected 0s and 1s).
            min_circularity_threshold: The minimum acceptable circularity score (0.0 to 1.0)
                                    for a contour to be classified as circular.
                                    (e.g., 0.8 means only shapes with a score of 0.8 or higher are accepted).
            min_area_threshold: Minimum pixel area a contour must have to be considered.
                                Used to filter out single-pixel noise.

        Returns:
            Tuple: (total_contours, circular_contours_count, circular_centers)
        """
        # --- 1. Prepare Image for OpenCV ---
        # OpenCV's contour finding expects 0/255. Convert 0/1 input to 0/255.
        image_255 = (image_bin * 255).astype(np.uint8)

        # --- 2. Find Contours ---
        # cv2.RETR_LIST retrieves all contours without establishing hierarchy.
        # cv2.CHAIN_APPROX_SIMPLE saves memory by only storing boundary points.
        contours, _ = cv2.findContours(
            image_255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLEcount_circular_contours
        )
        ImageUtils.plot_contours(ImageUtils.convert_image_to_color(image_bin), contours)

        total_contours = len(contours)
        circular_contours_count = 0
        circular_centers: List[Tuple[float, float]] = []

        # --- 3. Iterate and Analyze ---
        for contour in contours:
            # Calculate geometric properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, closed=True)

            # Filter out very small artifacts/noise
            if area < min_area_threshold or perimeter == 0:
                total_contours -= (
                    1  # Optionally decrease total count if noise is ignored
                )
                continue

            # Calculate circularity using the provided formula
            circularity_score = MathUtils.calc_contour_circularity(area, perimeter)

            # --- NEW FILTERING LOGIC ---
            # Check if the score meets the minimum required threshold
            if circularity_score >= min_circularity_threshold:
                circular_contours_count += 1

                # Calculate Centroid (Center of Mass) using Moments
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    # Centroid X and Y coordinates (kept as floats)
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                    circular_centers.append((cX, cY))
                # If M["m00"] is zero, it's a degenerate contour, so we skip center calculation.

        return total_contours, circular_contours_count, circular_centers

    # --- Image cleaning functions ---
    @staticmethod
    def clean_contours_to_grid__centroid_based(
        image_bin: np.ndarray,
        num_cells: int,
        min_area_threshold: int = 10,
        min_circularity_threshold: float = 0.8,
    ) -> np.ndarray:
        """
        Filters white objects (contours) in a binary image by size and circularity.
        Then, it assigns each remaining circular object to a grid cell based on its
        center of mass, and zeroes out any part of the object that falls outside
        its assigned cell.

        Args:
            image_bin: The square binary input image (NumPy array, 0=black, 1=white).
            num_cells: The total number of cells to divide the image into (must be a perfect square, e.g., 4, 9, 16).
            min_area_threshold: Minimum pixel area a contour must have to be considered.
            min_circularity_threshold: The minimum circularity score required to keep a contour.

        Returns:
            np.ndarray: The new binary image (0s and 1s) with non-circular/small
                        objects removed and circular objects cropped to their respective cells.
        """
        h, w = image_bin.shape

        # 1. Validation and Grid Setup
        if h != w:
            raise ValueError("Error: The input binary image is not square.")

        n = int(math.sqrt(num_cells))
        if n * n != num_cells or n == 0:
            raise ValueError(
                f"The input argument 'num_cells' must be a positive perfect square (e.g., 4, 9, 16), but got {num_cells}."
            )

        cell_size = w // n

        # 2. Preparation for Contours and Output
        # The output image will start as all black (0)
        centroid_output_image = np.zeros_like(image_bin, dtype=np.uint8)

        # Convert 0/1 input image to 0/255 for OpenCV contour finding (required)
        image_255 = (image_bin * 255).astype(np.uint8)

        # Find Contours
        contours, _ = cv2.findContours(
            image_255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # 3. Iterate, Filter, and Crop
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, closed=True)

            # Filter 1: Area and Perimeter check (noise removal)
            if area < min_area_threshold or perimeter == 0:
                continue

            circularity_score = MathUtils.calc_contour_circularity(area, perimeter)

            # Filter 2: Circularity check (shape removal)
            if circularity_score < min_circularity_threshold:
                continue

            # At this point, the contour is considered a valid, circular object.

            # 4. Find Mass Center (Centroid)
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue  # Skip degenerate contours

            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]

            # 5. Assign to Grid Cell
            col_idx = int(cX // cell_size)
            row_idx = int(cY // cell_size)

            # Ensure indices are within bounds (important for images not perfectly divisible)
            col_idx = max(0, min(n - 1, col_idx))
            row_idx = max(0, min(n - 1, row_idx))

            # Calculate pixel boundaries of the assigned cell
            x_start, y_start = col_idx * cell_size, row_idx * cell_size
            x_end, y_end = (col_idx + 1) * cell_size, (row_idx + 1) * cell_size

            # 6. Crop and Draw (Pixel Cleaning)
            # Create a temporary mask for the current contour
            contour_mask = np.zeros_like(image_bin, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], 0, 1, -1)  # Draw the contour (1s)

            # Create a mask for the assigned cell boundary (1s only within the cell)
            cell_mask = np.zeros_like(image_bin, dtype=np.uint8)
            cell_mask[y_start:y_end, x_start:x_end] = 1

            # The pixels to KEEP are those that belong to BOTH the contour AND the cell
            pixels_to_keep = np.logical_and(contour_mask, cell_mask)

            # Copy the kept pixels into the final output image
            centroid_output_image[pixels_to_keep] = 1

        return centroid_output_image

    @staticmethod
    def clean_contours_to_grid__proportion_based(
        image_bin: np.ndarray,
        num_cells: int,
        min_area_threshold: int = 10,
        min_circularity_threshold: float = 0.8,
    ) -> np.ndarray:
        """
        Filters white objects (contours) in a binary image by size and circularity, 
        then reconstructs the objects by only drawing the portions that lie within 
        "dominant" grid cells.
        
        A grid cell is considered "dominant" for a contour if the modified proportion 
        value for that cell exceeds 0.75 (3/4).
        
        The modified proportion is calculated as: 
        (Area of Contour in Cell / Total Contour Area) * (Number of Occupied Cells)

        Args:
            image_bin: The square binary input image (NumPy array, 0=black, 1=white).
            num_cells: The total number of cells to divide the image into (must be a 
                    positive perfect square, e.g., 4, 9, 16).
            min_area_threshold: Minimum pixel area a contour must have to be considered 
                                for grid assignment.
            min_circularity_threshold: The minimum circularity score required to keep a 
                                    contour (circularity = 1.0 for a perfect circle).

        Returns:
            np.ndarray: A new binary image (0s and 1s) where small/non-circular objects 
                        are removed, and the remaining circular objects are drawn only 
                        in the grid cells determined to be "dominant" by the 0.75 threshold.
        """
        h, w = image_bin.shape

        # 1. Validation and Grid Setup
        if h != w:
            raise ValueError("Error: The input binary image is not square.")
        n = int(math.sqrt(num_cells))
        if n * n != num_cells or n == 0:
            raise ValueError(f"The input argument 'num_cells' must be a positive perfect square.")
        cell_size = w // n

        # 2. Preparation for Contours and Output
        # The output image will start as all black (0)
        proportion_output_image = np.zeros_like(image_bin, dtype=np.uint8)

        # Convert 0/1 input image to 0/255 for OpenCV contour finding (required)
        image_255 = (image_bin * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            image_255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # 3. Iterate, Filter, and Process Output
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, closed=True)

            # Filter 1: Area and Perimeter check (noise removal)
            if area < min_area_threshold or perimeter == 0:
                continue

            circularity_score = MathUtils.calc_contour_circularity(area, perimeter)
            # print(f"=== TEST: circularity_score = {circularity_score}")

            # Filter 2: Circularity check (shape removal)
            if circularity_score < min_circularity_threshold:
                continue
                
            # --- Data Setup ---
            
            # Create a temporary mask for the current contour
            contour_mask = np.zeros_like(image_bin, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], 0, 1, -1)  # Draw the contour (1s)

            # Calculate the Proportion Matrix
            proportion_matrix_raw = ImageUtils._get_single_contour_proportion_matrix(
                image_shape=image_bin.shape,
                contour=contour,
                num_cells=num_cells
            )
            num_occupied_cells = np.sum(proportion_matrix_raw != 0)
            
            # Apply the required modification: proportion_matrix *= num_occupied_cells
            proportion_matrix = proportion_matrix_raw * num_occupied_cells 
            # print(f"=== TEST: proportion_matrix = {proportion_matrix}")
            
            # -------------------------------------------------------------
            # Proportion Assignment Logic
            # -------------------------------------------------------------
            
            # Iterate over all cells in the n x n grid
            threshold_value = 3/4.  #:SETTING:
            if circularity_score > 0.8: # The contour is very circular and therefore it is only for one dot
                threshold_value = 1 # means account only the cell with the highest proportion of the contour

            for r in range(n):
                for c in range(n):
                    # Check the threshold condition
                    if proportion_matrix[r, c] >= threshold_value:
                        
                        # This cell (r, c) is chosen. Calculate its boundaries.
                        px_start, py_start = c * cell_size, r * cell_size
                        px_end, py_end = (c + 1) * cell_size, (r + 1) * cell_size
                        
                        # Create a mask for this cell
                        prop_cell_mask = np.zeros_like(image_bin, dtype=np.uint8)
                        prop_cell_mask[py_start:py_end, px_start:px_end] = 1
                        
                        # Keep pixels belonging to BOTH the contour AND this high-proportion cell
                        pixels_to_keep_prop = np.logical_and(contour_mask, prop_cell_mask)
                        proportion_output_image[pixels_to_keep_prop] = 1

        # Return the single proportion-based result
        return proportion_output_image
    
    @staticmethod
    def _get_single_contour_proportion_matrix(
        image_shape: tuple,
        contour: np.ndarray, # Takes a single contour (a NumPy array)
        num_cells: int,
    ) -> np.ndarray: # Returns a single n x n matrix
        """
        Generates an n x n matrix for a single contour where each element represents 
        the proportion of the contour's total pixel area that falls within the 
        respective grid cell.
        """
        h, w = image_shape
        
        # 1. Validation and Grid Setup
        if h != w:
            raise ValueError("Error: The image must be square.")

        n = int(math.sqrt(num_cells))
        if n * n != num_cells or n == 0:
            raise ValueError(
                f"The 'num_cells' must be a positive perfect square, but got {num_cells}."
            )

        cell_size = w // n
        
        # 2. Contour Masking and Total Area Calculation
        contour_mask = np.zeros(image_shape, dtype=np.uint8)
        # Draw the single contour filled (1s) onto the mask
        cv2.drawContours(contour_mask, [contour], 0, 1, -1) 
        
        total_contour_area = np.sum(contour_mask)
        
        if total_contour_area == 0:
            return np.zeros((n, n), dtype=np.float32)

        proportion_matrix = np.zeros((n, n), dtype=np.float32)
        
        # 3. Iterate Through Grid Cells and Calculate Proportions
        for row_idx in range(n):
            for col_idx in range(n):
                
                # Calculate pixel boundaries of the current cell
                x_start = col_idx * cell_size
                y_start = row_idx * cell_size
                x_end = (col_idx + 1) * cell_size
                y_end = (row_idx + 1) * cell_size
                
                # Extract the portion of the contour mask that falls within this cell
                cell_slice = contour_mask[y_start:y_end, x_start:x_end]
                
                # Calculate the area (pixel count) of the contour inside this cell
                area_in_cell = np.sum(cell_slice)
                
                # Calculate the proportion (Area in Cell / Total Contour Area)
                proportion = area_in_cell / total_contour_area
                
                # Store the proportion in the result matrix
                proportion_matrix[row_idx, col_idx] = proportion
        
        return proportion_matrix
    
    # --- Image cleaning functions (The Combined Hybrid Method) ---
    @staticmethod
    def clean_contours_to_grid__hybrid_size_based(
        image_bin: np.ndarray,
        num_cells: int,
        min_area_threshold: int = 10,
        min_circularity_threshold: float = 0.8,
    ) -> np.ndarray:
        """
        Filters circular objects by size and shape. Then, it uses a hybrid approach 
        to assign them to a grid:
        1. If the object's bounding box is entirely smaller than a single cell, 
           it uses the **Centroid-Based** assignment (assigned to the cell containing its center).
        2. Otherwise (if the object is larger than a cell), it uses the 
           **Proportion-Based** assignment (assigned to dominant cells based on area proportion).

        Args:
            image_bin: The square binary input image (NumPy array, 0=black, 1=white).
            num_cells: The total number of cells (a positive perfect square).
            min_area_threshold: Minimum pixel area a contour must have.
            min_circularity_threshold: Minimum circularity score required.

        Returns:
            np.ndarray: The cleaned binary image (0s and 1s) with objects cropped 
                        to their respective cells based on the hybrid logic.
        """
        h, w = image_bin.shape

        # 1. Validation and Grid Setup
        if h != w:
            raise ValueError("Error: The input binary image is not square.")
        n = int(math.sqrt(num_cells))
        if n * n != num_cells or n == 0:
            raise ValueError(
                f"The input argument 'num_cells' must be a positive perfect square (e.g., 4, 9, 16), but got {num_cells}."
            )

        cell_size = w // n

        # 2. Preparation for Contours and Output
        final_output_image = np.zeros_like(image_bin, dtype=np.uint8)
        image_255 = (image_bin * 255).astype(np.uint8)

        # Find Contours
        contours, _ = cv2.findContours(
            image_255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        # 3. Iterate, Filter, and Process
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, closed=True)
            
            # Initial Filters (Size and Shape)
            if area < min_area_threshold or perimeter == 0:
                continue
            
            circularity_score = MathUtils.calc_contour_circularity(area, perimeter)
            if circularity_score < min_circularity_threshold:
                continue

            # Get Bounding Box to determine contour size (x, y, w_box, h_box)
            x, y, w_box, h_box = cv2.boundingRect(contour)

            # Create a temporary mask for the current contour (used by both methods)
            contour_mask = np.zeros_like(image_bin, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], 0, 1, -1) 
            
            # --- DECISION POINT: Centroid vs. Proportion ---
            
            # Condition: If the contour is smaller than a single cell in BOTH dimensions
            if w_box <= cell_size and h_box <= cell_size:
                # -----------------------------------------------------
                # A. Centroid-Based Logic (For small, confined objects)
                # -----------------------------------------------------
                
                M = cv2.moments(contour)
                if M["m00"] == 0: continue
                cX = M["m10"] / M["m00"]
                cY = M["m01"] / M["m00"]

                # Assign to Grid Cell based on centroid
                col_idx = max(0, min(n - 1, int(cX // cell_size)))
                row_idx = max(0, min(n - 1, int(cY // cell_size)))

                # Calculate pixel boundaries of the assigned cell
                x_start, y_start = col_idx * cell_size, row_idx * cell_size
                x_end, y_end = (col_idx + 1) * cell_size, (row_idx + 1) * cell_size

                # Crop and Draw: Keep only the pixels inside the assigned cell
                cell_mask = np.zeros_like(image_bin, dtype=np.uint8)
                cell_mask[y_start:y_end, x_start:x_end] = 1

                pixels_to_keep = np.logical_and(contour_mask, cell_mask)
                final_output_image[pixels_to_keep] = 1

            else:
                # -----------------------------------------------------
                # B. Proportion-Based Logic (For large, boundary-crossing objects)
                # -----------------------------------------------------
                
                # Get the raw proportion matrix
                proportion_matrix_raw = ImageUtils._get_single_contour_proportion_matrix(
                    image_shape=image_bin.shape,
                    contour=contour,
                    num_cells=num_cells
                )
                
                # Calculate modified proportion (as defined in the original proportion method)
                num_occupied_cells = np.sum(proportion_matrix_raw != 0)
                proportion_matrix = proportion_matrix_raw * num_occupied_cells 
                
                # Dynamic Thresholding based on circularity (as defined in the original proportion method)
                threshold_value = 3/4. 
                if circularity_score > 0.8:
                    # If very circular, force assignment to only the most dominant cell
                    threshold_value = 1 

                # Iterate through grid cells and apply threshold
                for r in range(n):
                    for c in range(n):
                        if proportion_matrix[r, c] >= threshold_value:
                            
                            # This cell (r, c) is chosen. Calculate its boundaries.
                            px_start, py_start = c * cell_size, r * cell_size
                            px_end, py_end = (c + 1) * cell_size, (r + 1) * cell_size
                            
                            # Create a mask for this cell
                            prop_cell_mask = np.zeros_like(image_bin, dtype=np.uint8)
                            prop_cell_mask[py_start:py_end, px_start:px_end] = 1
                            
                            # Keep pixels belonging to BOTH the contour AND this high-proportion cell
                            pixels_to_keep_prop = np.logical_and(contour_mask, prop_cell_mask)
                            final_output_image[pixels_to_keep_prop] = 1

        return final_output_image

    # --- Angle functions ---
    @staticmethod
    def find_dominant_angle(image_bin: ImageBinary, round_result: bool = False) -> Optional[float]:
        """
        Identifies the single most dominant line orientation (angle) in a binary image.

        This function is designed to work particularly well with shapes like crosses (X)
        by detecting two dominant perpendicular angles and normalizing them to a single
        representative orientation for correction/rotation purposes.

        Args:
            image_bin: A binary (0/255 or 0/1) NumPy array representing the image.
            round_result: If True (default), rounds the final angle to the nearest integer.
                          If False, returns the precise float value.
        Returns:
            Optional[float]: The dominant angle in degrees (0 to 90) for image
                             correction, or None if no significant lines are found.
        """
        # Find line segments from object contours in the binary image.
        lines = ImageUtils._find_lines_from_contours(image_bin)

        # Calculate the dominant angle
        dom_angle = LineUtils.find_dominant_angle(lines, round_result)

        # Return
        return dom_angle

    # --- Lines functions ---
    @staticmethod
    def _find_lines_from_contours(image_bin: ImageBinary) -> Optional[List[LineSegmentArray]]:
        """
        Converts a binary image (0/1 or 0/255) to 0/255, finds contours,
        draws those contours onto a new canvas, detects line segments
        using HoughLinesP, filters boundary-spanning segments, and returns them.

        Args:
            image_bin: A single-channel NumPy array (the binary image).

        Returns:
            Optional[List[LineSegment]]: A NumPy array of line segments (N, 1, 4)
                                  where each line is (x1, y1, x2, y2), or None if no
                                  lines are found.
        """
        # Validate the input image
        if image_bin.ndim != 2:
            raise ValueError("Error: Input must be a single-channel 2D image.")

        # --- 1. Preparation and Rescaling ---
        max_val = np.max(image_bin)
        if max_val <= 1.0:
            # Handles 0/1 range
            image_uint8 = (image_bin * 255).astype(np.uint8)
        else:
            # Handles 0/255 range
            image_uint8 = image_bin.astype(np.uint8)

        # --- 2. Find Contours ---
        # Find all shapes/boundaries in the image
        contours, _ = cv2.findContours(
            image_uint8,
            cv2.RETR_EXTERNAL,  # Only finds outer boundaries
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # --- 3. Create Contour-Only Canvas ---
        H, W = image_uint8.shape
        contour_canvas_image = np.zeros((H, W), dtype=np.uint8)

        # Draw the found contours onto the new canvas in white (255) with thickness 1
        cv2.drawContours(contour_canvas_image, contours, -1, color=255, thickness=1)

        # --- 4. Line Detection (Probabilistic Hough Transform) on the CONTOURS ---
        lines = cv2.HoughLinesP(
            contour_canvas_image,  # Detect lines ONLY on the contour boundaries
            rho=1,
            theta=np.pi / 180,
            threshold=30,  # Lower threshold since we're only looking at thin lines
            minLineLength=15,
            maxLineGap=5,
        )
        # ImageUtils.plot_lines(image_bin, lines, color='red', thickness=1, draw_end_points=True)

        # --- 5. Filter Lines (left only the lines that are not close to image boundary) ---
        # Define a small tolerance for "close to the edge" (e.g., 5 pixels)
        TOLERANCE = 5  #:SETTING:
        filtered_lines = []
        if lines is not None:
            # Define image dimensions
            H, W = image_uint8.shape  # Re-use H, W from Step 3 for clarity

            def is_point_close_to_boundary(x, y, w, h, tol):
                """Checks if a point is within the specified tolerance of any image edge."""
                # Convert to integer coordinates for comparison
                x, y = int(x), int(y)

                # Check proximity to the left (0) or top (0) edge
                if x < tol or y < tol:
                    return True

                # Check proximity to the right (w-1) or bottom (h-1) edge
                if x >= w - tol or y >= h - tol:
                    return True

                return False

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Check if center point is close to the boundary
                # p1_on_boundary = is_point_close_to_boundary(x1, y1, W, H, TOLERANCE)
                # p2_on_boundary = is_point_close_to_boundary(x2, y2, W, H, TOLERANCE)
                p_center_on_boundary = is_point_close_to_boundary((x1 + x2) / 2, (y1 + y2) / 2, W, H, TOLERANCE)

                # Keep the line ONLY if NOT both points are close to the boundary.
                # Rationale: We only want to discard lines that are exclusively part of the frame.
                # if not (p1_on_boundary and p2_on_boundary and p_center_on_boundary):
                if not (p_center_on_boundary):
                    filtered_lines.append(line)

            # Convert the list back to the standard NumPy array format
            filtered_lines = np.array(filtered_lines) if filtered_lines else None

        # --- 6. TEST ---
        # ImageUtils.plot_lines(image_bin, filtered_lines, color='red', thickness=1, draw_end_points=True)

        # --- 7. Return ---
        return filtered_lines

    # --- Image clip function ---
    @staticmethod
    def clip_center_square_by_ratio(image: Image, clip_ratio: float = 1.0) -> Image:
        """
        Clips a square section from the exact center of the input square image,
        where the size of the square is determined by a ratio of the original
        image side length. (Requires square input image).

        This method is implemented by delegating to clip_offset_square_by_ratio
        with a zero offset.

        Args:
            image: The square input image (e.g., 100x100) as a NumPy array.
            clip_ratio: The ratio [0.0, 1.0] defining the side length of the
                        output square relative to the original image side length.

        Returns:
            Image: The clipped, centered square image. Returns the original
                        image if the ratio is outside the valid range (0, 1].

        Raises:
            ValueError: If the input image is not square, ratios are invalid,
                        or clipping indices are out of bounds.
        """

        # Delegate to the offset clipping function with zero offset (0.0, 0.0)
        # Note: clip_offset_square_by_ratio handles the square image check and ratio validation.
        return ImageUtils.clip_offset_square_by_ratio(
            image=image, clip_ratio=clip_ratio, center_offset_ratio=(0.0, 0.0)
        )

    @staticmethod
    def clip_center_square(image: Image, new_size: int) -> Image:
        """
        Clips a square section from the exact center of the input image,
        specifying the size in pixels. (Requires square input image).

        This method uses delegation to clip_offset_square with a zero offset.

        Args:
            image: The square input image (e.g., 100x100) as a NumPy array.
            new_size: The desired size of the output square image (e.g., 80).

        Returns:
            Image: The clipped, centered square image (e.g., 80x80),
                        or the original image if new_size is too large.

        Raises:
            ValueError: If the input image is not square or the new_size is invalid.
        """
        # Check image size (assuming square input)
        height, width = image.shape[:2]

        if height != width:
            raise ValueError("Input image must be square (height == width).")

        original_size = height

        # If the new size is >= the original size, just return the original image.
        if new_size >= original_size:
            return image

        # Delegate to the offset clipping function with zero offset
        return ImageUtils.clip_offset_square(
            image=image, clip_size_pixels=new_size, center_offset_pixels=(0, 0)
        )

    @staticmethod
    def clip_offset_square(
        image: Image,
        clip_size_pixels: int,
        center_offset_pixels: Tuple[int, int] = (0, 0),
    ) -> Image:
        """
        Clips a square section from the input image at a specified center offset
        using direct pixel units.

        Args:
            image: The square input image (e.g., 100x100) as a NumPy array.
            clip_size_pixels: The side length of the square to clip (e.g., 50).
            center_offset_pixels: A tuple (offset_x, offset_y) in pixels relative
                                  to the image's center. Default is (0, 0) (centered clip).

        Returns:
            Image: The clipped square image.

        Raises:
            ValueError: If the input image is not square, the clip size is invalid,
                        or the clipping indices are out of bounds.
            RuntimeError: If the resulting clipped size does not match the expected size.
        """
        height, width = image.shape[:2]

        if height != width:
            raise ValueError(
                "Input image must be square (height == width) for this function."
            )

        original_size = height
        clip_side_length = clip_size_pixels
        offset_x, offset_y = center_offset_pixels

        # 1. Validate pixel size
        if clip_side_length <= 0 or clip_side_length > original_size:
            raise ValueError(
                f"Clip size {clip_side_length} is invalid or exceeds original image size."
            )

        # 2. Call the core pixel-based clipping logic
        return ImageUtils._clip_offset_square_core_pixels(
            image=image,
            clip_side_length=clip_side_length,
            offset_x=offset_x,
            offset_y=offset_y,
        )

    @staticmethod
    def clip_offset_square_by_ratio(
        image: Image,
        clip_ratio: float,
        center_offset_ratio: Tuple[float, float] = (0.0, 0.0),
    ) -> Image:
        """
        Clips a square section from the input image using only relative ratio parameters.

        Args:
            image: The square input image (e.g., 100x100) as a NumPy array.
            clip_ratio: A float representing the fraction of the original image
                        side length to use for the clip size (0.0 to 1.0).
            center_offset_ratio: A tuple (offset_ratio_x, offset_ratio_y) representing
                                 the shift of the clip center as a fraction of the
                                 original image side length. Default is (0.0, 0.0).

        Returns:
            Image: The clipped square image.

        Raises:
            ValueError: If the input image is not square, ratios are invalid,
                        or clipping indices are out of bounds.
            RuntimeError: If the resulting clipped size does not match the expected size.
        """
        height, width = image.shape[:2]

        if height != width:
            raise ValueError(
                "Input image must be square (height == width) for this function."
            )

        original_size = height

        # Validate clip_ratio
        if not (0.0 < clip_ratio <= 1.0):
            raise ValueError("clip_ratio must be between 0.0 and 1.0.")

        # --- Convert Ratios to Pixels ---

        # 1. Clip Size
        clip_side_length = int(round(original_size * clip_ratio))

        # 2. Offset Pixels
        offset_x_ratio, offset_y_ratio = center_offset_ratio
        offset_x_pixels = int(round(original_size * offset_x_ratio))
        offset_y_pixels = int(round(original_size * offset_y_ratio))

        # Validate pixel size
        if clip_side_length <= 0 or clip_side_length > original_size:
            raise ValueError(
                f"Calculated clip size {clip_side_length} is invalid or exceeds original image size."
            )

        # Call the core pixel-based clipping logic
        return ImageUtils._clip_offset_square_core_pixels(
            image=image,
            clip_side_length=clip_side_length,
            offset_x=offset_x_pixels,
            offset_y=offset_y_pixels,
        )

    @staticmethod
    def _clip_offset_square_core_pixels(
        image: Image, clip_side_length: int, offset_x: int, offset_y: int
    ) -> Image:
        """
        Core logic to perform the image slicing based on computed pixel values.

        This method is designed to be called internally by public methods
        after all parameters have been converted to integer pixel units and validated.

        Raises:
            ValueError: If the clipping indices are out of bounds.
            RuntimeError: If the resulting clipped size does not match the expected size.
        """

        original_size = image.shape[0]

        half_clip = clip_side_length / 2.0
        center_original = original_size / 2.0

        # Calculate the target center point for the new clip
        target_center_x = center_original + offset_x
        target_center_y = center_original + offset_y

        # Calculate Slicing Indices
        # Calculate START index using rounding
        start_x = int(round(target_center_x - half_clip))
        start_y = int(round(target_center_y - half_clip))

        # Calculate END index by adding the clip_side_length to the start index.
        # This guarantees the resulting slice size will be exactly clip_side_length,
        # avoiding a secondary rounding error.
        end_x = start_x + clip_side_length
        end_y = start_y + clip_side_length

        # Final Boundary Check (Ensure indices are within the image limits)
        if start_x < 0 or start_y < 0 or end_x > original_size or end_y > original_size:
            raise ValueError(
                f"Clip indices are out of bounds. "
                f"Clip: [({start_x}:{end_x}), ({start_y}:{end_y})]. "
                f"The requested clip exceeds the image boundaries."
            )

        # Perform Slicing
        clipped_image = image[start_y:end_y, start_x:end_x, ...]

        # The RuntimeError check is technically redundant now, but kept for strictness
        # Ensure the clipped image size matches the requested size
        if (
            clipped_image.shape[0] != clip_side_length
            or clipped_image.shape[1] != clip_side_length
        ):
            raise RuntimeError(
                f"Clipping error: Expected {clip_side_length}x{clip_side_length}, got {clipped_image.shape[0]}x{clipped_image.shape[1]}."
            )

        return clipped_image

    # --- Split image functions ---
    @staticmethod
    def split_image_to_binary_matrix(
        image_bin: ImageBinary,
        numb_parts: int,
        black_ratio_down_threshold: float = 0.3,
        black_ratio_up_threshold: float = 0.7,
    ) -> ImageBinary:
        """
        Splits a square binary image into a square matrix (4 or 9 parts) and converts
        each segment into a binary value (0 or 1) based on a specified black pixel
        ratio range.

        The classification rule is:
        - 1 (Within Range): Only if the black pixel ratio is > down_threshold AND < up_threshold.
        - 0 (Outside Range): If the ratio is below the down threshold or above the up threshold.

        Args:
            image_bin: The square binary input image (values 0 or 1).
            numb_parts: The total number of segments in the grid (must be 4 for 2x2 or 9 for 3x3).
            black_ratio_up_threshold: The upper boundary for the black pixel ratio.
                                      The ratio must be strictly less than this value to result in 1. Default is 0.7.
            black_ratio_down_threshold: The lower boundary for the black pixel ratio.
                                        The ratio must be strictly greater than this value to result in 1. Default is 0.3.

        Returns:
            ImageBinary: An (N x N) matrix of 0s and 1s.

        Raises:
            ValueError: If the input image is not square, the total number of parts is invalid,
                        the thresholds are invalid, or the image size is not divisible by the grid side length.
        """
        if image_bin.ndim > 2:
            image_bin = image_bin.squeeze()

        height, width = image_bin.shape[:2]

        if height != width:
            raise ValueError("Input image must be square (height == width).")

        original_size = height

        # 1. Determine the side length (N) from the total number of parts
        if numb_parts == 4:
            N = 2
        elif numb_parts == 9:
            N = 3
        else:
            raise ValueError(
                f"numb_parts must be 4 (for 2x2) or 9 (for 3x3), got {numb_parts}."
            )

        # 2. Validate thresholds and Divisibility
        if not (0.0 <= black_ratio_down_threshold < black_ratio_up_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy: 0.0 <= down < up <= 1.0."
                f"It is ({black_ratio_down_threshold:.2f},{black_ratio_up_threshold:.2f})."
            )

        if original_size % N != 0:
            raise ValueError(
                f"Image side size ({original_size}) must be exactly divisible by the grid side length N ({N})."
            )

        cell_size = original_size // N

        # Initialize the result matrix (N x N)
        result_matrix = np.zeros((N, N), dtype=int)
        total_pixels = cell_size * cell_size

        # 3. Process each segment
        for r in range(N):
            for c in range(N):
                # Calculate slice indices
                start_y = r * cell_size
                end_y = (r + 1) * cell_size
                start_x = c * cell_size
                end_x = (c + 1) * cell_size

                # Extract the segment
                segment = image_bin[start_y:end_y, start_x:end_x]

                # Count black pixels (value 0)
                white_pixels = np.sum(segment)
                black_pixels = total_pixels - white_pixels

                # Calculate black pixel ratio
                black_ratio = black_pixels / total_pixels
                # print(f"=== TEST: black_ratio: {black_ratio}") #@TEST

                # 4. Determine binary output for the matrix element based on the range
                if (
                    black_ratio_down_threshold
                    <= black_ratio
                    <= black_ratio_up_threshold
                ):
                    result_matrix[r, c] = 1
                elif black_ratio < black_ratio_down_threshold:
                    result_matrix[r, c] = 0
                elif black_ratio_up_threshold < black_ratio:
                    result_matrix[r, c] = -1
                else:
                    result_matrix[r, c] = None

        return result_matrix

    # --- Plot and Draw image functions ---
    @staticmethod
    def plot_image(
        image: Image, title: Optional[str] = None, assume_opencv_image: bool = True
    ) -> None:
        """
        Plots an image using Matplotlib, handling both BGR/RGB color and grayscale formats.

        Args:
            image (np.ndarray): The image to plot (expected to be a NumPy array).
            title (Optional[str]): The title of the image plot.
            assume_opencv_image (bool): If True, assumes the image is a BGR color
                                        image from OpenCV and converts it to RGB
                                        for correct plotting with Matplotlib.
        """
        # Check if the image has 3 color channels (like BGR or RGB)
        if len(image.shape) == 3:
            if assume_opencv_image:
                # Assume it's a BGR image from OpenCV and convert to RGB
                # for correct plotting with Matplotlib.
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(image_rgb)
            else:
                # If not assumed BGR, plot directly (assuming RGB or other format)
                plt.imshow(image)
        # Check if the image is a 2D array (grayscale)
        elif len(image.shape) == 2:
            # For grayscale images, specify the 'gray' colormap.
            plt.imshow(image, cmap="gray")
        else:
            # Handle cases where the image format is not supported.
            raise ValueError("Unsupported image format. Cannot plot.")

        if title is not None:
            plt.title(title)

        # plt.axis('off')  # Hide axis for better visualization (commented out in original)
        plt.show()

    @staticmethod
    def plot_contours(
        image: Image,
        contours: List[np.ndarray],
        color: str = "green",
        thickness: int = 1,
    ) -> None:
        """
        A function to plot contours on an image.
        It assumes the input image is in BGR format and plots it correctly.

        Args:
            image (np.ndarray): The original image (in BGR format).
            contours (List[np.ndarray]): A list of contours (e.g., from cv2.findContours) to draw on the image.
            color (str): The color name (e.g., "green") for the contours.
            thickness (int): The thickness of the contour lines.
        """
        # Create a copy of the original image to draw contours on
        image_with_contours = image.copy()

        # Draw all contours on the copied image
        bgr_color = ColorUtils.standardize_color_input(color)
        cv2.drawContours(
            image_with_contours,
            contours,
            contourIdx=-1,
            color=bgr_color,
            thickness=thickness,
        )

        # Plot the image with contours using the debug plot function
        ImageUtils.plot_image(image_with_contours, title="Contours Plotted")

    @staticmethod
    def plot_polylines(
        image: Image,
        polylines: List[np.ndarray],
        isClosed: bool = True,
        color: str = "blue",
        thickness: int = 1,
    ) -> None:
        """
        A function to plot polylines on an image.
        It assumes the input image is in BGR format and plots it correctly.

        Args:
            image (np.ndarray): The original image (in BGR format).
            polylines (List[np.ndarray]): A list of polylines to draw on the image.
            isClosed (bool): Whether the polylines are closed shapes.
            color (str): The color name (e.g., "blue") for the polylines.
            thickness (int): The thickness of the polyline lines.
        """
        # Create a copy of the original image to draw polylines on
        image_with_polylines = image.copy()

        # Draw all polylines on the copied image
        bgr_color = ColorUtils.standardize_color_input(color)
        cv2.polylines(
            image_with_polylines,
            polylines,
            isClosed=isClosed,
            color=bgr_color,
            thickness=thickness,
        )

        # Plot the image with polylines using the debug plot function
        ImageUtils.plot_image(image_with_polylines, title="Polylines Plotted")

    @staticmethod
    def draw_box_points(
        image: Image, box_points: np.ndarray, color: str = "red", thickness: int = 1
    ) -> None:
        """
        A function to draw a box defined by 4 points (e.g., from cv2.minAreaRect) on an image.

        Args:
            image (np.ndarray): The image to draw on (modified in place).
            box_points (np.ndarray): A 4x2 or 4x1x2 array of points defining the box vertices.
            color (str): The color name (e.g., "red") of the box.
            thickness (int): The thickness of the box lines.
        """
        bgr_color = ColorUtils.standardize_color_input(color)
        # np.int0 converts points to integer format suitable for cv2.polylines
        cv2.polylines(image, [np.int0(box_points)], True, bgr_color, thickness)

    @staticmethod
    def draw_rectangle(
        image: Image,
        rect_min_pt: Tuple[int, int],
        rect_max_pt: Tuple[int, int],
        color: str = "blue",
        thickness: int = 1,
    ) -> None:
        """
        Draws a rectangle defined by its top-left and bottom-right corners on an image.

        Args:
            image (np.ndarray): The image to draw on (modified in place).
            rect_min_pt (Tuple[int, int]): The top-left corner (x, y).
            rect_max_pt (Tuple[int, int]): The bottom-right corner (x, y).
            color (str): The color name (e.g., "blue") of the rectangle.
            thickness (int): The thickness of the rectangle lines.
        """
        bgr_color = ColorUtils.standardize_color_input(color)
        cv2.rectangle(image, rect_min_pt, rect_max_pt, bgr_color, thickness)

    @staticmethod
    def draw_line(
        image: Image,
        point1_xy_coord: Tuple[int, int],
        point2_xy_coord: Tuple[int, int],
        color: str = "blue",
        thickness: int = 1,
        draw_end_points: bool = False
    ) -> None:
        """
        Draws a line between two specified points on an image.

        Args:
            image (np.ndarray): The image to draw on (modified in place).
            point1_xy_coord (Tuple[int, int]): The starting point (x, y).
            point2_xy_coord (Tuple[int, int]): The ending point (x, y).
            color (str): The color name (e.g., "blue") of the line.
            thickness (int): The thickness of the line.
            draw_end_points (bool): Enable/Disable the end points drawing.
        """
        bgr_color = ColorUtils.standardize_color_input(color)
        cv2.line(image, point1_xy_coord, point2_xy_coord, bgr_color, thickness)

        if draw_end_points: 
            cv2.circle(image, point1_xy_coord, 2, (255, 0, 0), 1)
            cv2.circle(image, point2_xy_coord, 3, (0, 255, 0), 1)

    @staticmethod
    def draw_center_circle(
        image: ImageBinary, radius: int, color: str = "blue", thickness: int = 1
    ) -> None:
        """
        Draws a circle at the exact center of the image.

        Args:
            image (np.ndarray): The image to draw on (modified in place).
            radius (int): The radius of the circle in pixels.
            color (str): The color name (e.g., "blue") of the circle.
            thickness (int): The thickness of the circle outline.
        """
        bgr_color = ColorUtils.standardize_color_input(color)
        h, w = ImageUtils.get_image_dimensions(image)
        center_x, center_y = w // 2, h // 2
        cv2.circle(image, (center_x, center_y), radius, bgr_color, thickness)

    @staticmethod
    def plot_lines(
        image: Image,
        lines: Union[List[np.ndarray], None],
        color: str = "blue",
        thickness: int = 1,
        draw_end_points: bool = False
    ) -> None:
        """
        Draws and plots lines found on an image.

        This function first ensures the image is color (by converting grayscale if necessary),
        then draws the lines, and finally plots the result using Matplotlib.

        Args:
            image (np.ndarray): The input image.
            lines (Union[List[np.ndarray], None]): A list of lines, where each line is typically
                                                    [x1, y1, x2, y2]. Can be None if no lines are found.
            color (str): The color name (e.g., "blue") of the lines.
            thickness (int): The thickness of the lines.
            draw_end_points (bool): Enable/Disable the line end points drawing.
        """
        # Create color image (if grayscale, converts to BGR)
        color_image = ImageUtils.convert_image_to_color(image)

        # Draw lines
        line_count = 0
        if lines is not None:
            line_count = len(lines)
            for line in lines:
                # Expects line format like [[x1, y1, x2, y2]] (common OpenCV format)
                x1, y1, x2, y2 = line[0]
                ImageUtils.draw_line(color_image, (x1, y1), (x2, y2), color, thickness, draw_end_points)


        # Plotting
        ImageUtils.plot_image(color_image, title=f"Number lines: {line_count}")

    # --- Generate image functions ---
    @staticmethod
    def generate_sample_image(height: int, width: int) -> Image:
        """
        Generates a simple, grayscale image of the specified size and returns it.

        Args:
            width: The desired width of the image in pixels.
            height: The desired height of the image in pixels.
        """
        # Create a blank grayscale image (one channel) initialized to a medium gray (127)
        # The data type 'np.uint8' is standard for image pixels (0-255).
        image = np.full((height, width), 127, dtype=np.uint8)

        # Optional: Draw a white rectangle in the center for better visualization
        # Define corners (20% padding on each side)
        rect_start_x = int(width * 0.2)
        rect_start_y = int(height * 0.2)
        rect_end_x = int(width * 0.8)
        rect_end_y = int(height * 0.8)

        # Draw the rectangle (color 255 is white in grayscale)
        cv2.rectangle(
            image,
            (rect_start_x, rect_start_y),
            (rect_end_x, rect_end_y),
            255,  # White color
            -1,
        )  # -1 means fill the rectangle

        return image

    @staticmethod
    def create_blank_image(
        width: int, height: int, color: tuple = (255, 255, 255)
    ) -> Image:
        """
        Creates and returns an empty BGR color image (NumPy array) of the specified
        width and height, filled with the given color.

        OpenCV uses the BGR (Blue, Green, Red) channel ordering by default,
        and colors are typically represented by 8-bit unsigned integers (0-255).

        Args:
            width: The desired width of the image in pixels.
            height: The desired height of the image in pixels.
            color: The BGR color tuple (B, G, R) to fill the image.
                Defaults to white: (255, 255, 255).

        Returns:
            A NumPy array representing the image.
        """
        # Image shape is (height, width, channels). Channels=3 for BGR color image.
        shape = (height, width, 3)

        # np.full is efficient for creating an array filled with a single value or tuple.
        # We use np.uint8, which is the standard data type for pixel values (0-255).
        blank_image = np.full(shape, color, dtype=np.uint8)

        return blank_image

    @staticmethod
    def create_square_blank_image(size: int, color_str: str = "white") -> Image:
        color = ColorUtils.standardize_color_input(color_str)
        image = ImageUtils.create_blank_image(width=size, height=size, color=color)
        return image

    # --- Color checking functions ---
    @staticmethod
    def is_image_almost_white(image: Image, min_white_ratio: float = 0.95) -> bool:
        """
        Checks if an image is predominantly white (i.e., the ratio of white
        (non-zero) pixels to total pixels is greater than the specified minimum ratio).

        This is designed for binary or grayscale images where non-zero values
        represent the foreground/white area.

        Args:
            image: The input image (NumPy array). Can be 2D (grayscale) or 3D (color).
            min_white_ratio: The minimum fraction (0.0 to 1.0) of white pixels required
                            for the image to be considered "almost white". (Default: 0.95)

        Returns:
            bool: True if the image is almost entirely white, False otherwise.

        Raises:
            ValueError: If the min_white_ratio is outside the valid range [0.0, 1.0].
        """
        if not 0.0 <= min_white_ratio <= 1.0:
            raise ValueError("min_white_ratio must be between 0.0 and 1.0.")

        # Flatten the array to simplify counting, and convert to boolean mask (True where image > 0)
        white_pixels_mask = image.flatten() > 0

        # Count the number of white pixels
        num_white_pixels = np.sum(white_pixels_mask)

        # Calculate the total number of pixels in the image
        total_pixels = image.size

        if total_pixels == 0:
            return False

        # Calculate the ratio
        white_ratio = num_white_pixels / total_pixels

        return white_ratio >= min_white_ratio

    # --- SVG & PDF functions ---
    @staticmethod
    def save_svg(file_path: str, svg_content: str) -> None:
        """
        Saves SVG content (a string) to the specified file path.

        This function opens the file in write mode ("w") with UTF-8 encoding
        and writes the provided SVG string content.

        Args:
            file_path (str): The path to the file where the SVG content should be saved.
            svg_content (str): The SVG content as a string.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(svg_content)

    @staticmethod
    def save_pdf(file_path: str, svg_content: str) -> None:
        """
        Saves the SVG content to a PDF file by converting it using CairoSVG.

        This function requires the external 'cairosvg' library to perform
        the SVG-to-PDF conversion.

        Args:
            file_path (str): The full path where the PDF file should be saved (e.g., 'output.pdf').
            svg_content (str): The SVG content string.
        """
        # CairoSVG takes a bytestring for in-memory content
        svg_bytes = svg_content.encode("utf-8")
        cairosvg.svg2pdf(bytestring=svg_bytes, write_to=file_path)

    @staticmethod
    def plot_histogram_from_image(image: Image, value: int = None, title: str = 'Image Analysis') -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Checks if the input image is color (3 channels) or grayscale (2D array).
        Converts color images to grayscale, calculates the histogram, and visualizes both.

        :param image: The image as a NumPy array (np.ndarray), which can be color or grayscale.
        :param value: An integer value to display on the histogram (e.g., mean intensity).
        :param title: Optional title for the plot window.
        :return: A tuple containing the processed grayscale image and the histogram array, 
                or None if the input is invalid.
        """
        
        # --- 1. Basic Image Validation and Grayscale Conversion ---
        if image is None:
            print("Error: The input image is None.")
            return None

        # Check number of dimensions/channels
        if image.ndim == 3:
            # Image is color (e.g., BGR). Convert it to grayscale.
            print("Image is color (3 channels). Converting to grayscale...")
            grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            # Image is already grayscale (2D array).
            print("Image is already grayscale (2 channels/2D array).")
            grayscale_img = image
        else:
            print(f"Error: Invalid number of dimensions ({image.ndim}). Expected 2 or 3.")
            return None
            
        # Ensure the data type is np.uint8 for standard histogram calculation
        if grayscale_img.dtype != np.uint8:
            grayscale_img = cv2.convertScaleAbs(grayscale_img)
            print("Warning: Image data type was adjusted to np.uint8.")


        # --- 2. Calculate the Histogram ---
        # We use the processed grayscale_img
        histogram = cv2.calcHist(
            [grayscale_img], # The image (in a list)
            [0],             # Channel 0 for grayscale
            None,            # No mask used
            [256],           # 256 bins for 8-bit image (0-255)
            [0, 256]         # Range of intensity
        )

        # --- 3. Calculate Mean Intensity (for visualization) ---
        value_label = 'Value'
        if value is None:
            value_label = 'Mean value'
            value = np.mean(grayscale_img)

        # --- 4. Visualization (Plotting) ---
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(title, fontsize=16)

        # 4.1. Display the Grayscale Image
        axes[0].imshow(grayscale_img, cmap='gray')
        axes[0].set_title('Processed Grayscale Image')
        axes[0].axis('off')

        # 4.2. Display the Histogram
        axes[1].plot(histogram, color='gray')
        axes[1].axvline(x=value, color='r', linestyle='--', linewidth=1, 
                        label=f'{value_label}: {value:.2f}')
        
        axes[1].set_title('Intensity Histogram')
        axes[1].set_xlabel('Pixel Value (0 - Black, 255 - White)')
        axes[1].set_ylabel('Frequency (Number of Pixels)')
        axes[1].set_xlim([0, 256])
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        return grayscale_img, histogram