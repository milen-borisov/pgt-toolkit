"""
PGTEngine class - Core utilities for photogrammetry targets

This module implements the central, high-level operations for processing
photogrammetry targets, forming the backbone of the PGT-Toolkit project.

For now, the class provides utilities for handling:
* CCT (Circular Coded Targets)
* HRCT (Human Readable Coded Targets)

Core functions available via the class include:
* Detection and decoding of targets in raw image data.
* Visualization (drawing) of found targets onto images.
* Exporting target data (ID, coordinates, size, etc.) to various CSV formats.
* Generating printable target patterns (PNG, JPG, SVG, PDF, etc. files).

This module is part of the PGT-Toolkit project.
"""

# Import external modules
import csv
import numpy as np
from pathlib import Path
from typing import List, Optional

# Import internal modules
from targets.target_data import TargetData
from targets.target_type import TargetTypeUtils
from targets.cct import CCTDecoder, CCTDrawer, CCTGenerator
from targets.hrct import HRCTDecoder, HRCTDrawer, HRCTGenerator
from utils.image import ImageUtils, Image
from app.logger import logger as log
from app.settings import settings


# --------------------------------------------
# Class: Photogrammetry Target Engine
# Handles all high-level operations on photogrammetry targets
# --------------------------------------------
class PGTEngine:
    # --- Find targets in image ---
    @staticmethod
    def find_targets_in_image(
        target_type: str,
        image: np.ndarray,
        image_file_name: str,
        threshold_modes: Optional[
            List[str]
        ],  # Accepts None or ['adaptive', 'otsu', 'max']
        threshold_values: Optional[List[int]],
        undecoded_targets_unique_index_ref: List[int],
    ) -> List[TargetData]:
        """
        Initializes the appropriate decoder and runs target finding on a single image.

        This method handles:
        1. Instantiating the correct decoder (CCTDecoder or HRCTDecoder).
        2. Calling the decoder's main finding routine.
        3. Assigning unique indexes to any undecoded targets.
        4. Conditionally sorting the found targets based on settings.

        Args:
            target_type (str): The type of target to look for (e.g., 'CCT', 'HRCT').
            image (Image): The image array (OpenCV format) to process.
            image_file_name (str): The name of the file being processed.
            threshold_modes (Optional[List[str]]): List of thresholding techniques to try.
            threshold_values (Optional[List[int]]): List of fixed threshold values to try.
            undecoded_targets_unique_index_ref (List[int]): A list used as a mutable
                reference to maintain a globally unique index count for undecoded targets
                across multiple images.

        Returns:
            List[Target]: A list of found Target objects.

        Raises:
            ValueError: If the target_type is not supported by the finding functionality.
        """
        # Initialize the appropriate TargetDecoder according by the input target_type
        decoder = None
        if TargetTypeUtils.is_cct_target(target_type):
            decoder = CCTDecoder()
        elif TargetTypeUtils.is_hrct_target(target_type):
            decoder = HRCTDecoder()
        else:
            raise ValueError(
                f"Find functionality is still not implemented for '{target_type}' target."
            )

        # Find targets in the image
        found_targets = decoder.find_targets(
            target_type, image, image_file_name, threshold_modes, threshold_values
        )

        # Process the found undecoded targets
        found_targets = decoder.process_undecoded_targets(
            found_targets, undecoded_targets_unique_index_ref
        )

        # Sort the found targets by id
        sort_targets_by_id_on = settings.get(
            "targets.export_to_csv_sort_targets_by_id_on"
        )
        if sort_targets_by_id_on:
            found_targets = sorted(
                found_targets,
                key=lambda target: (
                    (0, target.id) if isinstance(target.id, int) else (1, target.id)
                ),
            )

        # Return the found targets
        return found_targets

    # --- Draw targets to image ---
    @staticmethod
    def draw_targets(targets: list[TargetData], image: Image) -> Image:
        """ "
        Draws the input targets on a copy of the input image.

        Args:
            targets (List[Target]): A list of Target objects containing data
                                    (ID, Center_X_px, Center_Y_px, Size_px).
            image (Image): The input OpenCV image used as the background for
                            target drawing.
        Returns:
            np.ndarray: A copy of the input image with drawn targets.

        Raises:
            ValueError: If the target's type is not supported by a defined Drawer.
        """
        # Copy the input image
        res_image = image.copy()

        # Draw the targets on the res_image
        for target in targets:
            # Initialize the appropriate TargetDrawer according by the input target_type
            drawer = None
            if target.is_cct_target():
                drawer = CCTDrawer
            elif target.is_hrct_target():
                drawer = HRCTDrawer
            else:
                raise ValueError(
                    f"Draw functionality is still not implemented for '{target.type}' target."
                )

            # Draw the input target
            res_image = drawer.draw_target(target, res_image)

        # Return the image with drawn targets
        return res_image

    # --- Export targets to CSV files ---
    @staticmethod
    def export_targets_to_csv(
        targets: list[TargetData], csv_filepath: str, mode="w", delimiter=","
    ):
        """
        Exports the targets data to a standard CSV file format.

        This is a general-purpose export containing comprehensive data for each target.

        Args:
            targets (List[TargetData]): List of TargetData objects to export.
            csv_filepath (str): The path to the CSV file to write.
            mode (str): File open mode (default 'w' for overwrite).
            delimiter (str): CSV delimiter (default ',').
        """
        with open(csv_filepath, mode, newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=delimiter)

            # Write the header row
            if csv_file.tell() == 0:
                csv_writer.writerow(
                    [
                        "ImageFileName",
                        "Target_Type",
                        "Target_ID",
                        "Center_X_px",
                        "Center_Y_px",
                        "Size_px",
                        "Img_Thresh_Mode",
                        "Img_Thresh_Value",
                        "IDs_raw",
                        "ID_Thresh_Mode",
                        "ID_Thresh_Value",
                        "ID_Thresh_Is_Alt",
                        "Geom_Type",
                        "Geom_Data",
                    ]
                )

            # Write targets data
            float_decimal_places = settings.get("targets.export_to_csv_decimal_places")
            float_format_spec = f".{float_decimal_places}f"
            for target in targets:
                row_data = [
                    target.image_file_name,
                    target.type,
                    target.id,
                    format(target.center_pt_x, float_format_spec),
                    format(target.center_pt_y, float_format_spec),
                    format(target.size, float_format_spec),
                    target.image_threshold_modes if target.image_threshold_modes else "none",
                    target.image_threshold_values if target.image_threshold_values is not None else "none",
                    target.ids_raw if target.id_raw else "none",
                    target.id_threshold_modes if target.id_threshold_modes else "none",
                    target.id_threshold_values if target.id_threshold_values is not None else "none",
                    target.id_threshold_is_alternatives,
                    target.geometry_type,
                    target.geometry_data,
                ]
                csv_writer.writerow(row_data)

    @staticmethod
    def export_targets_to_metashape_csv(
        targets: list[TargetData], csv_filepath: str, mode="w", delimiter=","
    ):
        """
        Exports the targets data in a format suitable for Agisoft Metashape.

        This format requires the image name (without extension), target ID,
        and pixel coordinates (X, Y).

        Args:
            targets (List[TargetData]): List of TargetData objects to export.
            csv_filepath (str): The path to the CSV file to write.
            mode (str): File open mode (default 'w' for overwrite).
            delimiter (str): CSV delimiter (default ',').
        """
        with open(csv_filepath, mode, newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=delimiter)

            # Write the header row
            if csv_file.tell() == 0:
                csv_writer.writerow(["ImageFileName", "Target_ID", "X_px", "Y_px"])

            # Write targets data
            float_decimal_places = settings.get("targets.export_to_csv_decimal_places")
            float_format_spec = f".{float_decimal_places}f"
            for target in targets:
                row_data = [
                    target.get_image_name(),
                    target.id,
                    format(target.center_pt_x, float_format_spec),
                    format(target.center_pt_y, float_format_spec),
                ]
                csv_writer.writerow(row_data)

    @staticmethod
    def export_targets_to_meshroom_csv(
        targets: list[TargetData], csv_filepath: str, mode="w", delimiter=","
    ):
        """
        Exports the targets data in a format suitable for Meshroom (AliceVision).

        The required order for Meshroom is typically X_px, Y_px, ImageFileName, Target_ID, Size_px,
        and it does not include a header row.

        Args:
            targets (List[Target]): List of Target objects to export.
            csv_filepath (str): The path to the CSV file to write.
            mode (str): File open mode (default 'w' for overwrite).
            delimiter (str): CSV delimiter (default ',').
        """
        with open(csv_filepath, mode, newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=delimiter)

            # Write targets data
            float_decimal_places = settings.get("targets.export_to_csv_decimal_places")
            float_format_spec = f".{float_decimal_places}f"
            for target in targets:
                row_data = [
                    format(target.center_pt_x, float_format_spec),
                    format(target.center_pt_y, float_format_spec),
                    target.image_file_name,
                    target.id,
                    format(target.size, float_format_spec),
                ]
                csv_writer.writerow(row_data)

    # --- Generate printable targets image (PNG, JPG, SVG, PDF, etc.) ---
    @staticmethod
    def generate_targets(
        target_type: str,
        res_folder_path: str,
        res_image_type: str,
        res_image_size: int,
        res_max_images_to_generate: int,
        progress_callback=None,
    ) -> int:
        """
        Generates images of the specified target type and saves them to a folder.

        This method validates generation parameters and initializes the correct
        TargetGenerator (CCT or HRCT) to produce synthetic target images.

        Args:
            target_type (str): The type of target to generate (e.g., 'CCT_12bit_black', 'HRCT_9dots_black').
            res_folder_path (str): The root folder path to save the generated images.
            res_image_type (str): The file extension/type for the generated images
                                  (e.g., 'jpg', 'png', 'svg', 'pdf').
            res_image_size (int): The target size in pixels (e.g., 500).
            res_max_images_to_generate (int): The maximum number of images to create.
                                              Use '-1' or 'all' to generate all possible unique images.
            progress_callback: Optional callback for reporting generation progress.

        Returns:
            int: The total number of targets successfully generated.

        Raises:
            ValueError: If image size is out of range, image type is unsupported,
                        or target type has no implemented generator.
        """
        # Validate the input arguments
        if res_image_size < 40 or res_image_size > 4000:
            raise ValueError(
                f"Invalid image size of {res_image_size} pixels. Expected between 40 and 4000 pixels."
            )

        if not res_image_type in ImageUtils.SUPPORT_GENERATED_IMAGE_TYPES:
            raise ValueError(
                f"Unsupported image type '{res_image_type}'. Expected types: {ImageUtils.SUPPORT_GENERATED_IMAGE_TYPES}."
            )

        # Initialize the appropriate TargetGenerator according by the input target_type
        generator = None
        if TargetTypeUtils.is_cct_target(target_type):
            generator = CCTGenerator
        elif TargetTypeUtils.is_hrct_target(target_type):
            generator = HRCTGenerator
        else:
            raise ValueError(
                f"Draw functionality is still not implemented for '{target_type}' target."
            )

        # Generate targets
        numb_generated_targets = generator.generate_targets(
            target_type,
            res_folder_path,
            res_image_type,
            res_image_size,
            res_max_images_to_generate,
            progress_callback,
        )

        # Return
        return numb_generated_targets

    # --- Reload settings from file ---
    @staticmethod
    def reload_settings():
        """Reload global setting from the settings file"""
        settings.reload()
