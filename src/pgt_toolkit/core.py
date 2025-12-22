"""
This module contains the core runner functions for the PGT-Toolkit.

It is designed to be called by the main CLI and GUI interfaces,
and the test harness.
"""

# Import external modules
import os
import sys

from pathlib import Path
from typing import Optional, List, Tuple, Any

# Import internal modules that are from the pgt_toolkit folder
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from app.logger import logger as log
from utils.image import ImageUtils
from targets.targets_engine import PGTEngine
from app.settings import settings


# --------------------------------------------
# Core Function: Find Targets
# --------------------------------------------
def run_find_targets(
    target_type: str,
    input_images_folder: str,
    res_folder_path: str,
    res_csv_file_name: str,
    threshold_modes: Optional[List[str]] = None,  # Accepts None or ['adaptive', 'otsu', 'max', 'triangle', 'mean']
    threshold_values: Optional[List[int]] = None,
    draw_found_targets_in_images: bool = True,
    export_found_targets_in_csv_meshroom: bool = True,
    export_found_targets_in_csv_metashape: bool = True,
    progress_callback: Optional[
        Any
    ] = None,  # Placeholder for progress signaling object
) -> int:
    """
    Runs target detection across all image files in a specified folder,
    managing the full workflow including logging, result saving, and data export.

    Args:
        target_type (str): The type of target to search for (e.g., 'A1', 'checkerboard').
        input_images_folder (str): Path to the folder containing input images.
        res_folder_path (str): Path to the folder where all results (images, CSVs) will be saved.
        res_csv_file_name (str): Base name for the main result CSV file.
        threshold_modes (Optional[List[str]]): List of thresholding methods to use.
        threshold_values (Optional[List[int]]): List of corresponding threshold values.
        draw_found_targets_in_images (bool): If True, draws detected targets on images and saves them.
        export_found_targets_in_csv_meshroom (bool): If True, exports a Meshroom-compatible CSV.
        export_found_targets_in_csv_metashape (bool): If True, exports a Metashape-compatible CSV.
        progress_callback (Optional[Any]): A callback object (e.g., a signal in Qt) for reporting progress.

    Returns:
        int: The total count of targets found across all processed images.
    """
    # Reload settings before starting the run to ensure all configuration is up-to-date
    PGTEngine.reload_settings()

    # Start logging for the entire process
    log.info_start(f"Find Targets")

    # --- Setup and Validation ---

    # Convert string path to Path object for validation
    input_images_folder_path = Path(input_images_folder)

    # Check if the input folder exists
    if not input_images_folder_path.exists():
        log.error(f"Input folder '{input_images_folder}' with images not found.")
        return 0

    # Ensure that the results folder exists (creates directories if they don't)
    os.makedirs(res_folder_path, exist_ok=True)

    # Initialize the master list to collect all targets found across all images
    found_targets = []

    # Find all image paths in the input folder
    image_paths_to_process = ImageUtils.find_all_images_path_in_folder(
        input_images_folder_path
    )
    total_image_count = len(image_paths_to_process)

    log.info(
        f"Begin finding '{target_type}' targets in {total_image_count} images in folder '{input_images_folder_path}'."
    )

    # Initialize a reference for unique indexing of targets that couldn't be decoded
    # This list holds a single integer that is passed by reference across function calls
    undecoded_targets_unique_index_ref = [1]

    # --- Image Iteration and Detection ---

    for ii, image_path in enumerate(image_paths_to_process):
        # Report progress via the optional callback
        if progress_callback:
            progress_callback.emit(100 * ii / total_image_count)

        log.info_separator_h1()
        log.info(
            f"Finding targets in image '{image_path.name}' ({ii+1}/{total_image_count})"
        )

        # Load the image using the utility function
        image = ImageUtils.load_image(image_path)
        image_file_name = os.path.basename(image_path)

        # Perform the actual target detection
        curr_found_targets = PGTEngine.find_targets_in_image(
            target_type,
            image,
            image_file_name,
            threshold_modes,
            threshold_values,
            undecoded_targets_unique_index_ref,
        )

        log.info_separator_h2()
        log.info(
            f"Found {len(curr_found_targets)} targets in image '{image_file_name}'."
        )

        # Add the targets found in the current image to the master list
        found_targets.extend(curr_found_targets)

        # --- Conditional Drawing and Saving ---

        # Draw the found targets on the image and save it, if requested and targets were found
        if draw_found_targets_in_images and curr_found_targets:
            # Draw targets using the engine's drawing function
            res_image = PGTEngine.draw_targets(curr_found_targets, image)

            # Define and create the output image directory
            res_image_folder_path = os.path.join(res_folder_path, "images")  #:SETTING:
            os.makedirs(res_image_folder_path, exist_ok=True)

            # Construct the full output path and save the image
            res_image_path = os.path.join(res_image_folder_path, image_file_name)
            log.info(
                f"The found targets are drawn and saved in image '{res_image_path}'."
            )
            ImageUtils.save_image(res_image_path, res_image)

    # --- Result Export (CSV) ---

    # Export CSVs only if at least one target was found
    if found_targets:
        log.info_separator_h2()
        res_csv_file_path = os.path.join(res_folder_path, res_csv_file_name)
        csv_root, csv_ext = os.path.splitext(res_csv_file_path)

        log.info(f"The found targets are exported in CSV '{res_csv_file_path}'.")

        # 1. Export standard CSV (always done if targets are found)
        PGTEngine.export_targets_to_csv(found_targets, res_csv_file_path)

        # 2. Export Meshroom CSV (Conditional)
        if export_found_targets_in_csv_meshroom:
            meshroom_path = f"{csv_root}_meshroom{csv_ext}"
            PGTEngine.export_targets_to_meshroom_csv(
                found_targets, meshroom_path
            )  #:SETTING:
            log.info(f"'{os.path.basename(meshroom_path)}' is also exported.")

        # 3. Export Metashape CSV (Conditional)
        if export_found_targets_in_csv_metashape:
            metashape_path = f"{csv_root}_metashape{csv_ext}"
            PGTEngine.export_targets_to_metashape_csv(
                found_targets, metashape_path
            )  #:SETTING:
            log.info(f"'{os.path.basename(metashape_path)}' is also exported.")

    # --- Finalization ---

    # Log the total number of found targets
    total_found_targets_count = len(found_targets)
    log.info_separator_h1()
    if total_found_targets_count > 0:
        log.info(
            f"Found {total_found_targets_count} targets in {total_image_count} image(s)."
        )
    else:
        log.info("Detection complete. No targets found in images.")
    log.info_separator_h1()
    log.info_end()

    # Save a backup of the current settings in the result folder
    settings_backup_file_path = os.path.join(
        res_folder_path, "settings_backup.yaml"
    )  #:SETTING: file name
    settings.save(settings_backup_file_path)

    # Return the total count of found targets
    return total_found_targets_count


# --------------------------------------------
# Core Function: Generate Printable Targets Patterns in (PNG, JPG, etc.) files
# --------------------------------------------


def run_generate_targets(
    target_type: str,
    res_folder_path: str,
    res_image_type: str,
    res_image_size: int,
    res_max_images_to_generate: int,
    progress_callback: Optional[Any] = None,
) -> int:
    """
    Generates a specified number of target images, saves them to a structured
    output folder, and logs the process and results.

    Args:
        target_type (str): The type of target to generate (e.g., 'A1', 'ring').
        res_folder_path (str): The root path where the results folder will be created.
        res_image_type (str): The file extension/format for the generated images (e.g., 'svg', 'pdf').
        res_image_size (int): The size (e.g., pixel dimension or scale) of the generated images.
        res_max_images_to_generate (int): The maximum number of images to generate.
        progress_callback (Optional[Any]): A callback object for reporting generation progress.

    Returns:
        int: The total number of targets successfully generated.
    """
    # Reload settings before starting the run to ensure all configuration is up-to-date
    PGTEngine.reload_settings()

    # Start logging for the entire process
    log.info_start(f"Generate Targets")

    # Define the specific output folder path based on target type and image type
    res_folder_path = os.path.join(
        res_folder_path, f"{target_type}_{res_image_type}_images"
    )  #:SETTING: The output folder name is constructed from parameters.

    # Ensure that the results folder exists (creates directories if they don't)
    os.makedirs(res_folder_path, exist_ok=True)

    # Call the core engine function to generate the targets
    numb_generated_targets = PGTEngine.generate_targets(
        target_type,
        res_folder_path,
        res_image_type,
        res_image_size,
        res_max_images_to_generate,
        progress_callback,
    )

    # --- Finalization and Logging ---

    # Log the number of generated targets
    log.info_separator_h1()
    if numb_generated_targets > 0:
        log.info(
            f"Successfully generated {numb_generated_targets} targets. Output saved to: {res_folder_path}"
        )
    else:
        log.info("Generation complete. No targets were generated.")
    log.info_separator_h1()
    log.info_end()

    # Save a backup of the current settings in the result folder
    settings_backup_file_path = os.path.join(
        res_folder_path, "settings_backup.yaml"
    )  #:SETTING: file name for settings backup
    settings.save(settings_backup_file_path)

    return numb_generated_targets


# --------------------------------------------
# Main execution block, if this script is executed directly
# Note. Use only for for development-time testing
# --------------------------------------------
if __name__ == "__main__":
    # Import tests.dev module
    from tests.dev import run_dev_test_find_cct_targets, run_dev_test_find_hrct_targets
    from tests.dev import (
        run_dev_test_generate_cct_targets,
        run_dev_test_generate_hrct_targets,
    )

    # Run dev tests (find)
    run_dev_test_find_cct_targets()
    run_dev_test_find_hrct_targets()

    # Run dev test (generate)
    # run_dev_test_generate_cct_targets()
    # run_dev_test_generate_hrct_targets()
