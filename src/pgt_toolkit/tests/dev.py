"""
This module contains simple test functions used for development-time testing.
"""

# Import internal modules
from app.logger import logger as log  # pyright: ignore[reportMissingImports]
from core import (
    run_find_targets,
    run_generate_targets,
)  # pyright: ignore[reportMissingImports]


# --------------------------------------------
# Test functions for development-time testing
# --------------------------------------------
def run_dev_test_find_cct_targets():
    try:
        run_find_targets(
            target_type="CCT_12bit_black",
            input_images_folder="data/dev_tests/test_find_cct/",
            res_folder_path="results/dev_tests/test_find_cct/",
            res_csv_file_name="found_targets.csv",
            threshold_modes=["adaptive"], #["adaptive", "otsu", "max", "triangle", "mean"],
            threshold_values=None,
            draw_found_targets_in_images=True,
            export_found_targets_in_csv_meshroom=True,
            export_found_targets_in_csv_metashape=True,
        )
    except Exception as e:
        log.exception(f"{e}")


def run_dev_test_find_hrct_targets():
    try:
        run_find_targets(
            target_type="HRCT_9dots_black",
            input_images_folder="data/dev_tests/test_find_hrct/",
            res_folder_path="results/dev_tests/test_find_hrct/",
            res_csv_file_name="found_targets.csv",
            threshold_modes=["adaptive", "otsu", "max", "triangle", "mean"], #["adaptive", "otsu", "max", "triangle", "mean"],
            threshold_values=None, #[215, 245],
            draw_found_targets_in_images=True,
            export_found_targets_in_csv_meshroom=True,
            export_found_targets_in_csv_metashape=True,
        )
    except Exception as e:
        log.exception(f"{e}")


def run_dev_test_generate_cct_targets():
    try:
        run_generate_targets(
            target_type="CCT_8bit_black",
            res_folder_path="results/dev_tests/test_generate_cct/",
            res_image_type="svg",
            res_image_size=1000,
            res_max_images_to_generate=33,
        )
    except Exception as e:
        log.exception(f"{e}")


def run_dev_test_generate_hrct_targets():
    try:
        run_generate_targets(
            target_type="HRCT_9dot_black",
            res_folder_path="results/dev_tests/test_generate_hrct/",
            res_image_type="svg",
            res_image_size=2000,
            res_max_images_to_generate=33,
        )
    except Exception as e:
        log.exception(f"{e}")
