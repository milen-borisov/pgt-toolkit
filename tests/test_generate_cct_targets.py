"""
Pytest file for verifying the CTT targets generation in images into a given specific test folder.
"""


# --------------------------------------------
# Pytest Test Case
# --------------------------------------------
def test_found_cct_targets_count():
    # Import the utility function from the source directory
    try:
        from core import run_generate_targets  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        print(
            f"Import Error: {e}. Ensure 'target_toolkit' is the root directory when running pytest."
        )

    # --- Configuration for this specific test case ---
    # Parameters updated per user request
    target_type = "CCT_8bit_black"
    res_folder_path_actual = "tests/test_results_actual/unit_tests/test_generate_cct"

    # --- Expected Result ---
    expected_numb_generated_image = 3

    # --- Act: Run the detection utility (Raster images) ---
    actual_numb_generated_image = run_generate_targets(
        target_type,
        res_folder_path_actual,
        res_image_type="jpg",
        res_image_size=400,
        res_max_images_to_generate=expected_numb_generated_image,
    )

    # Assert that the raw text content is identical
    assert actual_numb_generated_image == expected_numb_generated_image, (
        f"Test Failed: There are mismatches between the expected and "
        "the actual generated CCT targets in JPG images. "
        "Review the detection logic or the test dataset."
    )

    # --- Act: Run the detection utility (Vector images) ---
    actual_numb_generated_image = run_generate_targets(
        target_type,
        res_folder_path_actual,
        res_image_type="svg",
        res_image_size=400,
        res_max_images_to_generate=expected_numb_generated_image,
    )

    # Assert that the raw text content is identical
    assert actual_numb_generated_image == expected_numb_generated_image, (
        f"Test Failed: There are mismatches between the expected and "
        "the actual generated CCT targets in SVG images."
        "Review the detection logic or the test dataset."
    )
