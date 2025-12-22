"""
Pytest file for verifying the CTT targets detected
in images into a given specific test folder.
"""


# --------------------------------------------
# Pytest Test Case
# --------------------------------------------
def test_found_hrct_targets_count():
    # Import the utility function from the source directory
    try:
        from core import run_find_targets  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        print(
            f"Import Error: {e}. Ensure 'target_toolkit' is the root directory when running pytest."
        )

    # --- Configuration for this specific test case ---
    # Parameters updated per user request
    target_type = "HRCT_9dots_black"
    input_images_folder = "tests/test_data/unit_tests/test_find_hrct"
    res_folder_path_actual = "tests/test_results_actual/unit_tests/test_find_hrct"
    res_csv_file_name = "found_targets.csv"

    # --- Expected Result ---
    res_folder_path_expected = "tests/test_results_expected/unit_tests/test_find_hrct"

    # --- Act: Run the detection utility ---
    found_targets = run_find_targets(
        target_type,
        input_images_folder,
        res_folder_path_actual,
        res_csv_file_name,
        threshold_modes=["adaptive", "otsu", "max"],
        threshold_values=None,
        draw_found_targets_in_images=True,
        export_found_targets_in_csv_meshroom=True,
        export_found_targets_in_csv_metashape=True,
    )

    # --- Act: Fetch the actual and expected data  ---
    import os

    res_csv_file_path_actual = os.path.join(res_folder_path_actual, res_csv_file_name)
    res_csv_file_path_expected = os.path.join(
        res_folder_path_expected, res_csv_file_name
    )

    with open(res_csv_file_path_expected, "r") as f_expected, open(
        res_csv_file_path_actual, "r"
    ) as f_actual:
        expected_content = f_expected.read()
        actual_content = f_actual.read()

    # Assert that the raw text content is identical
    assert expected_content == actual_content, (
        f"Test Failed: There are mismatches between the expected and found HRCT targets. "
        "Review the detection logic or the test dataset."
    )
