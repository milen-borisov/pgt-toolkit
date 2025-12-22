"""
Main Command-Line Interface (CLI) entry point for the PGT-Toolkit.

This script uses argparse subparsers to support multiple functionalities
(e.g., 'find', 'generate') of targets/markers used in photogrammetry..
"""

# Import external modules
import os
import sys
import argparse
from pathlib import Path

# Import internal modules
curr_file_parent = Path(__file__).resolve().parent
sys.path.insert(0, os.path.join(curr_file_parent, "pgt_toolkit"))
from pgt_toolkit.core import run_find_targets, run_generate_targets


# --------------------------------------------
# Command Handlers
# --------------------------------------------
def handle_find_command(args):
    """Handles the 'find' subcommand for detection and decoding."""

    # Threshold parsing logic
    threshold_values = None
    if args.threshold_values:
        # Expect 1 or 2 integer values (e.g., ['10'] or ['10', '245'])
        try:
            threshold_values = [int(t) for t in args.threshold_values]
        except ValueError:
            print(f"\n--- ARGUMENT ERROR ---")
            print("Threshold values must be integers.")
            sys.exit(1)

        # Optional constraint check
        if len(threshold_values) > 2:
            print(f"\n--- ARGUMENT ERROR ---")
            print("Only 1 or 2 threshold values are supported.")
            sys.exit(1)

    # Run find targets
    try:
        run_find_targets(
            target_type=args.target_type,
            input_images_folder=args.input_images_folder,
            res_folder_path=args.res_folder_path,
            res_csv_file_name=args.res_csv_file_name,
            threshold_modes=args.threshold_modes,
            threshold_values=threshold_values,
            draw_found_targets_in_images=not args.no_draw,
            export_found_targets_in_csv_meshroom=not args.no_meshroom,
            export_found_targets_in_csv_metashape=not args.no_metashape,
        )
    except Exception as e:
        print(f"\n--- FATAL RUNTIME ERROR ---")
        print(f"An error occurred during targets detection: {e}")
        sys.exit(1)


def handle_generate_command(args):
    """Handles the 'generate' subcommand for marker creation."""
    try:
        # Check the input arguments
        res_max_images_to_generate_arg = args.res_max_images_to_generate
        if res_max_images_to_generate_arg.lower() in ("all", "-1"):
            res_max_images_to_generate = -1
        elif res_max_images_to_generate_arg.isdigit():
            res_max_images_to_generate = int(res_max_images_to_generate_arg)
        else:
            raise ValueError(
                f"Invalid value for --res-max-images: {res_max_images_to_generate_arg}. Must be a non-negative integer or 'all'."
            )

        # Note: The flags are handled by inverting the --no-X arguments
        run_generate_targets(
            target_type=args.target_type,
            res_folder_path=args.res_folder_path,
            res_image_type=args.res_image_type,
            res_image_size=args.res_image_size,
            res_max_images_to_generate=res_max_images_to_generate,
        )
    except Exception as e:
        print(f"\n--- FATAL RUNTIME ERROR ---")
        print(f"An error occurred during target generation: {e}")
        sys.exit(1)


# --------------------------------------------
# Main CLI Setup
# --------------------------------------------
def main():
    """Sets up the argparse subparsers for multi-command CLI."""
    parser = argparse.ArgumentParser(
        description="PGT-Toolkit: Find (Detect) or Generate photogrammetry targets.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Create the subparsers container
    subparsers = parser.add_subparsers(
        title="Available Commands",
        description="Specify whether to find targets in images or generate new targets.",
        required=True,
        dest="command",
    )

    # --- FIND Subparser ---
    find_parser = subparsers.add_parser(
        "find",
        help="Run detection and decoding on a folder of images.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    find_parser.add_argument(
        "--type",
        type=str,
        required=True,
        dest="target_type",
        help="The type of target to detect (e.g., CCT_12bit_black, HRCT_9dots_black, etc.).",
    )
    find_parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        dest="input_images_folder",
        help="Path to the folder containing the input images.",
    )

    # Output Control Arguments
    find_parser.add_argument(
        "--res-folder",
        type=str,
        default="results/latest_find",
        dest="res_folder_path",
        help="Path to the output folder for detection results (images and CSVs).",
    )
    find_parser.add_argument(
        "--csv-name",
        type=str,
        default="found_targets.csv",
        dest="res_csv_file_name",
        help="Base filename for the output detection CSV files.",
    )

    find_parser.add_argument(
        "--threshold-modes",
        nargs="+",  # Requires one or more arguments
        choices=["adaptive", "otsu", "triangle", "max", "mean"],
        default="adaptive",
        dest="threshold_modes",
        help=(
            "Specify which thresholding modes to enable. "
            "Choose one or more from: adaptive, otsu, triangle, max, mean"
        ),
    )
    find_parser.add_argument(
        "--threshold-values",
        type=str,
        nargs="*",  # Accepts 0 or more arguments
        dest="threshold_values",
        help=(
            "Optional: Add 1 or 2 integer thresholds (0-255) for enhanced "
            "dark and/or bright target detection.\n"
            "Usage: --threshold-values [DARK_TH] [BRIGHT_TH]\n"
            "Example: --threshold-values 10 245"
        ),
    )

    # Conditional Output Flags (using action='store_true' for negative flags)
    find_parser.add_argument(
        "--no-draw",
        action="store_true",
        dest="no_draw",
        help="Disable drawing detected targets on output images (default: enabled).",
    )
    find_parser.add_argument(
        "--no-meshroom",
        action="store_true",
        dest="no_meshroom",
        help="Disable exporting results to Meshroom-compatible CSV format (default: enabled).",
    )
    find_parser.add_argument(
        "--no-metashape",
        action="store_true",
        dest="no_metashape",
        help="Disable exporting results to Metashape-compatible CSV format (default: enabled).",
    )

    find_parser.set_defaults(func=handle_find_command)

    # --- GENERATE Subparser ---
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate target patterns.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    generate_parser.add_argument(
        "--type",
        type=str,
        required=True,
        dest="target_type",
        help="The type of target to generate (e.g., CCT_12bit_black, HRCT_9dots_black, etc.).",
    )
    # Output Control Arguments
    generate_parser.add_argument(
        "--res-folder",
        type=str,
        required=True,
        dest="res_folder_path",
        help="Path to the output folder for generated targets images.",
    )
    generate_parser.add_argument(
        "--res-image-type",
        type=str,
        default="png",
        dest="res_image_type",
        help="The file format for the generated result images. Options:  png, jpg,, bmp, tif, svg, pdf",
    )
    generate_parser.add_argument(
        "--res-image-size",
        type=int,
        default=480,
        dest="res_image_size",
        help="The size of the square side of the target image (in pixels).",
    )
    generate_parser.add_argument(
        "--res-max-images",
        type=str,
        default=-1,
        dest="res_max_images_to_generate",
        help="The maximum number of images (targets) to generate. Set to '-1' or 'all' to generate all possible unique targets.",
    )

    generate_parser.set_defaults(func=handle_generate_command)

    # --- Parse and Execute ---
    args = parser.parse_args()

    # args.func holds the handler function (handle_find_command or handle_generate_command)
    if "func" in args:
        args.func(args)
    else:
        # Should not happen if required=True for subparsers, but good practice
        parser.print_help()


# --------------------------------------------
# Main execution block
# --------------------------------------------
if __name__ == "__main__":
    main()

"""
# --- Example for calling 'find' command ---

python src/main_cli.py find
    --type "CCT_12bit_black" 
    --input-folder "data/cli_tests/test_find_cct/"
    --res-folder "results/cli_tests/test_find_cct"
    --csv-name "found_targets.csv"
    --threshold-mode adaptive otsu
    --threshold-values 10 245
    --no-draw
    --no-metashape
    --no-meshroom

python src/main_cli.py find
    --type "HRCT_9dots_black" 
    --input-folder "data/cli_tests/test_find_hrct/"
    --res-folder "results/cli_tests/test_find_hrct"
    --csv-name "found_targets.csv"
    --threshold-mode adaptive otsu
    --threshold-values 10 245
    --no-draw
    --no-metashape
    --no-meshroom


    
# --- Example for calling 'generate' command ---

python src/main_cli.py generate 
    --type "CCT_4bit_black" 
    --res-folder "results/cli_tests/test_generate_cct/" 
    --res-image-type "png" 
    --res-image-size 480 
    --res-max-images 5

python src/main_cli.py generate 
    --type "HRCT_9dots_black" 
    --res-folder "results/cli_tests/test_generate_hrct/" 
    --res-image-type "svg" 
    --res-image-size 480 
    --res-max-images 33
    
"""
