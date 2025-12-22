"""
Defines utilities for handling and managing target type strings.

This module is part of the PGT-Toolkit project.
"""

# Import external modules
import numpy as np
from typing import Tuple


# --------------------------------------------
# Target Type Utilities Class
# --------------------------------------------
class TargetTypeUtils:
    # --- Target type checking methods ---
    @staticmethod
    def is_cct_target(target_type: str) -> bool:
        if target_type.startswith("CCT"):
            return True

    @staticmethod
    def is_hrct_target(target_type: str) -> bool:
        if target_type.startswith("HRCT"):
            return True

    # --- Target type building methods ---
    @staticmethod
    def build_cct_target_type(
        code_bit_number: int, cct_target_center_color: str
    ) -> str:
        return f"CCT_{code_bit_number}bit_{cct_target_center_color}"

    @staticmethod
    def build_hrct_target_type(code_dots_number: int, dots_color: str) -> str:
        return f"HRCT_{code_dots_number}dots_{dots_color}"

    # --- Target type getting metadata ---
    @staticmethod
    def get_metadata(target_type: str) -> Tuple[str, int, str]:
        return TargetTypeUtils._parse_target_type(target_type)

    @staticmethod
    def _parse_target_type(target_type: str) -> Tuple[str, int, str]:
        """
        Extracts (Name, Number, Color) from a target type string.
        Example: "CCT_8bit_black" -> ('CCT', 4, 'black')
        Example: "HRCT_9dots_black" -> ('HRCT', 9, 'black')
        """
        # Split the target type string by the underscore
        parts = target_type.split("_")

        if len(parts) < 3:
            raise ValueError(
                f"Incorrect target type format: '{target_type}'. Expected at least 3 parts."
            )

        # Extract the main components
        tt_name_str = parts[0]
        tt_number_str = parts[1]
        tt_color_str = parts[-1]  # Safely takes the last element as color

        # Check for validity
        if tt_name_str not in {"CCT", "HRCT"}:
            raise ValueError(
                f"Invalid type name: {tt_name_str} in {target_type}. Must be 'CCT', 'HRCT', etc."
            )

        if tt_color_str not in {"white", "black"}:
            raise ValueError(
                f"Invalid color: {tt_color_str} in {target_type}. Must be 'white' or 'black'."
            )

        # Extract the number from the feature part (e.g., "8bit" -> 8, "4dots" -> 4)
        number_str = ""
        for char in tt_number_str:
            if char.isdigit():  # Check if the character is a digit
                number_str += char
            else:  # Stop as soon as a non-digit character is encountered
                break

        if number_str:
            tt_number = int(number_str)
        else:
            raise ValueError(
                f"Invalid type number: {tt_number_str} in {target_type}. Must be '8bit', '9dots', etc."
            )

        # Check for validity
        if tt_name_str == "HRCT":
            if tt_number != 9:
                raise ValueError(
                    f"Invalid type number: {tt_number_str} in {target_type}. "
                    f"Only HRCT_9dots_black target is supported."
                )

            if tt_color_str != "black":
                raise ValueError(
                    f"Invalid type color: '{tt_color_str}' in {target_type}. "
                    f"Only HRCT_9dots_black target is supported."
                )

        if tt_name_str == "CCT":
            if tt_number < 4 or tt_number > 36:
                raise ValueError(
                    f"Invalid type number: {tt_number_str} in {target_type}. "
                    f"Bit number must be in the range [4, 36]."
                )

        # Return parsed information
        return (tt_name_str, tt_number, tt_color_str)
