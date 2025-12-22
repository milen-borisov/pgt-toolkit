# Import external modules
from typing import Union, Tuple

# Predefine types
ColorInput = Union[str, Tuple[int, int, int]]


# --------------------------------------------
# Color Utils Class
# --------------------------------------------
class ColorUtils:
    # --- HEX colors ---
    COLOR_BLACK_HEX = "#000000"
    COLOR_WHITE_HEX = "#FFFFFF"

    # --- BGR colors ---
    COLOR_BLACK = (0, 0, 0)
    COLOR_WHITE = (255, 255, 255)

    # --- Utils functions ---
    @staticmethod
    def standardize_color_input(color_input: ColorInput) -> Tuple[int, int, int]:
        """
        Converts color input (string or RGB/BGR tuple) into a standard BGR tuple (B, G, R).

        Args:
            color_input: The input color, e.g., 'red', 'g', or (0, 255, 0).

        Returns:
            A BGR tuple (0, 0, 0) to (255, 255, 255).

        Raises:
            ValueError: If the input is not a recognized color string or a valid tuple.
        """
        if isinstance(color_input, tuple):
            if len(color_input) == 3 and all(0 <= c <= 255 for c in color_input):
                # Assumes direct tuple input is already in BGR format for OpenCV
                return color_input
            else:
                raise ValueError("Tuple must be (B, G, R) with values 0-255.")

        if isinstance(color_input, str):
            color_map = {
                "red": (0, 0, 255),
                "r": (0, 0, 255),
                "green": (0, 255, 0),
                "g": (0, 255, 0),
                "blue": (255, 0, 0),
                "b": (255, 0, 0),
                "white": (255, 255, 255),
                "w": (255, 255, 255),
                "black": (0, 0, 0),
                "bk": (0, 0, 0),
            }

            normalized_input = color_input.lower().strip()

            if normalized_input in color_map:
                return color_map[normalized_input]

        raise ValueError(
            f"Invalid color input: {color_input}. Must be 'red', 'green', 'blue', 'r', 'g', 'b', ..., or a BGR tuple."
        )
