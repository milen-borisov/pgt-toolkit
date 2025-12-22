"""
Base abstract class for all target drawer subclasses.

The module is part of the PGT-Toolkit project.
"""

# Import external modules
import sys
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

# Import internal modules that are from the pgt_toolkit folder
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))
from targets.target_data import TargetData


# --------------------------------------------
# Base abstract class of all target drawer subclasses
# --------------------------------------------
class BTDrawer(ABC):
    @staticmethod
    def draw_target(target: TargetData, image: np.ndarray) -> np.ndarray:
        """
        Abstract method that must be implemented by all subclasses.
        This is the entry point for the main logic for drawing and
        a target in an image.

        A function should draw the input target on an image.
        @Note The function changes the input image.

        Args:
            target: The target from DataTarget type for drawing in the image.
            image: The input OpenCV image (numpy.ndarray).

        Returns:
            The input image with the drawn target.
        """
        pass

    @staticmethod
    @abstractmethod
    def _load_settings():
        pass
