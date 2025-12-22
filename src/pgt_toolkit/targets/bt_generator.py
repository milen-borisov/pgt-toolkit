"""
Base abstract class for all target generator subclasses.

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
# Base abstract class of all target generator subclasses
# --------------------------------------------
class BTGenerator(ABC):
    @staticmethod
    @abstractmethod
    def generate_targets(
        target_type: str,
        res_folder_path: str,
        res_image_type: str,
        res_image_size: int,
        res_max_images_to_generate: int,
        progress_callback=None,
    ) -> int:
        """
        Abstract method that must be implemented by all subclasses.
        This is the entry point for the main logic for generating
        targets in printable images.
        """
        pass

    @staticmethod
    @abstractmethod
    def _load_settings():
        pass
