# Import external modules
import os
from utils.settings import Settings


# Create a single global settings instance for reusing throughout the application
SETTINGS_FILE_NAME = "settings.yaml"
settings = Settings(SETTINGS_FILE_NAME)
