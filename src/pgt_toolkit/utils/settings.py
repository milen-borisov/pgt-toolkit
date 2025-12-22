# Import external modules
import yaml
from typing import Any, Dict


# --------------------------------------------
# Settings Class
# --------------------------------------------
class Settings:
    _instance = None

    def reload(self):
        self._load(self.filename)

    def __new__(cls, filename: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(filename)
            cls._instance.filename = filename
        return cls._instance

    def _load(self, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

    def get(self, path: str, default: Any = None) -> Any:
        keys = path.split(".")
        val = self._data
        for key in keys:
            if isinstance(val, dict) and key in val:
                val = val[key]
            else:
                return default
        return val

    def get_all(self) -> Dict:
        return dict(self._data)

    def save(self, settings_file_path: str):
        """
        Saves the current internal settings dictionary (self._data) to YAML file.
        """
        try:
            with open(settings_file_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self._data, f, sort_keys=False
                )  # Using safe_dump and disabling sort_keys preserves the file's original order
        except Exception as e:
            # Handle potential writing errors
            print(f"=== ERROR saving the settings to {self.filename}: {e}")
