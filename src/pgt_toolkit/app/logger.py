# Import external modules
from utils.logger import LogManager
from app.settings import settings


# Create and get a single global application logger
def get_app_logger():
    log_name = settings.get("logging.log_name", "app_logger")
    log_file = settings.get("logging.log_file", "app.log")
    log_level = settings.get("logging.level", "INFO")
    return LogManager.get_logger(log_name, log_file, log_level)


# Create a single global logger instance for reusing throughout the application
logger = get_app_logger()
