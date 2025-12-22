# Import external modules
import time
import logging


# --------------------------------------------
# CustomLogger class
# --------------------------------------------
class CustomLogger:
    """A wrapper class for the standard logging.Logger object."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self._logger.exception(msg, *args, **kwargs)

    def info_start(self, process_name: str):
        self.process_name = process_name
        self.start_time = time.time()

        self._logger.info(f"=== START: {process_name} ===")

    def info_end(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self._logger.info(
            f"=== END: {self.process_name}; Elapsed time: {elapsed_time:.2f} s. ==="
        )

    def info_separator_h1(self):
        self._logger.info("------------------------------------------")

    def info_separator_h2(self):
        self._logger.info("..........................................")


# --------------------------------------------
# LogManger class
# --------------------------------------------
class LogManager:
    """Manages the creation and configuration of the custom logger."""

    _logger = None

    @classmethod
    def get_logger(cls, log_name: str, log_file: str, log_level: str) -> CustomLogger:
        if cls._logger is None:
            # Configure the standard logger
            logger = logging.getLogger(log_name)
            logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

            # File handler
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            # Create an instance of our custom logger and store it
            cls._logger = CustomLogger(logger)

        return cls._logger
