# Imports
import logging
from colorama import Fore, Style, init

# Init
init(autoreset=True)

# Define colors
LEVEL_COLORS = {
    'DEBUG': Fore.CYAN,
    'INFO': Fore.GREEN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'CRITICAL': Fore.MAGENTA,
}

# Logging levels
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

# Formatter
class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelName = record.levelname
        color = LEVEL_COLORS.get(levelName, '')

        message = super().format(record)

        return f'{color}{message}{Style.RESET_ALL}'

# Logger definition
def get_logger(name, level = DEBUG):
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger
    
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = ColorFormatter('[%(levelname)s] %(asctime)s - %(name)s.%(funcName)s - %(message)s', '%H:%M:%S.%m')

    handler.setFormatter(formatter)

    logger.addHandler(handler)
    
    return logger