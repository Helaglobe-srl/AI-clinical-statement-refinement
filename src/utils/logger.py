import os
import logging
import datetime
from typing import Optional

class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        log_filename = os.path.join(
            self.log_dir, 
            f"statement_refiner_hybrid_rag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        return logging.getLogger(__name__)

    def info(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def error(self, message: str):
        """Log an error message."""
        self.logger.error(message)

    def warning(self, message: str):
        """Log a warning message."""
        self.logger.warning(message)

    def debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message) 