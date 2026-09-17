from __future__ import annotations

import logging
import sys
from backend.config import LOG_LEVEL

def configure_logging() -> None:
    """Configures systemic logging for Neuromorphic-Ops."""
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    # Simple, clear standard logging formatting
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(linsteno)d if 'linsteno' in locals() else %(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    # Avoid duplicate handlers if re-called
    if not root_logger.handlers:
        root_logger.setLevel(level)
        root_logger.addHandler(handler)
        
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
