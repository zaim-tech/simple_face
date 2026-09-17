import logging
import cv2
import numpy as np
from typing import Tuple

def setup_logger(name: str = __name__) -> logging.Logger:
    """Sets up a basic logger for the module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = setup_logger(__name__)

def draw_face_box(frame: np.ndarray, face: np.ndarray, name: str, score: float = None, is_known: bool = False) -> None:
    """
    Helper method to draw a bounding box and label around a single face.
    """
    try:
        x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
    except (IndexError, ValueError) as e:
        logger.debug(f"Invalid face bounding box: {face}. Error: {e}")
        return

    color = (0, 255, 0) if is_known else (0, 0, 255)
    
    display_text = name
    if is_known and score is not None:
        display_text += f" ({score:.2f})"
        
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    # Ensure text doesn't go off top of screen
    text_y = max(y - 10, 20)
    cv2.putText(frame, display_text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
