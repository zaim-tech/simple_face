import cv2
import platform
import logging
from typing import Optional, Tuple
from .utils import setup_logger

logger = setup_logger(__name__)

class Camera:
    """Handles webcam initialization and frame reading."""
    
    def __init__(self, camera_id: int = 0, backend: str = "auto", width: int = 640, height: int = 480):
        """
        Initializes the camera.
        
        Args:
            camera_id (int): Camera device ID.
            backend (str): Backend selection ("auto", "dshow", "msmf", "default").
            width (int): Frame width.
            height (int): Frame height.
        """
        self.camera_id = camera_id
        self.backend_choice = backend.lower()
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        """Starts the camera with the selected or fallback backend."""
        if self.cap is not None and self.cap.isOpened():
            return True

        os_name = platform.system()
        
        backends_to_try = []
        if self.backend_choice == "auto":
            if os_name == "Windows":
                backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            else:
                backends_to_try = [cv2.CAP_ANY]
        elif self.backend_choice == "dshow":
            backends_to_try = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        elif self.backend_choice == "msmf":
            backends_to_try = [cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends_to_try = [cv2.CAP_ANY]

        for backend in backends_to_try:
            logger.info(f"Trying to open camera {self.camera_id} with backend ID {backend}...")
            cap = cv2.VideoCapture(self.camera_id, backend)
            
            if cap is not None and cap.isOpened():
                # Optimize frame size and buffer
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Check if camera actually reads a frame (some backends open but fail to read)
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
                    logger.info("Camera initialized successfully.")
                    return True
                else:
                    logger.warning(f"Camera opened with backend {backend} but failed to read frames.")
                    cap.release()
            else:
                logger.warning(f"Failed to open camera with backend {backend}.")

        logger.error(f"Could not open webcam with ID {self.camera_id} using any backend.")
        return False

    def read_frame(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """Reads a frame from the camera."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def stop(self) -> None:
        """Releases the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released.")
