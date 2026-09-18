"""Framework-neutral face-recognition API."""

import json
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .camera import Camera
from .recognizer import FaceRecognizerWrapper
from .utils import draw_face_box, setup_logger

logger = setup_logger(__name__)


class FaceAI:
    """Offline face recognition with a universal frame-in/result-out API."""

    def __init__(self, threshold: float = 0.363, camera_backend: str = "auto") -> None:
        """Create the engine; camera_backend is only for start_webcam()."""
        self.recognizer_wrap = FaceRecognizerWrapper(threshold=threshold)
        self.camera_backend = camera_backend
        self.known_faces_db: Dict[str, np.ndarray] = {}

    def add_person(self, name: str, image_path: str) -> bool:
        """Enroll a person from an image file (desktop convenience method)."""
        if not os.path.exists(image_path):
            logger.error("Image file not found: %s", image_path)
            return False
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Could not read image %s", image_path)
            return False
        return self.enroll_frame(name, image)

    def enroll_frame(self, name: str, frame: np.ndarray, color_format: str = "bgr") -> bool:
        """Enroll the largest face in a NumPy frame from any camera framework.

        ``color_format`` may be ``bgr`` (default), ``rgb``, or ``rgba``.
        """
        image = self._as_bgr(frame, color_format)
        if image is None:
            return False
        faces = self.recognizer_wrap.detect_faces(image)
        if faces is None or len(faces) == 0:
            logger.warning("No face found while enrolling %s", name)
            return False
        face = max(faces, key=lambda item: float(item[2]) * float(item[3]))
        feature = self.recognizer_wrap.extract_feature(image, face)
        if feature is None:
            logger.error("Could not create an embedding for %s", name)
            return False
        self.known_faces_db[name] = feature
        return True

    def recognize_frame(self, frame: np.ndarray, color_format: str = "bgr") -> List[Dict[str, Any]]:
        """Return faces in a frame without opening a window or owning a camera.

        This is the portable API for OpenCV, Flet, Kivy, Android and iOS. It
        returns dictionaries like ``{"name": "Alice", "score": 0.71,
        "box": [x, y, width, height]}``. The host app displays the preview and
        draws the result overlays.
        """
        image = self._as_bgr(frame, color_format)
        if image is None:
            return []
        faces = self.recognizer_wrap.detect_faces(image)
        if faces is None:
            return []

        results: List[Dict[str, Any]] = []
        for face in faces:
            feature = self.recognizer_wrap.extract_feature(image, face)
            if feature is None:
                continue
            name, score = self.recognizer_wrap.match(feature, self.known_faces_db)
            x, y, width, height = (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
            results.append({"name": name, "score": float(score), "box": [x, y, width, height]})
        return results

    @staticmethod
    def draw_results(frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Optional OpenCV helper to draw universal results on a BGR frame."""
        for result in results:
            x, y, width, height = result["box"]
            face = np.array([x, y, width, height])
            draw_face_box(frame, face, result["name"], result["score"], result["name"] != "Unknown")
        return frame

    def check_image(self, image_path: str, return_results: bool = False) -> Optional[List[Dict[str, Any]]]:
        """Recognize a still image; preview it only when requested for desktop use."""
        image = cv2.imread(image_path)
        if image is None:
            logger.error("Could not read image %s", image_path)
            return None
        results = self.recognize_frame(image)
        if return_results:
            return results
        cv2.imshow("Face AI - Image Check", self.draw_results(image, results))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None

    def start_webcam(self, camera_id: int = 0) -> None:
        """Optional desktop OpenCV demo; mobile apps use recognize_frame()."""
        camera = Camera(camera_id=camera_id, backend=self.camera_backend)
        if not camera.start():
            return
        try:
            while True:
                success, frame = camera.read_frame()
                if not success or frame is None:
                    break
                cv2.imshow("Face AI - Webcam", self.draw_results(frame, self.recognize_frame(frame)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.stop()
            cv2.destroyAllWindows()

    def save_db(self, path: str = "faces_db.json") -> bool:
        """Save enrolled embeddings; use private app storage on mobile."""
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump({name: feature.tolist() for name, feature in self.known_faces_db.items()}, file)
            return True
        except (OSError, TypeError) as error:
            logger.error("Failed to save database: %s", error)
            return False

    def load_db(self, path: str = "faces_db.json") -> bool:
        """Load enrolled embeddings from a JSON database."""
        try:
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
            self.known_faces_db.update({name: np.array(feature, dtype=np.float32) for name, feature in data.items()})
            return True
        except (OSError, ValueError, TypeError) as error:
            logger.error("Failed to load database: %s", error)
            return False

    def clear_db(self) -> None:
        """Remove all enrolled people from memory."""
        self.known_faces_db.clear()

    @staticmethod
    def _as_bgr(frame: np.ndarray, color_format: str) -> Optional[np.ndarray]:
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            logger.warning("Frame must be a NumPy image array.")
            return None
        image_format = color_format.lower()
        if image_format == "bgr":
            return frame
        if image_format == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if image_format == "rgba":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        logger.warning("Unsupported color_format '%s'; use bgr, rgb, or rgba.", color_format)
        return None
