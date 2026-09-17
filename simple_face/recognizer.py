import cv2
import numpy as np
import os
from importlib import resources
from typing import Tuple, List, Optional
from .utils import setup_logger

logger = setup_logger(__name__)

class FaceRecognizerWrapper:
    """Wraps OpenCV's FaceDetectorYN and FaceRecognizerSF."""
    
    def __init__(self, threshold: float = 0.363):
        self.threshold = threshold
        self.detector = None
        self.recognizer = None
        self._load_models()

    def _load_models(self) -> None:
        """Loads the ONNX models using importlib.resources."""
        try:
            model_dir = resources.files("simple_face.models")
            det_model_path = str(model_dir / "face_detection_yunet_2023mar.onnx")
            rec_model_path = str(model_dir / "face_recognition_sface_2021dec.onnx")
            
            if not os.path.exists(det_model_path) or not os.path.exists(rec_model_path):
                raise FileNotFoundError("ONNX models not found in the package data.")
                
            self.detector = cv2.FaceDetectorYN.create(det_model_path, "", (320, 320))
            self.recognizer = cv2.FaceRecognizerSF.create(rec_model_path, "")
            logger.info("ONNX models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ONNX models: {e}")
            raise

    def detect_faces(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects faces in an image.
        
        Args:
            image (np.ndarray): The input image.
            
        Returns:
            Optional[np.ndarray]: An array of detected faces or None.
        """
        if image is None or self.detector is None:
            return None
            
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)
        return faces

    def extract_feature(self, image: np.ndarray, face: np.ndarray) -> Optional[np.ndarray]:
        """
        Aligns a face and extracts the 128D feature embedding.
        
        Args:
            image (np.ndarray): The full image.
            face (np.ndarray): The bounding box and landmarks for a single face.
            
        Returns:
            Optional[np.ndarray]: The feature embedding.
        """
        if self.recognizer is None:
            return None
            
        try:
            aligned = self.recognizer.alignCrop(image, face)
            return self.recognizer.feature(aligned)
        except Exception as e:
            logger.debug(f"Failed to extract feature: {e}")
            return None

    def match(self, live_feature: np.ndarray, known_features: dict) -> Tuple[str, float]:
        """
        Matches a feature against a database of known features.
        
        Args:
            live_feature (np.ndarray): The feature to match.
            known_features (dict): Dictionary mapping names to feature embeddings.
            
        Returns:
            Tuple[str, float]: The name of the best match and the confidence score.
        """
        if self.recognizer is None or live_feature is None:
            return "Unknown", 0.0

        best_match_name = "Unknown"
        highest_score = 0.0
        
        for name, known_feat in known_features.items():
            try:
                score = self.recognizer.match(known_feat, live_feature, cv2.FaceRecognizerSF_FR_COSINE)
                
                if score >= self.threshold and score > highest_score:
                    highest_score = score
                    best_match_name = name
            except Exception as e:
                logger.debug(f"Error matching face {name}: {e}")
                continue

        # If it matched an explicitly learned unknown, just call them Unknown
        if best_match_name.startswith("Unknown_"):
            best_match_name = "Unknown"
            
        return best_match_name, highest_score
