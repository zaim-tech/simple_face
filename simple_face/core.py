import cv2
import numpy as np
import os
import glob
import json
from importlib import resources
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

from .utils import setup_logger, draw_face_box
from .camera import Camera
from .recognizer import FaceRecognizerWrapper
from .cache import EmbeddingCache

logger = setup_logger(__name__)

class FaceAI:
    """
    A lightweight, kid-friendly Face Recognition AI class, ready for production use.
    """
    
    def __init__(self, threshold: float = 0.363, unknowns_folder: str = None, camera_backend: str = "auto") -> None:
        """
        Initializes the FaceAI models and optionally loads unknown faces.
        
        Args:
            threshold (float): The cosine similarity threshold for a match. 
                               Default is 0.363 for SFace.
            unknowns_folder (str, optional): A path to a custom folder of unknown faces.
            camera_backend (str): The camera backend to use ("auto", "dshow", "msmf").
        """
        logger.info("Initializing FaceAI...")
        self.recognizer_wrap = FaceRecognizerWrapper(threshold=threshold)
        self.cache = EmbeddingCache()
        self.camera_backend = camera_backend
        
        self.known_faces_db: Dict[str, np.ndarray] = {}
        
        # Load unknowns from cache if available, else process and cache
        cached_unknowns = self.cache.load_cache()
        if cached_unknowns:
            self.known_faces_db.update(cached_unknowns)
        else:
            if unknowns_folder is None:
                # Load from bundled unknowns package data
                bundled_unknowns = resources.files("simple_face.unknowns")
                self._load_bundled_unknowns(bundled_unknowns)
            else:
                if os.path.exists(unknowns_folder):
                    self.add_unknowns_from_folder(unknowns_folder)
                else:
                    logger.warning(f"Custom unknowns folder not found: {unknowns_folder}")
                    
            # Extract only unknown features to cache
            unknowns_to_cache = {k: v for k, v in self.known_faces_db.items() if k.startswith("Unknown_")}
            if unknowns_to_cache:
                self.cache.save_cache(unknowns_to_cache)

    def add_person(self, name: str, image_path: str) -> bool:
        """Adds a known person to the recognition database from an image file."""
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False
            
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image {image_path} for {name}")
            return False
            
        faces = self.recognizer_wrap.detect_faces(img)
        
        if faces is not None and len(faces) > 0:
            feat = self.recognizer_wrap.extract_feature(img, faces[0])
            if feat is not None:
                self.known_faces_db[name] = feat
                logger.info(f"Added {name} to the database!")
                return True
            else:
                logger.error(f"Failed to extract features for {name}")
                return False
        else:
            logger.warning(f"No face found in {image_path} for {name}")
            return False

    def _load_bundled_unknowns(self, bundled_path) -> None:
        """Helper to load unknowns from importlib.resources"""
        logger.info("Learning unknown faces from bundled package data...")
        count = 0
        try:
            items = list(bundled_path.iterdir())
            for item in tqdm(items, desc="Loading Unknowns"):
                if item.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(str(item))
                    if img is not None:
                        faces = self.recognizer_wrap.detect_faces(img)
                        if faces is not None:
                            for face in faces:
                                feat = self.recognizer_wrap.extract_feature(img, face)
                                if feat is not None:
                                    self.known_faces_db[f"Unknown_{count}"] = feat
                                    count += 1
            logger.info(f"Learned {count} unknown faces to avoid confusion!")
        except Exception as e:
            logger.warning(f"Could not load bundled unknowns: {e}")

    def add_unknowns_from_folder(self, folder_path: str) -> None:
        """Scans a local folder for faces to learn what an 'Unknown' person looks like."""
        if not os.path.exists(folder_path):
            logger.warning(f"Folder {folder_path} does not exist.")
            return

        logger.info(f"Learning unknown faces from {folder_path}...")
        image_files = glob.glob(os.path.join(folder_path, "*.*"))
        
        count = 0
        for file_path in tqdm(image_files, desc="Loading Unknowns"):
            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img = cv2.imread(file_path)
            if img is None:
                continue
                
            faces = self.recognizer_wrap.detect_faces(img)
            if faces is not None:
                for face in faces:
                    feat = self.recognizer_wrap.extract_feature(img, face)
                    if feat is not None:
                        self.known_faces_db[f"Unknown_custom_{count}"] = feat
                        count += 1
                    
        logger.info(f"Learned {count} unknown faces from folder to avoid confusion!")

    def check_image(self, image_path: str, return_results: bool = False) -> Optional[List[Dict]]:
        """Detects and recognizes faces in a single image."""
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None
            
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image {image_path}")
            return None
            
        faces = self.recognizer_wrap.detect_faces(img)
        results = []
        
        if faces is not None:
            for face in faces:
                name, score = self._process_and_draw_face(img, face)
                if return_results:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    results.append({"name": name, "score": score, "box": [x, y, w, h]})
                
        if not return_results:
            cv2.imshow("Face AI - Image Check", img)
            logger.info("Press any key on the image window to close it.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return results if return_results else None

    def start_webcam(self, camera_id: int = 0) -> None:
        """Opens a webcam feed and runs face recognition in real-time."""
        logger.info(f"Starting webcam (ID: {camera_id})... Press 'q' to quit.")
        
        camera = Camera(camera_id=camera_id, backend=self.camera_backend)
        if not camera.start():
            return
            
        try:
            self._run_inference_loop(camera, "Face AI - Webcam")
        finally:
            camera.stop()
            cv2.destroyAllWindows()

    def check_video(self, video_path: str) -> None:
        """Opens a video file and runs face recognition frame-by-frame."""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return
            
        logger.info(f"Checking video: {video_path}... Press 'q' to quit early.")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            return
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video feed ended.")
                    break
                
                faces = self.recognizer_wrap.detect_faces(frame)
                if faces is not None:
                    for face in faces:
                        self._process_and_draw_face(frame, face)
                        
                cv2.imshow("Face AI - Video Check", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Video playback stopped.")

    def _run_inference_loop(self, camera: Camera, window_name: str) -> None:
        """Helper to run the inference loop for webcams."""
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                logger.info("Camera feed interrupted.")
                break
                
            faces = self.recognizer_wrap.detect_faces(frame)
            
            if faces is not None:
                for face in faces:
                    self._process_and_draw_face(frame, face)
                    
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _process_and_draw_face(self, frame: np.ndarray, face: np.ndarray) -> Tuple[str, float]:
        """Recognizes a face and draws bounding boxes."""
        live_feature = self.recognizer_wrap.extract_feature(frame, face)
        if live_feature is None:
            return "Unknown", 0.0
            
        best_match_name, highest_score = self.recognizer_wrap.match(live_feature, self.known_faces_db)
        
        is_known = best_match_name != "Unknown"
        draw_face_box(frame, face, best_match_name, highest_score, is_known=is_known)
        
        return best_match_name, highest_score

    def save_db(self, path: str = "faces_db.json") -> bool:
        """Saves the known faces database to a JSON file."""
        try:
            # Filter out unknowns for the saved DB
            known_only = {k: v for k, v in self.known_faces_db.items() if not k.startswith("Unknown_")}
            serializable_db = {k: v.tolist() for k, v in known_only.items()}
            with open(path, 'w') as f:
                json.dump(serializable_db, f)
            logger.info(f"Database saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            return False
            
    def load_db(self, path: str = "faces_db.json") -> bool:
        """Loads a known faces database from a JSON file."""
        if not os.path.exists(path):
            logger.error(f"Database file not found: {path}")
            return False
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            # Convert back to numpy arrays
            loaded_db = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
            # Merge with existing (keeping unknowns)
            self.known_faces_db.update(loaded_db)
            logger.info(f"Database loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return False
            
    def clear_db(self, clear_unknowns: bool = False) -> None:
        """Clears the known faces database."""
        if clear_unknowns:
            self.known_faces_db.clear()
            self.cache.clear_cache()
            logger.info("Cleared entire database including unknowns cache.")
        else:
            self.known_faces_db = {k: v for k, v in self.known_faces_db.items() if k.startswith("Unknown_")}
            logger.info("Cleared known faces database (unknowns retained).")
