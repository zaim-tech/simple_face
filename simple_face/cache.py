import os
import numpy as np
from pathlib import Path
from typing import Dict
from .utils import setup_logger

logger = setup_logger(__name__)

def get_cache_dir() -> Path:
    """Returns the user-specific cache directory for simple_face."""
    home = Path.home()
    cache_dir = home / ".simple_face"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

class EmbeddingCache:
    """Handles saving and loading face embeddings to/from disk."""
    
    def __init__(self, cache_file_name: str = "unknown_embeddings.npz"):
        self.cache_path = get_cache_dir() / cache_file_name

    def load_cache(self) -> Dict[str, np.ndarray]:
        """Loads embeddings from the cache file if it exists."""
        if self.cache_path.exists():
            try:
                data = np.load(self.cache_path)
                logger.info(f"Loaded cached unknowns from {self.cache_path}")
                return {k: v for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_path}: {e}")
        return {}

    def save_cache(self, embeddings: Dict[str, np.ndarray]) -> bool:
        """Saves embeddings to the cache file."""
        if not embeddings:
            return False
        try:
            np.savez(self.cache_path, **embeddings)
            logger.info(f"Saved unknowns cache to {self.cache_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_path}: {e}")
            return False

    def clear_cache(self) -> bool:
        """Deletes the cache file."""
        if self.cache_path.exists():
            try:
                self.cache_path.unlink()
                logger.info(f"Cleared cache at {self.cache_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to clear cache at {self.cache_path}: {e}")
                return False
        return True
