import pytest
import numpy as np
import simple_face
from simple_face import FaceAI

def test_import():
    """Ensure the package and core class can be imported properly."""
    assert simple_face.__version__ == "0.1.0"
    assert FaceAI is not None

def test_initialization():
    """Test that FaceAI can be instantiated."""
    ai = FaceAI()
    assert ai is not None

def test_recognize_frame_accepts_a_universal_rgb_frame():
    """A UI/camera adapter can pass an RGB NumPy frame without a window."""
    ai = FaceAI()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assert ai.recognize_frame(frame, color_format="rgb") == []
