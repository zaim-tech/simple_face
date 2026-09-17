import pytest
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
