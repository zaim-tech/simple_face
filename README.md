# Simple Face

**Simple Face** is a lightweight Python face-recognition library for building desktop applications that identify people from images, video files, or a live webcam. It bundles OpenCV ONNX face-detection and face-recognition models, so there is no separate model-download step.

It is a practical starting point for classroom attendance, visitor check-in, photo review, and small computer-vision projects. Use face recognition responsibly: obtain consent, protect stored face data, and do not use it as the only basis for high-impact decisions.

## Features

- Detect and recognise faces in still images, videos, and live camera feeds.
- Register a person from one clear reference photo.
- Return machine-readable image results, including a name, similarity score, and bounding box.
- Save and reload registered people as a JSON database.
- Include bundled reference faces to help reduce incorrect matches.
- Choose automatic, DirectShow, or Media Foundation camera backends on Windows.

## Supported platforms and devices

| Platform | Status | Camera support |
| --- | --- | --- |
| Windows | Supported | Built-in and USB webcams; automatic backend selection, DirectShow, and Media Foundation are available. |
| macOS | Supported through Python and OpenCV | Built-in and USB webcams exposed to OpenCV. |
| Linux | Supported through Python and OpenCV | Built-in, USB, and other cameras exposed to OpenCV. |
| Android / iOS | Not supported directly | Use a desktop/server application or build a separate mobile integration. |

The library requires Python 3.7 or newer. Camera availability ultimately depends on your operating system permissions, OpenCV installation, and the device driver.

## Installation

Clone this repository and install it from its root directory:

```bash
git clone https://github.com/zaim-tech/simple_face.git
cd simple_face
pip install .
```

For local development, use an editable installation:

```bash
pip install -e .
```

The installation includes `opencv-contrib-python`, `numpy`, and `tqdm`.

## Quick start

Place a clear, front-facing photo of each person somewhere accessible to your script. Then register the person and start the webcam:

```python
from simple_face import FaceAI

ai = FaceAI()
ai.add_person("Zaim", "photos/zaim.jpg")
ai.add_person("Alice", "photos/alice.jpg")

# Press q in the video window to stop.
ai.start_webcam(camera_id=0)
```

`add_person()` returns `True` when a face was found and registered, otherwise `False`.

## Usage examples

### Recognise faces in one image

Use `return_results=True` when your app needs data instead of an OpenCV preview window.

```python
from simple_face import FaceAI

ai = FaceAI()
ai.add_person("Zaim", "photos/zaim.jpg")

results = ai.check_image("photos/group-photo.jpg", return_results=True)
for face in results or []:
    print(face["name"], face["score"], face["box"])
```

Each result has this shape:

```python
{"name": "Zaim", "score": 0.72, "box": [x, y, width, height]}
```

Without `return_results=True`, `check_image()` opens a labelled preview window. Press any key to close it.

### Scan a recorded video

```python
from simple_face import FaceAI

ai = FaceAI()
ai.add_person("Alice", "photos/alice.jpg")
ai.check_video("videos/event.mp4")
```

Press `q` to stop the video early.

### Save registered people and load them later

```python
from simple_face import FaceAI

# First run: create the database.
ai = FaceAI()
ai.add_person("Zaim", "photos/zaim.jpg")
ai.add_person("Alice", "photos/alice.jpg")
ai.save_db("faces.json")

# A later run: reuse it.
ai = FaceAI()
ai.load_db("faces.json")
ai.start_webcam()
```

The saved JSON contains face embeddings. Treat it as sensitive biometric data and keep it out of public repositories.

### Use a custom unknown-faces folder

The package contains bundled unknown-face references. You can supply your own folder of `.jpg`, `.jpeg`, or `.png` images instead:

```python
from simple_face import FaceAI

ai = FaceAI(
    threshold=0.363,
    unknowns_folder="training/unknown-faces",
)
```

### Choose a Windows camera backend

If a webcam has trouble opening on Windows, try a specific backend:

```python
from simple_face import FaceAI

ai = FaceAI(camera_backend="dshow")  # or "msmf" or "auto"
ai.start_webcam(camera_id=1)
```

## API overview

| Method | Purpose |
| --- | --- |
| `FaceAI(threshold=0.363, unknowns_folder=None, camera_backend="auto")` | Creates the recogniser. |
| `add_person(name, image_path)` | Adds one person from a reference image. |
| `check_image(image_path, return_results=False)` | Recognises all faces in an image. |
| `start_webcam(camera_id=0)` | Starts real-time webcam recognition. |
| `check_video(video_path)` | Recognises faces frame by frame in a video. |
| `save_db(path="faces_db.json")` | Saves known-person embeddings. |
| `load_db(path="faces_db.json")` | Loads known-person embeddings. |
| `clear_db(clear_unknowns=False)` | Clears registered people; optionally also clears unknown references. |

## Tips for better recognition

- Use sharp, well-lit, front-facing reference photos with one visible face.
- Register more than one good photo per person only if you manage the entries deliberately; registering again with the same name replaces its previous embedding.
- Adjust `threshold` carefully: higher values make matches stricter; lower values accept more possible matches.
- Make sure your operating system has granted Python or your terminal camera permission.

## Project example

See [examples/main.py](examples/main.py) for a customised webcam attendance-style example using `examples/zaim.png`.

## License

MIT License.

---

Made with ❤️ by Zaim Sheali
