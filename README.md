# Simple Face

**Simple Face** is a lightweight Python face-recognition library for building desktop applications that identify people from images, video files, or a live webcam. It bundles OpenCV ONNX face-detection and face-recognition models, so there is no separate model-download step.

It is a practical starting point for classroom attendance, visitor check-in, photo review, and small computer-vision projects. Use face recognition responsibly: obtain consent, protect stored face data, and do not use it as the only basis for high-impact decisions.

## Features

- Detect and recognise faces in still images, videos, and live camera feeds.
- Register a person from one clear reference photo.
- Return machine-readable image results, including a name, similarity score, and bounding box.
- Save and reload registered people as a JSON database.
- Choose automatic, DirectShow, or Media Foundation camera backends on Windows.

## Supported platforms and devices

| Platform | Status | Camera support |
| --- | --- | --- |
| Windows | Supported | Built-in and USB webcams; automatic backend selection, DirectShow, and Media Foundation are available. |
| macOS | Supported through Python and OpenCV | Built-in and USB webcams exposed to OpenCV. |
| Linux | Supported through Python and OpenCV | Built-in, USB, and other cameras exposed to OpenCV. |
| Android / iOS | Supported through the included Flet example | Use `flet-camera` for the camera preview and OpenCV for recognition. |

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

The installation includes `opencv-contrib-python` and `numpy`.

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

### Choose a Windows camera backend

If a webcam has trouble opening on Windows, try a specific backend:

```python
from simple_face import FaceAI

ai = FaceAI(camera_backend="dshow")  # or "msmf" or "auto"
ai.start_webcam(camera_id=1)
```

### Build a custom webcam experience

For attendance screens, welcome messages, or your own interface, read webcam frames yourself and use Simple Face to detect and label each face. This example records each recognised person once and displays a custom message. Press `q` to close the window.

```python
import cv2
from simple_face import FaceAI

ai = FaceAI(camera_backend="dshow")
ai.add_person("Zaim", "photos/zaim.jpg")

attendance = set()
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

try:
    while True:
        success, frame = camera.read()
        if not success:
            break

        faces = ai.recognize_frame(frame)
        ai.draw_results(frame, faces)
        for face in faces:
            if face["name"] != "Unknown" and face["name"] not in attendance:
                attendance.add(face["name"])
                print(f"{face['name']} marked present (score: {face['score']:.2f})")

        cv2.imshow("Simple Face Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
```

## Flet Camera example

The Flet example, [examples/app.py](examples/app.py), adds `examples/zaim.png` as **Zaim**, opens a live camera preview, and displays a second preview with OpenCV-drawn `Zaim` or `Unknown` boxes and labels.

Run it from the repository root:

```bash
pip install -e .
pip install "flet>=0.81.0" "flet-camera>=0.81.0"
flet run examples/app.py
```

It uses this pipeline:

```text
Flet Camera frame bytes -> cv2.imdecode() -> FaceAI.recognize_frame()
-> FaceAI.draw_results() -> JPEG bytes displayed by ft.Image
```

On Android/iOS devices where `flet-camera` supports image streaming, it uses the live `on_stream_image` feed. Some browser/camera combinations show a camera preview but do not provide image streaming; the example automatically falls back to repeated `take_picture()` captures. The fallback is slower, but still performs recognition.

Build Android from the example directory:

```bash
cd examples
flet build apk
```

[examples/pyproject.toml](examples/pyproject.toml) includes the camera permission and `extract_packages = ["cv2"]`, required for OpenCV on Android. Build iOS from macOS with `flet build ipa`.

## API overview

| Method | Purpose |
| --- | --- |
| `FaceAI(threshold=0.363, camera_backend="auto")` | Creates the recogniser. |
| `add_person(name, image_path)` | Adds one person from a reference image. |
| `enroll_frame(name, frame, color_format="bgr")` | Adds one person from a NumPy frame. |
| `recognize_frame(frame, color_format="bgr")` | Recognises all faces in a frame and returns result dictionaries. |
| `check_image(image_path, return_results=False)` | Recognises all faces in an image. |
| `start_webcam(camera_id=0)` | Starts real-time desktop webcam recognition. |
| `save_db(path="faces_db.json")` | Saves known-person embeddings. |
| `load_db(path="faces_db.json")` | Loads known-person embeddings. |
| `clear_db()` | Clears registered people. |

## Tips for better recognition

- Use sharp, well-lit, front-facing reference photos with one visible face.
- Register more than one good photo per person only if you manage the entries deliberately; registering again with the same name replaces its previous embedding.
- Adjust `threshold` carefully: higher values make matches stricter; lower values accept more possible matches.
- Make sure your operating system has granted Python or your terminal camera permission.

## Project examples

See [examples/main.py](examples/main.py) for a customised OpenCV webcam attendance-style example using `examples/zaim.png`, or [examples/app.py](examples/app.py) for the Flet Camera version.

## License

MIT License.

---

Made with ❤️ by Zaim Sheali
