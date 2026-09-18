# Simple Face

**Simple Face** is an offline face-recognition engine for Python applications.
It uses OpenCV's YuNet detector and SFace embedding model, but it does not own
your camera or user interface. That keeps one recognition API usable with an
OpenCV desktop app, a Flet app, Kivy, or a native Android/iOS camera bridge.

The package includes only the two ONNX models; it does **not** ship a dataset of
strangers or create an unknown-face cache. A face is `Unknown` when it does not
meet the configured similarity threshold.

## Install

```bash
pip install .
```

The package uses `opencv-contrib-python`. Flet provides compatible mobile
OpenCV packages for Android and iOS; the included Flet project configures this
correctly for its mobile builds.

## Universal integration

Your framework opens the camera, converts its image into a NumPy array, and
calls `recognize_frame()`. The library returns plain dictionaries, making it
easy to draw labels using any UI toolkit.

```python
from simple_face import FaceAI

ai = FaceAI(threshold=0.363)

# Enrol from any frame.  Flet/Kivy/mobile camera plugins often provide RGB.
ai.enroll_frame("Alice", camera_rgb_frame, color_format="rgb")

# Call this for selected live-stream frames (for example, 2–5 times per second).
faces = ai.recognize_frame(camera_rgb_frame, color_format="rgb")
# [{"name": "Alice", "score": 0.71, "box": [x, y, width, height]}]
```

`color_format` accepts `"bgr"` (the default for OpenCV), `"rgb"`, and
`"rgba"`. The host application should render the live preview and draw each
result box itself. This avoids desktop-only OpenCV windows and camera APIs.

## Desktop OpenCV example

```python
import cv2
from simple_face import FaceAI

ai = FaceAI()
ai.add_person("Alice", "photos/alice.jpg")

camera = cv2.VideoCapture(0)
try:
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        results = ai.recognize_frame(frame)
        cv2.imshow("Recognition", ai.draw_results(frame, results))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
```

## Flet Camera example

The ready-to-run Flet example is [examples/app.py](examples/app.py). It enrols
`examples/zaim.png` as **Zaim**, opens a Flet Camera preview, and renders a
second live preview containing OpenCV's `Zaim` or `Unknown` labels and boxes.

Install the example dependencies, then run it from the repository root:

```bash
pip install -e .
pip install "flet>=0.81.0" "flet-camera>=0.81.0"
flet run examples/app.py
```

The recognition path is:

```text
Flet Camera frame bytes
→ cv2.imdecode()
→ FaceAI.recognize_frame()
→ FaceAI.draw_results()
→ JPEG bytes displayed by ft.Image
```

The essential frame handler is:

```python
def on_stream_image(event: fc.CameraImageEvent):
    frame = cv2.imdecode(np.frombuffer(event.bytes, np.uint8), cv2.IMREAD_COLOR)
    faces = ai.recognize_frame(frame)
    labelled = ai.draw_results(frame, faces)
    _, jpeg = cv2.imencode(".jpg", labelled)
    result_image.src = jpeg.tobytes()
    result_image.update()
```

Flet Camera supports Android, iOS, and web. On Android/iOS devices that support
image streaming, the example processes the live `on_stream_image` feed. Some
browser/camera combinations provide a preview but no image stream; the example
then falls back to repeated `take_picture()` captures. That fallback is slower
but still performs labelled recognition.

Build Android from the example directory:

```bash
cd examples
flet build apk
```

[examples/pyproject.toml](examples/pyproject.toml) declares camera permission
and `extract_packages = ["cv2"]`, required for OpenCV on Android. Build iOS
from macOS with `flet build ipa`.

## API

| Method | Purpose |
| --- | --- |
| `FaceAI(threshold=0.363)` | Creates the offline recognizer. |
| `enroll_frame(name, frame, color_format="bgr")` | Enrols the largest face from any camera/UI frame. |
| `recognize_frame(frame, color_format="bgr")` | Returns result dictionaries; does not open a window. |
| `add_person(name, image_path)` | Desktop helper to enrol from a file. |
| `draw_results(frame, results)` | Optional OpenCV-only drawing helper. |
| `save_db(path)` / `load_db(path)` | Stores or restores enrolment embeddings. |
| `start_webcam(camera_id=0)` | Optional desktop OpenCV demo, not for mobile apps. |

## Mobile notes

- Use Flet/Kivy/native code for camera permission and live preview.
- The Flet example runs recognition on a background task and drops incoming
  frames only while inference is busy, preventing a laggy processing queue.
- Store `faces_db.json` only in private app storage; embeddings are sensitive
  biometric data.
- Test thresholds with your own consented users before release. A default of
  `0.363` is the SFace cosine threshold used by OpenCV's example, not a
  universal security guarantee.

Use face recognition only with informed consent and never as the sole basis for
high-impact decisions.

## License

MIT License.
