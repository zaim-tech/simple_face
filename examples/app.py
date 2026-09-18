"""Flet Camera test app for Simple Face.

Run from this directory with ``flet run app.py``. The Flet Camera control
does not currently support Windows, macOS, or Linux; use examples/main.py for a
desktop OpenCV camera test instead.
"""

import asyncio
import base64
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import flet as ft
import flet_camera as fc
import numpy as np

# Allow `flet run examples/app.py` without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simple_face import FaceAI


@dataclass
class CameraState:
    cameras: list[fc.CameraDescription] = field(default_factory=list)
    selected: fc.CameraDescription | None = None
    initialized: bool = False
    streaming_supported: bool = False
    streaming: bool = False
    fallback_capturing: bool = False


async def main(page: ft.Page):
    page.title = "Simple Face Camera"
    page.padding = 16
    page.scroll = ft.ScrollMode.AUTO
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    ai = FaceAI()
    reference_image = Path(__file__).with_name("zaim.png")
    enrolled = ai.add_person("Zaim", str(reference_image))
    state = CameraState()
    processing = False

    status = ft.Text(
        "Zaim reference loaded. Start the camera."
        if enrolled
        else "Could not enrol zaim.png; make sure it contains one clear face."
    )
    camera_picker = ft.Dropdown(label="Camera", width=280)
    processed_preview = ft.Image(
        src=base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wIAAgMBAp0YVwAAAABJRU5ErkJggg=="
        ),
        width=500,
        height=360,
        fit=ft.BoxFit.CONTAIN,
        gapless_playback=True,
        visible=False,
    )

    async def recognize_sample(frame: np.ndarray) -> None:
        """Run inference away from the Flet camera callback."""
        nonlocal processing
        try:
            faces = await asyncio.to_thread(ai.recognize_frame, frame)
            if faces:
                status.value = "Zaim detected." if any(face["name"] == "Zaim" for face in faces) else "Face detected: Unknown"
            else:
                status.value = "No face detected."
            # Render the same labelled BGR frame that the OpenCV desktop demo
            # shows with cv2.imshow(), but through a Flet Image control.
            labelled_frame = ai.draw_results(frame.copy(), faces)
            encoded_ok, jpeg = cv2.imencode(".jpg", labelled_frame)
            if encoded_ok:
                processed_preview.src = jpeg.tobytes()
                processed_preview.visible = True
                processed_preview.update()
            status.update()
        except Exception as error:
            logging.exception("Face recognition failed: %s", error)
            status.value = "Recognition error; see application log."
            status.update()
        finally:
            processing = False

    def process_image_bytes(image_bytes: bytes) -> None:
        """Show each stream frame, then replace it with the labelled AI result."""
        nonlocal processing
        # A new event may arrive while the previous inference is running. Drop
        # it so the displayed result stays current instead of building a laggy queue.
        if processing:
            return
        # This matches Flet's Camera example: show the incoming JPEG immediately.
        processed_preview.src = image_bytes
        processed_preview.visible = True
        processed_preview.update()
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            return
        processing = True
        page.run_task(recognize_sample, frame)

    def on_stream_image(event: fc.CameraImageEvent) -> None:
        process_image_bytes(event.bytes)

    async def capture_fallback_loop() -> None:
        """Web fallback when the selected camera cannot provide an image stream."""
        state.fallback_capturing = True
        try:
            while state.initialized and not state.streaming:
                image_bytes = await camera.take_picture()
                if image_bytes:
                    process_image_bytes(image_bytes)
                # Avoid repeatedly requesting still captures faster than a phone
                # or browser can supply them.
                await asyncio.sleep(0.15)
        except Exception as error:
            logging.exception("Camera capture fallback failed: %s", error)
            status.value = "Camera capture failed; see application log."
            status.update()
        finally:
            state.fallback_capturing = False

    camera = fc.Camera(
        expand=True,
        preview_enabled=True,
        content=ft.Container(
            alignment=ft.Alignment.CENTER,
            content=ft.Icon(ft.Icons.FACE, color=ft.Colors.WHITE_70, size=48),
        ),
    )
    camera.on_stream_image = on_stream_image

    async def load_cameras(event=None) -> None:
        state.cameras = await camera.get_available_cameras()
        camera_picker.options = [
            ft.DropdownOption(key=item.name, text=item.name) for item in state.cameras
        ]
        if state.cameras and camera_picker.value is None:
            camera_picker.value = state.cameras[0].name
        state.selected = next(
            (item for item in state.cameras if item.name == camera_picker.value), None
        )
        camera_picker.update()

    async def start_camera(event) -> None:
        if not state.cameras:
            await load_cameras()
        state.selected = next(
            (item for item in state.cameras if item.name == camera_picker.value), None
        )
        if state.selected is None:
            status.value = "No camera was found on this device."
            status.update()
            return
        status.value = "Starting camera…"
        status.update()
        await camera.initialize(
            description=state.selected,
            resolution_preset=fc.ResolutionPreset.MEDIUM,
            enable_audio=False,
            image_format_group=fc.ImageFormatGroup.JPEG,
        )
        if not page.web:
            try:
                await camera.lock_capture_orientation()
            except RuntimeError as error:
                logging.warning("Could not lock camera orientation: %s", error)
        state.initialized = True
        state.streaming_supported = await camera.supports_image_streaming()
        if not state.streaming_supported:
            status.value = "Image streaming is unavailable; using camera-capture fallback."
            status.update()
            if not state.fallback_capturing:
                page.run_task(capture_fallback_loop)
            return
        await camera.start_image_stream()
        state.streaming = True
        status.value = "Live recognition is running."
        status.update()

    async def on_state_change(event: fc.CameraStateEvent) -> None:
        """Reflect camera-extension errors and stream state in the interface."""
        if event.description != state.selected:
            return
        state.streaming = event.is_streaming_images
        if event.has_error:
            status.value = f"Camera error: {event.error_description}"
            status.update()

    camera.on_state_change = on_state_change

    page.add(
        ft.SafeArea(
            content=ft.Column(
                width=500,
                controls=[
                    ft.Text("Simple Face", theme_style=ft.TextThemeStyle.HEADLINE_MEDIUM),
                    ft.Text("Zaim face-recognition demo. Recognition stays on-device."),
                    camera_picker,
                    ft.FilledButton("Start camera", on_click=start_camera),
                    ft.Container(camera, height=360, border_radius=12, clip_behavior=ft.ClipBehavior.ANTI_ALIAS),
                    ft.Text("Recognition preview", weight=ft.FontWeight.BOLD),
                    ft.Container(
                        processed_preview,
                        height=360,
                        bgcolor=ft.Colors.BLACK,
                        border_radius=12,
                        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                    ),
                    status,
                ],
            )
        ),
    )
    page.on_connect = load_cameras
    await load_cameras()


if __name__ == "__main__":
    ft.run(main)
