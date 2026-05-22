import cv2
import time
import threading
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

_app = Flask(__name__)
_socketio = SocketIO(_app, cors_allowed_origins='*', async_mode='threading')

_frame_source = None

_encoded_lock = threading.Lock()
_encoded_frame: bytes = None
# Activate a thread event each time a new frame is ready
_frame_event = threading.Event()

def _encoding_worker(target_fps = 15, jpeg_quality = 55):
    """
    Pulls the latest raw frame on a background thread at a fixed rate,
    encodes it once, and stores the bytes. The MJPEG generators read from this cache.
    """
    global _encoded_frame
    interval = 1.0 / target_fps

    while True:
        t0 = time.monotonic()

        frame = _frame_source() if _frame_source else None
        if frame is not None:

            ok, buf = cv2.imencode(
                '.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if ok:
                with _encoded_lock:
                    _encoded_frame = buf.tobytes()
                _frame_event.set()

        elapsed = time.monotonic() - t0
        sleep_for = interval - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

def _mjpeg_generator():
    while True:
        _frame_event.wait()
        _frame_event.clear()

        with _encoded_lock:
            data = _encoded_frame

        if data is None:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            data +
            b'\r\n'
        )

@_app.route('/video')
def _video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@_app.route('/')
def _index():
    return render_template('dashboard.html')


class DashboardServer:
    """Thin wrapper: start once, call .emit(dict) from any thread."""

    def __init__(self, host: str = '0.0.0.0', port: int = 5000,
                 stream_fps: int = 15, jpeg_quality: int = 55):
        self.host = host
        self.port = port
        self.stream_fps = stream_fps
        self.jpeg_quality = jpeg_quality
        self._server_thread = None
        self._encoder_thread = None

    def set_frame_source(self, getter_fn):
        """
        Register a callable that returns the latest BGR frame.
        Example: dashboard.set_frame_source(vine_classifier.get_latest_frame)
        """
        global _frame_source
        _frame_source = getter_fn

    def start(self):
        if self._server_thread and self._server_thread.is_alive():
            return

        # Start encoder
        self._encoder_thread = threading.Thread(
            target=_encoding_worker,
            kwargs={'target_fps': self.stream_fps, 'jpeg_quality': self.jpeg_quality},
            daemon=True,
            name='frame-encoder',
        )
        self._encoder_thread.start()

        # Start Flask server
        self._server_thread = threading.Thread(
            target=lambda: _socketio.run(
                _app,
                host=self.host,
                port=self.port,
                use_reloader=False,
                log_output=False,
            ),
            daemon=True,
            name='dashboard-server',
        )
        self._server_thread.start()
        print(f"[Dashboard] Open  http://<orange-pi-ip>:{self.port}  in your browser")
        print(f"[Video]     http://<orange-pi-ip>:{self.port}/video")

    def emit(self, data: dict):
        """Push one telemetry sample to all connected browsers."""
        _socketio.emit('telemetry', data)