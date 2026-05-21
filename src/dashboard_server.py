import cv2
import threading
from flask import Flask, Response, render_template
from flask_socketio import SocketIO

_app = Flask(__name__)
_socketio = SocketIO(_app, cors_allowed_origins='*', async_mode='threading')

_frame_source = None

def _mjpeg_generator():
    while True:
        frame = _frame_source() if _frame_source else None
        if frame is None:
            continue
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            jpg.tobytes() +
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

    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port
        self._thread = None

    def set_frame_source(self, getter_fn):
        """
        Register a callable that returns the latest BGR frame.
        Example: dashboard.set_frame_source(vine_classifier.get_latest_frame)
        """
        global _frame_source
        _frame_source = getter_fn

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
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
        self._thread.start()
        print(f"[Dashboard] Open  http://<orange-pi-ip>:{self.port}  in your browser")
        print(f"[Video]     http://<orange-pi-ip>:{self.port}/video")

    def emit(self, data: dict):
        """Push one telemetry sample to all connected browsers."""
        _socketio.emit('telemetry', data)