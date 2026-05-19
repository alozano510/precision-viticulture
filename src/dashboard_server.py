import threading
from flask import Flask, render_template
from flask_socketio import SocketIO

_app = Flask(__name__)
_socketio = SocketIO(_app, cors_allowed_origins='*', async_mode='threading')

_HTML = """

"""

@_app.route('/')
def _index():
    return render_template('dashboard.html')


class DashboardServer:
    """Thin wrapper: start once, call .emit(dict) from any thread."""

    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=lambda: _socketio.run(
                _app,
                host=self.host,
                port=self.port,
                use_reloader=False,
                log_output=False,   # suppress Flask request noise in the CLI
            ),
            daemon=True,
            name='dashboard-server',
        )
        self._thread.start()
        print(f"[Dashboard] Open  http://<orange-pi-ip>:{self.port}  in your browser")

    def emit(self, data: dict):
        """Push one telemetry sample to all connected browsers."""
        _socketio.emit('telemetry', data)