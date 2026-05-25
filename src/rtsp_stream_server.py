import time
import threading
import cv2
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

Gst.init(None)

class _VideoFactory(GstRtspServer.RTSPMediaFactory):
    """
    GStreamer RTSP media factory.
    Builds the pipeline once (shared=True) and pushes annotated
    frames from the vision system into it via appsrc.
    """

    def __init__(self, frame_source, fps: int, width: int, height: int,
                 use_hw_encoder: bool):
        super().__init__()
        self._frame_source = frame_source
        self._fps = fps
        self._width = width
        self._height = height
        self._use_hw_encoder = use_hw_encoder
        self._frame_duration = int(Gst.SECOND / fps)
        self._appsrc = None
        self._running = False

        # Share one pipeline across all connected clients
        self.set_shared(True)

    def do_create_element(self, url):
        encoder = (
            "mpph264enc"                                              # Rockchip NPU
            if self._use_hw_encoder else
            "x264enc tune=zerolatency bitrate=800 speed-preset=ultrafast"  # software
        )

        pipeline_str = (
            f"appsrc name=src is-live=true block=true format=time "
            f"caps=video/x-raw,format=BGR,"
            f"width={self._width},height={self._height},"
            f"framerate={self._fps}/1 "
            f"! videoconvert "
            f"! {encoder} "
            f"! h264parse "
            f"! rtph264pay name=pay0 pt=96 config-interval=1"
        )

        pipeline = Gst.parse_launch(pipeline_str)
        self._appsrc = pipeline.get_by_name('src')

        self._running = True
        threading.Thread(
            target=self._push_frames,
            daemon=True,
            name='rtsp-push'
        ).start()

        return pipeline

    def _push_frames(self):
        timestamp = 0
        interval = 1.0 / self._fps

        while self._running:
            t0 = time.monotonic()

            frame = self._frame_source()
            if frame is None:
                time.sleep(0.01)
                continue

            # Resize only if the frame doesn't match the configured resolution
            h, w = frame.shape[:2]
            if w != self._width or h != self._height:
                frame = cv2.resize(frame, (self._width, self._height))

            raw = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(raw), None)
            buf.fill(0, raw)
            buf.pts = timestamp
            buf.dts = timestamp
            buf.duration = self._frame_duration
            timestamp += self._frame_duration

            flow = self._appsrc.emit('push-buffer', buf)
            if flow != Gst.FlowReturn.OK:
                break

            elapsed = time.monotonic() - t0
            remainder = interval - elapsed
            if remainder > 0:
                time.sleep(remainder)

    def stop(self):
        self._running = False


class RTSPStreamServer:
    """
    Standalone RTSP server that streams annotated frames from the
    vision system. Drop-in alongside the existing DashboardServer.

    Usage:
        rtsp = RTSPStreamServer()
        rtsp.set_frame_source(vine_classifier.get_latest_frame)
        rtsp.start()
        # Connect with VLC: rtsp://<orange-pi-ip>:8554/stream
    """

    def __init__(self,
                 host: str = '0.0.0.0',
                 port: int = 8554,
                 path: str = '/stream',
                 fps: int = 20,
                 width: int = 640,
                 height: int = 480,
                 use_hw_encoder: bool = True):
        self.host = host
        self.port = port
        self.path = path
        self.fps = fps
        self.width = width
        self.height = height
        self.use_hw_encoder = use_hw_encoder

        self._frame_source = None
        self._factory: _VideoFactory | None = None
        self._loop: GLib.MainLoop | None = None
        self._thread: threading.Thread | None = None

    def set_frame_source(self, getter_fn):
        self._frame_source = getter_fn

    def start(self):
        if self._thread and self._thread.is_alive():
            return

        server = GstRtspServer.RTSPServer()
        server.set_address(self.host)
        server.set_service(str(self.port))

        self._factory = _VideoFactory(
            frame_source=self._frame_source,
            fps=self.fps,
            width=self.width,
            height=self.height,
            use_hw_encoder=self.use_hw_encoder,
        )

        server.get_mount_points().add_factory(self.path, self._factory)
        server.attach(None)

        self._loop = GLib.MainLoop()
        self._thread = threading.Thread(
            target=self._loop.run,
            daemon=True,
            name='rtsp-server',
        )
        self._thread.start()
        print(f"[RTSP] rtsp://<orange-pi-ip>:{self.port}{self.path}")

    def stop(self):
        if self._factory:
            self._factory.stop()
        if self._loop:
            self._loop.quit()