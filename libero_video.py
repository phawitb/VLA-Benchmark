"""Browser-compatible episode videos, published only after the encoder flushes."""
from pathlib import Path


class EpisodeVideoWriter:
    def __init__(self, path, width, height, fps=20):
        import av
        self.av = av
        self.path = Path(path)
        self.temp = self.path.with_suffix('.pending.mp4')
        self.container = av.open(str(self.temp), mode='w', options={'movflags': '+faststart'})
        try:
            self.stream = self.container.add_stream('libx264', rate=fps)
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = 'yuv420p'
            self.stream.options = {'crf': '23', 'preset': 'fast'}
        except Exception:
            self.container.close()
            raise

    def write(self, bgr):
        frame = self.av.VideoFrame.from_ndarray(bgr, format='bgr24')
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def release(self):
        if self.container is None:
            return
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
            self.container = None
        self.temp.replace(self.path)
