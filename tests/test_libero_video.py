import importlib.util
from pathlib import Path
import tempfile
import unittest

from libero_video import EpisodeVideoWriter


@unittest.skipUnless(importlib.util.find_spec('av') and importlib.util.find_spec('numpy'), 'Requires video runtime')
class EpisodeVideoTests(unittest.TestCase):
    def test_h264_video_flushes_all_frames_and_supports_streaming(self):
        import av
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'episode.mp4'
            writer = EpisodeVideoWriter(path, 64, 64)
            for value in range(10):
                writer.write(np.full((64, 64, 3), value * 20, dtype=np.uint8))
            self.assertFalse(path.exists())
            writer.release()
            writer.release()
            with av.open(str(path)) as video:
                self.assertEqual(video.streams.video[0].codec_context.name, 'h264')
                frames = list(video.decode(video=0))
                self.assertEqual(len(frames), 10)
                self.assertEqual(frames[0].format.name, 'yuv420p')
            data = path.read_bytes()
            self.assertLess(data.index(b'moov'), data.index(b'mdat'))
            self.assertFalse(writer.temp.exists())
