import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from main import _aug_frame, _aug_process_video, _augp_light


class OcclusionTests(unittest.TestCase):
    def test_fix_color_jitter_accepts_unset_optional_offsets(self):
        params = _augp_light({"color_jitter": True, "occlusion": True}, fixed={
            "hue": 4, "sat_off": None, "val_off": None,
            "occlusion": [.1, .2, .3, .4],
        })
        source = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = _aug_frame(source, None, params)
        self.assertTrue(np.all(result[20:60, 10:40] == 0))
        np.testing.assert_array_equal(result[0, 0], source[0, 0])

    def test_fixed_rectangle_uses_requested_position_and_size(self):
        source = np.full((100, 200, 3), 200, dtype=np.uint8)
        expected = source.copy()
        expected[30:70, 20:60] = 0
        for intensity in (1, 5):
            params = _augp_light({"occlusion": True}, intensity,
                                 {"occlusion": [0.1, 0.3, 0.2, 0.4]})
            np.testing.assert_array_equal(_aug_frame(source, None, params), expected)

    def test_fixed_rectangle_stays_inside_image_at_edges(self):
        source = np.full((100, 100, 3), 200, dtype=np.uint8)
        expected = source.copy()
        expected[70:100, 80:100] = 0
        params = _augp_light({"occlusion": True}, fixed={"occlusion": [1, 1, .2, .3]})
        np.testing.assert_array_equal(_aug_frame(source, None, params), expected)

    def test_black_rectangle_is_stable_and_does_not_change_source(self):
        source = np.full((120, 160, 3), 200, dtype=np.uint8)
        for fixed in (None, {"brightness": 1.1}):
            params = _augp_light({"occlusion": True}, fixed=fixed)
            result = _aug_frame(source, None, params)
            mask = np.all(result == 0, axis=2)
            self.assertGreater(mask.sum(), 0)
            self.assertLess(mask.sum(), mask.size)
            ys, xs = np.where(mask)
            self.assertEqual(mask.sum(), (ys.max()-ys.min()+1)*(xs.max()-xs.min()+1))
            np.testing.assert_array_equal(result, _aug_frame(source, None, params))
            self.assertTrue(np.all(source == 200))

    def test_positions_and_sizes_vary_and_intensity_scales_size(self):
        frame = np.full((120, 160, 3), 200, dtype=np.uint8)
        boxes = []
        for intensity in (1, 5):
            areas = []
            for _ in range(20):
                result = _aug_frame(frame, None, _augp_light({"occlusion": True}, intensity))
                ys, xs = np.where(np.all(result == 0, axis=2))
                self.assertGreater(len(xs), 0)
                boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
                areas.append(len(xs))
            if intensity == 1:
                small_max = max(areas)
            else:
                self.assertGreater(min(areas), small_max)
        self.assertGreater(len(set(boxes)), 10)

    def test_disabled_is_unchanged_and_noise_does_not_lighten_rectangle(self):
        frame = np.full((120, 160, 3), 200, dtype=np.uint8)
        np.testing.assert_array_equal(frame, _aug_frame(frame, None, _augp_light({})))
        params = _augp_light({"occlusion": True, "noise": True, "blur": True})
        self.assertTrue(np.any(np.all(_aug_frame(frame, None, params) == 0, axis=2)))

    def test_video_output_retains_frame_count_and_stable_black_area(self):
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory)/"src.mp4", Path(directory)/"out.mp4"
            writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
            for _ in range(5):
                writer.write(np.full((120, 160, 3), 200, dtype=np.uint8))
            writer.release()
            params = _augp_light({"occlusion": True})
            _, _, count = _aug_process_video(src, dst, 0, .5, 10, None, params, 160, 120, expected_frames=5)
            self.assertEqual(count, 5)
            cap = cv2.VideoCapture(str(dst))
            masks = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                masks.append(np.all(frame < 15, axis=2))
            cap.release()
            self.assertEqual(len(masks), 5)
            self.assertGreater(np.logical_and.reduce(masks).sum(), 0)


if __name__ == "__main__":
    unittest.main()
