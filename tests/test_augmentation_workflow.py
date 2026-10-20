import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

import main


class AugmentationWorkflowTests(unittest.TestCase):
    def add_views(self):
        ds = self.data / 'source'
        info_path = ds / 'meta/info.json'
        info = json.loads(info_path.read_text())
        meta_path = ds / 'meta/episodes/chunk-000/file-000.parquet'
        meta = pq.read_table(meta_path).to_pylist()[0]
        for view, width, height in [('wrist', 80, 60), ('side', 160, 120), ('front', 160, 120)]:
            key = f'observation.images.{view}'
            info['features'][key] = dict(dtype='video', shape=[height, width, 3])
            path = ds / f'videos/{key}/chunk-000/file-000.mp4'
            path.parent.mkdir(parents=True)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
            for _ in range(5):
                writer.write(np.full((height, width, 3), 200, np.uint8))
            writer.release()
            for field, value in [('chunk_index', 0), ('file_index', 0), ('from_timestamp', 0.), ('to_timestamp', .5)]:
                meta[f'videos/{key}/{field}'] = value
        info_path.write_text(json.dumps(info))
        pq.write_table(pa.Table.from_pylist([meta]), meta_path)
        return list(info['features'])

    def test_dropout_preview_shows_all_views_and_cycles_per_output_episode(self):
        keys = self.add_views()
        result = self.client.post('/api/datasets/source/augment-preview', json={
            'count': 4, 'techniques': {'camera_dropout': {'enabled': True, 'keep_count': 1}}}).json()
        self.assertTrue(result['ok'], result)
        self.assertEqual(len(result['previews']), 4)
        for index, preview in enumerate(result['previews']):
            self.assertEqual(len(preview['cameras']), 4)
            self.assertEqual(preview['camera_dropout']['kept_cameras'], [keys[index]])
            for view in preview['cameras']:
                frame = cv2.imdecode(np.frombuffer(base64.b64decode(view['augmented']), np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(view['dropped'], view['camera'] != keys[index])
                if view['dropped']:
                    self.assertLess(frame.max(), 10)
                else:
                    self.assertGreater(frame.min(), 150)

    def test_dropout_generation_blacks_entire_videos_and_preserves_shapes_and_actions(self):
        keys = self.add_views()
        source_video = self.data / f'source/videos/{keys[0]}/chunk-000/file-000.mp4'
        before = source_video.read_bytes()
        main._run_augmentation('source', 'dropout', 4, {
            'camera_dropout': {'enabled': True, 'keep_count': 2}}, mode='fix')
        self.assertIsNone(main.augment_state['error'])
        saved = json.loads((self.configs / 'dropout.json').read_text())
        expected = [keys[:2], keys[1:3], keys[2:], [keys[3], keys[0]]]
        for index, kept in enumerate(expected):
            episode = saved['episodes'][index]
            self.assertFalse(episode['is_original'])
            self.assertEqual(episode['camera_dropout']['kept_cameras'], kept)
            for key in keys:
                path = self.data / f'dropout/videos/{key}/chunk-000/file-{index:03d}.mp4'
                cap = cv2.VideoCapture(str(path))
                frames = []
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                cap.release()
                self.assertEqual(len(frames), 5)
                self.assertEqual(frames[0].shape, (60, 80, 3) if key.endswith('wrist') else (120, 160, 3))
                for frame in frames:
                    if key not in kept:
                        self.assertLess(frame.max(), 10)
                    else:
                        self.assertGreater(frame.min(), 150)
        rows = pq.read_table(self.data / 'dropout/data/chunk-000/file-000.parquet').to_pylist()
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(row['action'] == [1., 2.] for row in rows))
        self.assertEqual(source_video.read_bytes(), before)

    def test_dropout_invalid_request_is_rejected_before_creating_dataset(self):
        for suffix, body in [('augment', {'name': 'invalid', 'target_episodes': 2}), ('augment-preview', {})]:
            response = self.client.post(f'/api/datasets/source/{suffix}', json={**body,
                'techniques': {'camera_dropout': {'enabled': True, 'keep_count': 0}}}).json()
            self.assertFalse(response['ok'], response)
            self.assertIn('Camera Dropout', response['error'])
        self.assertFalse((self.data / 'invalid').exists())

    def test_dropout_ratio_matches_preview_generation_and_saved_metadata(self):
        keys = self.add_views()
        techniques = {'camera_dropout': {'enabled': True, 'keep_count': 1, 'drop_camera_percent': 40}}
        previews = self.client.post('/api/datasets/source/augment-preview', json={
            'count': 5, 'target_episodes': 5, 'techniques': techniques}).json()['previews']
        expected = [[keys[0]], keys, [keys[1]], keys, keys]
        self.assertEqual([p['camera_dropout']['kept_cameras'] for p in previews], expected)
        main._run_augmentation('source', 'mixed', 5, techniques, mode='fix')
        self.assertIsNone(main.augment_state['error'])
        saved = json.loads((self.configs / 'mixed.json').read_text())
        self.assertEqual(saved['techniques']['camera_dropout']['drop_camera_percent'], 40)
        self.assertEqual([ep['camera_dropout']['kept_cameras'] for ep in saved['episodes']], expected)
        for index, kept in enumerate(expected):
            for key in keys:
                cap = cv2.VideoCapture(str(self.data / f'mixed/videos/{key}/chunk-000/file-{index:03d}.mp4'))
                ok, frame = cap.read()
                cap.release()
                self.assertTrue(ok)
                if key in kept:
                    self.assertGreater(frame.min(), 150)
                else:
                    self.assertLess(frame.max(), 10)

    def test_evaluation_dropout_preview_matches_live_parameters_for_every_view(self):
        keys = self.add_views()
        challenge = {'enabled': True, 'camera': {'enabled': False}, 'light': {'enabled': False},
                     'camera_dropout': {'enabled': True, 'drop_camera_percent': 50, 'keep_count': 1}}
        result = self.client.post('/api/eval/challenge-preview', json={
            'dataset': 'source', 'count': 4, 'target_runs': 4, 'challenge_env': challenge}).json()
        self.assertTrue(result['ok'], result)
        self.assertEqual(len(result['previews']), 16)
        for preview in result['previews']:
            episode = preview['challenge_episode']
            self.assertEqual(preview['params'], main._eval_challenge_params(
                challenge, episode, 30, 12, camera_keys=keys, total_episodes=4))
            black = episode in (1, 3) and preview['camera'] != ('top' if episode == 1 else 'wrist')
            self.assertEqual(preview['dropped'], black)
            image = cv2.imdecode(np.frombuffer(base64.b64decode(preview['augmented']), np.uint8), cv2.IMREAD_COLOR)
            if black:
                self.assertLess(image.max(), 10)
            else:
                self.assertGreater(image.min(), 150)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.configs = self.root / 'augment_configs'
        self.configs.mkdir()
        for key, value in [('ROOT', self.root), ('DATA_DIR', self.data),
                           ('AUGMENT_CONFIG_DIR', self.configs),
                           ('augment_state', dict(running=False, progress=0, status='', error=None))]:
            p = patch.object(main, key, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app)
        self.cam = 'observation.images.top'
        ds = self.data / 'source'
        for directory in ['meta/episodes/chunk-000', 'data/chunk-000',
                          f'videos/{self.cam}/chunk-000']:
            (ds / directory).mkdir(parents=True)
        info = dict(fps=10, total_episodes=1, total_frames=5,
                    features={self.cam: dict(dtype='video', shape=[120, 160, 3])})
        (ds / 'meta/info.json').write_text(json.dumps(info))
        pq.write_table(pa.table({'task': ['pick cube']}), ds / 'meta/tasks.parquet')
        pq.write_table(pa.Table.from_pylist([dict(episode_index=0, frame_index=i,
            index=i, timestamp=i / 10, task_index=0, action=[1., 2.],
            **{'observation.state': [3., 4.]}) for i in range(5)]),
            ds / 'data/chunk-000/file-000.parquet')
        pq.write_table(pa.Table.from_pylist([dict(episode_index=0, length=5,
            tasks=['pick cube'], **{f'videos/{self.cam}/chunk_index': 0,
            f'videos/{self.cam}/file_index': 0, f'videos/{self.cam}/from_timestamp': 0.,
            f'videos/{self.cam}/to_timestamp': .5})]),
            ds / 'meta/episodes/chunk-000/file-000.parquet')
        writer = cv2.VideoWriter(str(ds / f'videos/{self.cam}/chunk-000/file-000.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
        for _ in range(5):
            writer.write(np.full((120, 160, 3), 200, np.uint8))
        writer.release()

    def test_preview_occlusion_works_with_light_disabled(self):
        response = self.client.post('/api/datasets/source/augment-preview', json={
            'mode': 'fix', 'count': 1, 'techniques': {'occlusion': {'enabled': True},
            'light': {'enabled': False, 'brightness': True}},
            'fixed_params': {'light': {'occlusion': [.25, .25, .5, .5], 'brightness': .5}}}).json()
        self.assertTrue(response['ok'])
        preview = response['previews'][0]
        original, augmented = [cv2.imdecode(np.frombuffer(base64.b64decode(preview[k]),
            np.uint8), cv2.IMREAD_COLOR) for k in ['original', 'augmented']]
        h, w = augmented.shape[:2]
        self.assertLess(augmented[h//2, w//2].max(), 10)
        np.testing.assert_allclose(augmented[0, 0], original[0, 0], atol=5)

    def test_challenge_preview_uses_evaluation_parameters_and_images(self):
        challenge = {'enabled': True, 'mode': 'fix',
            'camera': {'enabled': False}, 'light': {'enabled': False},
            'occlusion': {'enabled': True},
            'robot': {'enabled': True, 'initial_noise': True, 'trajectory_jitter': True},
            'fixed_params': {'light': {'occlusion': [.25,.25,.5,.5]},
                             'robot': {'offset_sigma': 1, 'jitter_s': .5}}}
        body = {'dataset': 'source', 'episode': 3, 'fps': 30, 'duration': 12, 'width':160, 'height':120,
                'challenge_env': challenge}
        response = self.client.post('/api/eval/challenge-preview', json=body)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result['ok'], result)
        preview = result['previews'][0]
        self.assertEqual(preview['params'], main._eval_challenge_params(challenge,3,30,12))
        self.assertEqual(preview['challenge_episode'], 3)
        self.assertEqual(preview['source_episode'], 0)
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(preview['augmented']), np.uint8), cv2.IMREAD_COLOR)
        self.assertLess(frame[60,80].max(), 10)
        self.assertGreater(frame[0,0].min(), 150)
        self.assertEqual(result, self.client.post('/api/eval/challenge-preview', json=body).json())

    def test_challenge_preview_transforms_at_evaluation_resolution(self):
        import infer_python
        shapes = []
        transform = infer_python._aug_frame
        def observed(frame, *args, **kwargs):
            shapes.append(frame.shape)
            return transform(frame, *args, **kwargs)
        with patch.object(infer_python, '_aug_frame', side_effect=observed):
            result = self.client.post('/api/eval/challenge-preview', json={
                'dataset':'source', 'width':640, 'height':480}).json()
        self.assertTrue(result['ok'],result)
        self.assertEqual(shapes,[(480,640,3)])
        preview = cv2.imdecode(np.frombuffer(base64.b64decode(result['previews'][0]['original']),np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(preview.shape,(240,320,3))

    def test_challenge_preview_disabled_is_unchanged_and_missing_dataset_is_explained(self):
        result = self.client.post('/api/eval/challenge-preview', json={
            'dataset': 'source', 'challenge_env': {'enabled': False}}).json()
        self.assertTrue(result.get('ok'), result)
        preview = result['previews'][0]
        self.assertEqual(preview['original'], preview['augmented'])
        missing = self.client.post('/api/eval/challenge-preview', json={'dataset': 'missing'}).json()
        self.assertFalse(missing['ok'])
        self.assertIn('dataset', missing['error'].lower())

    def test_generation_saves_occlusion_and_preserves_original(self):
        for mode in ['fix', 'random']:
            with self.subTest(mode=mode):
                main._run_augmentation('source', mode, 2, {
                    'occlusion': {'enabled': True},
                    'light_quality': {'enabled': False, 'brightness': True}},
                    mode=mode, fixed_params={'light': {'occlusion': [.25, .25, .5, .5]}}
                    if mode == 'fix' else None)
                self.assertIsNone(main.augment_state['error'])
                self.assertEqual(main.augment_state['progress'], 100)
                config = self.client.get(f'/api/datasets/{mode}/augment-config').json()['config']
                self.assertTrue(config['episodes'][0]['is_original'])
                self.assertIsNone(config['episodes'][0]['light'])
                self.assertIn('occlusion', config['episodes'][1]['light'] or {})
                self.assertNotIn('brightness', config['episodes'][1]['light'])
                output = self.data / mode
                rows = pq.read_table(output / 'data/chunk-000/file-000.parquet').to_pylist()
                self.assertEqual(len(rows), 10)
                self.assertEqual([r['index'] for r in rows], list(range(10)))
                self.assertEqual(rows[5]['frame_index'], 0)
                self.assertEqual(rows[5]['timestamp'], 0)
                self.assertEqual(rows[5]['action'], [1., 2.])
                tasks = pq.read_table(output / 'meta/tasks.parquet').to_pydict()
                self.assertEqual(tasks['task'], ['pick cube'])
                self.assertEqual(tasks['task_index'], [0])
                stats = json.loads((output / 'meta/stats.json').read_text())
                self.assertIn(self.cam, stats)
                cap = cv2.VideoCapture(str(output / f'videos/{self.cam}/chunk-000/file-001.mp4'))
                ok, frame = cap.read()
                cap.release()
                self.assertTrue(ok)
                self.assertGreater(np.all(frame < 15, axis=2).sum(), 0)

    def test_rename_moves_augmentation_config(self):
        (self.configs / 'source.json').write_text('{"episodes": []}')
        response = self.client.post('/api/datasets/source/rename', json={'new_name': 'renamed'}).json()
        self.assertTrue(response['ok'])
        self.assertFalse((self.configs / 'source.json').exists())
        self.assertTrue(self.client.get('/api/datasets/renamed/augment-config').json()['is_augmented'])

    def test_delete_episodes_remaps_augmentation_metadata(self):
        main._run_augmentation('source', 'generated', 2,
            {'occlusion': {'enabled': True}}, mode='fix',
            fixed_params={'light': {'occlusion': [.25, .25, .5, .5]}})
        self.assertIsNone(main.augment_state['error'])
        response = self.client.post('/api/datasets/generated/delete-episodes',
                                    json={'episodes': [0]}).json()
        self.assertTrue(response['ok'])
        config = self.client.get('/api/datasets/generated/augment-config').json()['config']
        self.assertEqual(len(config['episodes']), 1)
        episode = config['episodes'][0]
        self.assertEqual(episode['new_ep'], 0)
        self.assertEqual(episode['source_ep'], 0)
        self.assertFalse(episode['is_original'])
        self.assertEqual(episode['light']['occlusion'], [.25, .25, .5, .5])
        self.assertTrue(self.client.post('/api/datasets/generated/delete-episodes',
            json={'episodes': [0]}).json()['ok'])
        config = self.client.get('/api/datasets/generated/augment-config').json()['config']
        self.assertEqual(config['episodes'], [])

    def test_delete_removes_augmentation_config(self):
        (self.configs / 'source.json').write_text('{"episodes": []}')
        self.assertTrue(self.client.delete('/api/datasets/source').json()['ok'])
        self.assertFalse((self.configs / 'source.json').exists())
