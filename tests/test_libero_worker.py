"""Worker contract tests use a fake simulator; no weights, robot, or GUI needed."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from libero_worker import run_episode, save_result, successful, validate_action_features, validate_config
from libero_worker import configure_evaluation_checkpoint
from libero_worker import setup_policy_dependencies


class FakeEnv:
    def __init__(self, success_step=None, terminate_step=None):
        self.success_step = success_step
        self.terminate_step = terminate_step
        self.steps = 0

    def reset(self, seed):
        self.seed = seed
        self.steps = 0
        return {'step': 0}, {}

    def step(self, action):
        self.steps += 1
        return {'step': self.steps}, 999, self.steps == self.terminate_step, False, {'is_success': self.steps == self.success_step}


class WorkerTests(unittest.TestCase):
    def test_discovers_causal_aug_beside_active_lerobot_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / 'causal_aug' / 'causal_aug'
            package.mkdir(parents=True)
            (package / '__init__.py').touch()
            spec = SimpleNamespace(submodule_search_locations=[str(root / 'lerobot/src/lerobot')])
            with patch('libero_worker.importlib.util.find_spec', side_effect=lambda name: spec if name == 'lerobot' else None), patch('libero_worker.sys.path', []) as paths:
                setup_policy_dependencies()
                self.assertEqual(paths, [str((root / 'causal_aug').resolve())])

    def test_preserves_installed_causal_aug(self):
        with patch('libero_worker.importlib.util.find_spec', return_value=object()), patch('libero_worker.sys.path', []) as paths:
            setup_policy_dependencies()
            self.assertEqual(paths, [])

    def test_evaluation_does_not_reuse_training_base_revision(self):
        cfg = SimpleNamespace(pretrained_path='lerobot/smolvla_base',
                              pretrained_revision='base-model-commit', device='cuda')
        configure_evaluation_checkpoint(cfg, 'owner/finetuned-policy', 'mps')
        self.assertEqual(cfg.pretrained_path, 'owner/finetuned-policy')
        self.assertIsNone(cfg.pretrained_revision)
        self.assertEqual(cfg.device, 'mps')

    def test_success_stops_and_uses_environment_truth(self):
        env = FakeEnv(success_step=2)
        updates = []
        record = run_episode(env, lambda obs, step: step, seed=34, max_steps=10, progress=updates.append)
        self.assertTrue(record['success'])
        self.assertEqual(record['steps'], 2)
        self.assertEqual(updates, [1, 2])
        self.assertEqual(env.seed, 34)

    def test_reward_and_termination_do_not_mean_success(self):
        record = run_episode(FakeEnv(terminate_step=1), lambda *_: 0, seed=1, max_steps=5, progress=lambda _: None)
        self.assertFalse(record['success'])
        self.assertEqual(record['steps'], 1)

    def test_step_limit_and_cancel(self):
        env = FakeEnv()
        record = run_episode(env, lambda *_: 0, seed=1, max_steps=3, progress=lambda _: None)
        self.assertEqual(record['steps'], 3)
        record = run_episode(env, lambda *_: 0, seed=1, max_steps=20, progress=lambda _: None, stopped=lambda: env.steps >= 2)
        self.assertEqual(record['steps'], 2)
        self.assertTrue(record['cancelled'])

    def test_cancel_during_inference_never_steps(self):
        cancelled = [False]
        def infer(*_):
            cancelled[0] = True
            return 0
        env = FakeEnv()
        record = run_episode(env, infer, seed=0, max_steps=2, progress=lambda _: None, stopped=lambda: cancelled[0])
        self.assertEqual(env.steps, 0)
        self.assertTrue(record['cancelled'])

    def test_final_info(self):
        self.assertTrue(successful({'final_info': [{'is_success': True}]}))
        self.assertFalse(successful({'final_info': [None]}))
        self.assertTrue(successful({'final_info': {'is_success': True}}))

    def test_reject_six_action_checkpoint_before_factory_overwrites_features(self):
        with self.assertRaisesRegex(ValueError, '7-dimensional'):
            validate_action_features({'action': SimpleNamespace(shape=(6,))})
        validate_action_features({'action': SimpleNamespace(shape=(7,))})

    def test_reject_so101_challenges(self):
        cfg = {'model_path': 'x', 'episodes': 1, 'max_steps': 2,
               'challenge_env': {'enabled': True, 'robot': {'enabled': True, 'initial_noise': True}}}
        with self.assertRaisesRegex(ValueError, 'SO101'):
            validate_config(cfg)

    def test_partial_results_atomic_and_cancel_not_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_result(output, {'task_ids': [1, 2], 'episodes': 2},
                        [{'success': True}, {'success': False, 'cancelled': True}], 'cancelled')
            result = json.loads((output / 'result.json').read_text())
            self.assertEqual(result['completed'], 1)
            self.assertEqual(result['total'], 4)
            self.assertEqual(result['success_rate'], 1.0)
            self.assertFalse((output / 'result.json.tmp').exists())

    def test_seeded_image_noise_restarts_for_each_episode(self):
        try:
            import numpy as np
            import torch
            import cv2
        except ImportError:
            self.skipTest('image runtime is optional for protocol unit tests')
        from libero_worker import ImageChallenge
        cfg = {'enabled': True, 'mode': 'fix', 'light': {'enabled': True, 'noise': True},
               'fixed_params': {'light': {'noise_s': 10}}}
        obs = {'observation.images.image': torch.ones(1, 3, 16, 16) * .5}
        first = ImageChallenge(cfg, 123, 10)
        second = ImageChallenge(cfg, 123, 10)
        a = first.apply(obs)['observation.images.image']
        b = second.apply(obs)['observation.images.image']
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, first.apply(obs)['observation.images.image']))
        self.assertTrue(torch.equal(obs['observation.images.image'], torch.ones(1, 3, 16, 16) * .5))


class AssetDiscoveryTests(unittest.TestCase):
    def test_catalog_and_runner_share_cached_asset_fallback(self):
        from unittest.mock import patch
        from libero_worker import libero_asset_paths
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            cached = home / '.cache/libero/assets'
            cached.mkdir(parents=True)
            with patch.object(Path, 'home', return_value=home):
                paths = libero_asset_paths(lambda kind: str(home / 'missing' / kind))
            self.assertEqual(paths['assets'], cached)
            self.assertFalse(paths['init_states'].exists())


if __name__ == '__main__':
    unittest.main()
