"""Verify every exposed challenge changes actual policy tensors, not only metadata."""
import importlib.util
import unittest

from libero_worker import ImageChallenge


@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ('cv2', 'numpy', 'torch')), 'Requires policy runtime')
class ChallengeEffectTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        import torch
        self.torch = torch
        pixels = np.random.default_rng(42).integers(30, 210, (2, 3, 64, 64), dtype=np.uint8)
        self.image = torch.from_numpy(pixels).float() / 255
        self.obs = {'observation.images.image': self.image,
                    'observation.images.wrist_image': self.image.clone(),
                    'observation.state': torch.zeros(2, 8)}

    def test_each_technique_changes_both_cameras_random_and_fixed(self):
        techniques = {'camera': {'perspective': {'persp_mag': .08},
                                  'affine': {'tx': .05, 'ty': .04, 'shear': .03, 'scale': 1.05},
                                  'rotation': {'angle': 7}},
                      'light': {'brightness': {'brightness': .5}, 'contrast': {'contrast': .5},
                                'saturation': {'saturation': .3},
                                'color_jitter': {'hue': 15, 'sat_off': 20, 'val_off': -20},
                                'shadow': {'shadow_a': .7, 'shadow_dir': 'left'},
                                'noise': {'noise_s': 20}, 'blur': {'blur_s': 2}},
                      'occlusion': {'occlusion': {'occlusion': [.2, .2, .3, .3]}}}
        for mode in ('random', 'fix'):
            for group, options in techniques.items():
                for technique, fixed in options.items():
                    with self.subTest(mode=mode, technique=technique):
                        cfg = {'enabled': True, 'mode': mode, 'intensity': 5,
                               group: {'enabled': True, technique: True},
                               'fixed_params': {group if group != 'occlusion' else 'light': fixed}}
                        challenge = ImageChallenge(cfg, 42, 20)
                        result = challenge.apply(self.obs)
                        for camera in ('image', 'wrist_image'):
                            key = 'observation.images.' + camera
                            self.assertGreater((result[key] - self.obs[key]).abs().mean().item(), .001)
                            self.assertEqual(result[key].shape, self.obs[key].shape)
                            self.assertTrue(self.torch.isfinite(result[key]).all())
                            self.assertGreaterEqual(result[key].min().item(), 0)
                            self.assertLessEqual(result[key].max().item(), 1)
                        self.assertIs(result['observation.state'], self.obs['observation.state'])
                        repeat = ImageChallenge(cfg, 42, 20).apply(self.obs)
                        self.assertTrue(self.torch.equal(result['observation.images.image'], repeat['observation.images.image']))
                        cfg[group]['enabled'] = False
                        self.assertIs(ImageChallenge(cfg, 42, 20).apply(self.obs), self.obs)

    def test_brightness_exact_and_master_off_is_unchanged(self):
        cfg = {'enabled': True, 'mode': 'fix', 'light': {'enabled': True, 'brightness': True},
               'fixed_params': {'light': {'brightness': .5}}}
        original = self.image.clone()
        changed = ImageChallenge(cfg, 0, 20).apply(self.obs)['observation.images.image']
        self.assertTrue(self.torch.allclose(changed, original * .5, atol=1/255))
        self.assertTrue(self.torch.equal(self.image, original))
        cfg['enabled'] = False
        self.assertIs(ImageChallenge(cfg, 0, 20).apply(self.obs), self.obs)
