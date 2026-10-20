import json
import random
import unittest

import cv2
import numpy as np
import main
import infer_python as worker


class ChallengeTests(unittest.TestCase):
    def test_dropout_uses_evaluation_cameras_exact_quota_and_repeatable_selection(self):
        config = {'enabled': True, 'camera': {'enabled': False}, 'light': {'enabled': False},
                  'camera_dropout': {'enabled': True, 'drop_camera_percent': 50,
                                     'keep_count': 1, 'mode': 'cycle'}}
        before = random.getstate()
        params = [main._eval_challenge_params(config, ep, 30, 12,
                  camera_keys=['top', 'wrist', 'side'], total_episodes=4) for ep in range(1, 5)]
        self.assertEqual([p['camera_dropout']['kept_cameras'] for p in params], [
            ['observation.images.top'], ['observation.images.top', 'observation.images.wrist', 'observation.images.side'],
            ['observation.images.wrist'], ['observation.images.top', 'observation.images.wrist', 'observation.images.side']])
        self.assertEqual(before, random.getstate())
        config['camera_dropout']['mode'] = 'random'
        self.assertEqual(main._eval_challenge_params(config, 1, 30, 12, camera_keys=['top', 'wrist']),
                         main._eval_challenge_params(config, 1, 30, 12, camera_keys=['top', 'wrist']))

    def test_dropout_worker_blacks_both_camera_key_formats_without_changing_state(self):
        original = np.full((12, 16, 3), 200, np.uint8)
        obs = {'top': original.copy(), 'observation.images.wrist': original.copy(),
               'observation.state': [1., 2.]}
        params = {'seed': 1, 'light': {'brightness': 1.2}, 'camera_dropout': {
            'kept_cameras': ['observation.images.top'], 'dropped_cameras': ['observation.images.wrist']}}
        for step in (0, 1, 10):
            worker.apply_challenge_camera_images(obs, ['top', 'wrist'], params, step, cv2, np)
            self.assertEqual(obs['observation.images.wrist'].shape, original.shape)
            self.assertEqual(obs['observation.images.wrist'].max(), 0)
            self.assertGreater(obs['top'].min(), 0)
            self.assertEqual(obs['observation.state'], [1., 2.])
        self.assertEqual(original.min(), 200)

    def test_dropout_cannot_run_with_only_one_active_camera(self):
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            main._eval_challenge_params({'enabled': True, 'camera_dropout': {'enabled': True}},
                1, 30, 12, camera_keys=['top'], total_episodes=10)

    def test_all_paths_use_one_augmentation_engine(self):
        import augmentation_core as core
        self.assertIs(main._aug_frame, core._aug_frame)
        self.assertIs(worker._aug_frame, core._aug_frame)
        for name in ('_augp_camera', '_augp_light', '_augp_robot', '_eval_challenge_params'):
            self.assertIs(getattr(main, name), getattr(core, name))

    def config(self, mode='fix'):
        return dict(enabled=True, mode=mode, intensity=3,
            camera=dict(enabled=True, perspective=True, affine=True, rotation=True),
            light=dict(enabled=True, brightness=True, contrast=True, saturation=True,
                       color_jitter=True, shadow=True, noise=True, blur=True),
            occlusion=dict(enabled=True),
            robot=dict(enabled=True, random_start=True, initial_noise=True, trajectory_jitter=True),
            fixed_params=dict(camera=dict(persp_mag=.03, tx=.02, ty=0, shear=0, scale=1, angle=2),
                light=dict(brightness=1.1, contrast=1.2, saturation=.9, hue=4, sat_off=None,
                    val_off=None, shadow_a=.2, noise_s=0, blur_s=.8, occlusion=[.2,.3,.4,.5]),
                robot=dict(trim=4, offset_sigma=1.2, jitter_s=.5)))

    def test_fixed_parameters_and_all_groups_reach_worker_payload(self):
        p = main._eval_challenge_params(self.config(), episode=7, fps=30, duration=12)
        self.assertEqual(p['camera']['angle'], 2)
        self.assertEqual(p['light']['occlusion'], [.2,.3,.4,.5])
        self.assertEqual(p['light']['sat_off'], 0)
        self.assertEqual(p['robot']['trim'], 4)
        self.assertEqual(p['robot']['joint_offset'], [1.2,-1.2,1.2,-1.2,1.2,0])
        self.assertEqual(p['seed'], 7)
        json.dumps(p)

    def test_random_is_repeatable_without_changing_global_random_state(self):
        before = random.getstate()
        np_before = np.random.get_state()
        a = main._eval_challenge_params(self.config('random'), 3, 30, 12)
        self.assertEqual(a, main._eval_challenge_params(self.config('random'), 3, 30, 12))
        self.assertNotEqual(a, main._eval_challenge_params(self.config('random'), 4, 30, 12))
        self.assertEqual(before, random.getstate())
        np.testing.assert_array_equal(np_before[1], np.random.get_state()[1])

    def test_disabled_groups_and_legacy_settings(self):
        c = self.config()
        for key in ['camera','light','robot']:
            c[key]['enabled'] = False
        p = main._eval_challenge_params(c, 1, 30, 12)
        self.assertNotIn('camera', p)
        self.assertNotIn('robot', p)
        self.assertEqual(set(p['light']), {'occlusion'})
        c['enabled'] = False
        self.assertIsNone(main._eval_challenge_params(c, 1, 30, 12))
        p = main._eval_challenge_params({'enabled':True, 'camera':{'rotation':True}}, 1,30,12)
        self.assertIn('angle',p['camera'])

    def test_worker_occlusion_matches_dataset_transform(self):
        frame = np.full((100,200,3),200,np.uint8)
        light = {'occlusion':[.1,.2,.3,.4]}
        np.testing.assert_array_equal(worker._aug_frame(frame,None,light,cv2,np),
                                      main._aug_frame(frame,None,light))

    def test_robot_perturbation_is_shared_by_state_action_and_excludes_gripper(self):
        params = dict(seed=7,robot=dict(joint_offset=[1,-1,2,-2,3,0],jitter_s=.5))
        keys=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos',
              'wrist_flex.pos','wrist_roll.pos','gripper.pos']
        deltas=worker.challenge_joint_deltas(params,keys,0)
        self.assertNotIn('gripper.pos',deltas)
        self.assertEqual(deltas,worker.challenge_joint_deltas(params,keys,0))
        self.assertNotEqual(deltas,worker.challenge_joint_deltas(params,keys,1))
        source={**dict.fromkeys(keys,10.),'top':np.zeros((2,2,3))}
        result=worker.apply_challenge_joint_deltas(source,deltas)
        self.assertEqual(result['shoulder_pan.pos'],10+deltas['shoulder_pan.pos'])
        self.assertEqual(result['gripper.pos'],10)
        self.assertEqual(source['shoulder_pan.pos'],10)

    def test_trim_skips_only_initial_commands_and_keeps_at_least_one(self):
        c=self.config()
        c['fixed_params']['robot']['trim']=30
        p=main._eval_challenge_params(c,1,10,1)
        self.assertEqual(p['robot']['trim'],9)
        self.assertFalse(worker.challenge_should_send_action(p,8))
        self.assertTrue(worker.challenge_should_send_action(p,9))
        self.assertTrue(worker.challenge_should_send_action(None,0))

    def test_random_trim_also_works_for_short_live_episodes(self):
        p=main._eval_challenge_params(self.config('random'),1,10,1)
        self.assertGreater(p['robot'].get('trim',0),0)
        self.assertLess(p['robot']['trim'],10)
