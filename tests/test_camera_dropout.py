import random
import unittest

import augmentation_core


class CameraDropoutTests(unittest.TestCase):
    def params(self, keys, opts, episode=0, **kwargs):
        return augmentation_core._camera_dropout_params(keys, opts, episode, **kwargs)

    def test_disabled_keeps_existing_behavior(self):
        self.assertIsNone(self.params(['top'], {'enabled': False}))

    def test_four_views_rotate_without_blacking_out_every_camera(self):
        keys = ['top', 'wrist', 'side', 'front']
        opts = {'enabled': True, 'keep_count': 2}
        expected = [['top', 'wrist'], ['wrist', 'side'], ['side', 'front'], ['front', 'top']]
        for index, kept in enumerate(expected):
            result = self.params(keys, opts, index)
            self.assertEqual(result['kept_cameras'], kept)
            self.assertEqual(set(result['dropped_cameras']), set(keys) - set(kept))
        self.assertEqual(self.params(keys, opts, 4)['kept_cameras'], expected[0])

    def test_selected_pool_can_exclude_views_and_random_mode_is_reproducible(self):
        keys = ['top', 'wrist', 'side', 'front']
        opts = {'enabled': True, 'cameras': ['wrist', 'front'], 'keep_count': 1}
        self.assertEqual(self.params(keys, opts, 1), {
            'kept_cameras': ['front'], 'dropped_cameras': ['top', 'wrist', 'side']})
        opts['mode'] = 'random'
        rng = random.Random(17)
        results = [self.params(keys, opts, i, rng=rng) for i in range(12)]
        again = random.Random(17)
        self.assertEqual(results, [self.params(keys, opts, i, rng=again) for i in range(12)])
        self.assertEqual({r['kept_cameras'][0] for r in results}, {'wrist', 'front'})

    def test_invalid_selection_fails_instead_of_producing_all_black_data(self):
        for bad in [{'cameras': []}, {'cameras': ['missing']}, {'keep_count': 0},
                    {'keep_count': 3}, {'keep_count': 1.5}, {'mode': 'unknown'}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.params(['top', 'wrist'], {'enabled': True, **bad})

    def test_half_full_views_keeps_exact_ratio_and_rotates_only_dropout_episodes(self):
        opts = {'enabled': True, 'full_camera_percent': 50}
        result = [self.params(['top', 'wrist'], opts, i, total_episodes=8) for i in range(8)]
        self.assertEqual([r['kept_cameras'] for r in result], [
            ['top'], ['top', 'wrist'], ['wrist'], ['top', 'wrist'],
            ['top'], ['top', 'wrist'], ['wrist'], ['top', 'wrist']])

    def test_full_view_percentage_rounding_and_endpoints(self):
        for percent, total, full in [(0, 5, 0), (100, 5, 5), (50, 3, 2), (30, 10, 3)]:
            with self.subTest(percent=percent, total=total):
                results = [self.params(['top', 'wrist'], {'enabled': True,
                    'full_camera_percent': percent}, i, total_episodes=total) for i in range(total)]
                self.assertEqual(sum(not r['dropped_cameras'] for r in results), full)
        for percent in [-1, 101, 'invalid', None]:
            with self.subTest(percent=percent), self.assertRaises(ValueError):
                self.params(['top', 'wrist'], {'enabled': True, 'full_camera_percent': percent})

    def test_drop_percentage_counts_dropped_episodes_not_full_camera_episodes(self):
        for percent, total, dropped in [(0, 5, 0), (100, 5, 5), (30, 10, 3), (50, 3, 2)]:
            with self.subTest(percent=percent):
                results = [self.params(['top', 'wrist'], {'enabled': True,
                    'drop_camera_percent': percent}, i, total_episodes=total) for i in range(total)]
                self.assertEqual(sum(bool(r['dropped_cameras']) for r in results), dropped)
        for percent in [-1, 101, None]:
            with self.subTest(percent=percent), self.assertRaises(ValueError):
                self.params(['top', 'wrist'], {'enabled': True, 'drop_camera_percent': percent})
