import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import main
import infer_python as worker


class EvalRestartTests(unittest.TestCase):
    def test_dropout_start_returns_parameters_for_target_runs_and_actual_cameras(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'models/act_test').mkdir(parents=True)
            record = dict(id='act_test', selectable=True, local_path='models/act_test', policy_type='act', camera_features=[])
            manager = Mock()
            manager.ensure_model.return_value = dict(ok=True, reused=True, pid=123)
            manager.run_task.return_value = dict(ok=True, pid=123)
            challenge = {'enabled': True, 'camera': {'enabled': False}, 'light': {'enabled': False},
                         'camera_dropout': {'enabled': True, 'drop_camera_percent': 50}}
            config = {'robot': {'cameras': {'top': {'index': 0}, 'wrist': {'index': 1}, 'side': {'index': 2}}}}
            with patch.multiple(main, ROOT=root, vla_manager=manager,
                    robot_state={'connected': False}, teleop_state={'running': False}, eval_state={'running': False}), \
                 patch.object(main, 'load_config', return_value=config), \
                 patch.object(main, 'load_local_models', return_value=[record]), \
                 patch.object(main, '_eval_load_reset_position', return_value=None):
                result = asyncio.run(main._eval_start_owned(dict(model_id='act_test', task='pick',
                    episode=3, target_runs=4, challenge_env=challenge)))
            self.assertTrue(result['ok'], result)
            self.assertEqual(result['augment_params']['camera_dropout']['kept_cameras'], ['observation.images.wrist'])
            self.assertEqual(result['augment_params']['camera_dropout']['dropped_cameras'],
                             ['observation.images.top', 'observation.images.side'])
            self.assertEqual(manager.run_task.call_args.args[5], result['augment_params'])

    def test_episode_uses_local_registry_and_worker_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'models' / 'act_test').mkdir(parents=True)
            record = dict(id='act_test', selectable=True, local_path='models/act_test',
                          policy_type='act', camera_features=[])
            manager = Mock()
            manager.ensure_model.return_value = dict(ok=True, reused=True, pid=123)
            manager.run_task.return_value = dict(ok=True, pid=123)
            reset = {'shoulder_pan': 10}
            with patch.multiple(main, ROOT=root, vla_manager=manager,
                                robot_state={'connected':False}, teleop_state={'running':False},
                                eval_state={'running':False}), \
                 patch.object(main, 'load_config', return_value={'robot':{'cameras':{}}}), \
                 patch.object(main, 'load_local_models', return_value=[record]) as local, \
                 patch.object(main, '_load_model_registry', side_effect=AssertionError('Network registry on episode path')), \
                 patch.object(main, '_eval_load_reset_position', return_value=reset), \
                 patch.object(main, '_eval_move_to_reset', side_effect=AssertionError('Second hardware connection')):
                result = asyncio.run(main._eval_start_owned(dict(model_id='user/act_test',task='pick')))
            self.assertTrue(result['ok'], result)
            self.assertTrue(result['reused_model'])
            local.assert_called_once()
            self.assertEqual(manager.run_task.call_args.kwargs['reset_position'], reset)

    def test_worker_reset_uses_existing_connection_and_no_camera_reads(self):
        robot = Mock()
        robot.bus.sync_read.return_value = {'shoulder_pan':0, 'gripper':5}
        args = SimpleNamespace(_reset_position={'shoulder_pan':10})
        with patch.object(worker.time, 'sleep'):
            worker.move_to_episode_reset(robot, args)
        robot.connect.assert_not_called()
        robot.disconnect.assert_not_called()
        robot.get_observation.assert_not_called()
        self.assertEqual(robot.send_action.call_count,30)
        self.assertEqual(robot.send_action.call_args.args[0], {'shoulder_pan.pos':10, 'gripper.pos':5})

    def test_no_reset_has_no_hardware_cost(self):
        robot = Mock()
        worker.move_to_episode_reset(robot, SimpleNamespace(_reset_position=None))
        self.assertEqual(robot.mock_calls, [])

    def test_episode_clears_act_action_queue_via_policy_reset(self):
        from collections import deque
        class Policy:
            def __init__(self):
                self._action_queue = deque([1, 2, 3])
            def reset(self):
                self._action_queue.clear()
        policy = Policy()
        cache = {'tokens': 'old'}
        worker.reset_policy_task_state(policy, cache)
        self.assertEqual(list(policy._action_queue), [])
        self.assertIsNone(cache['tokens'])

    def test_manager_reuses_model_and_forwards_reset_without_restart(self):
        from vla_execution import VlaProcessManager
        manager = VlaProcessManager()
        manager._process = Mock(pid=123)
        manager._process.poll.return_value = None
        manager._model_ready = True
        manager._model_id = 'user/model'
        manager._command = ['python', 'worker.py']
        manager._hardware_connected = True
        with patch.object(manager, 'start') as start:
            result = manager.ensure_model(['python', 'worker.py'], 'user/model')
        self.assertTrue(result['reused'])
        start.assert_not_called()
        with patch.object(manager, '_send_command', return_value={'ok':True}) as send, \
             patch.object(manager, '_wait_for', return_value={'ok':True,'pid':123}):
            manager.run_task('pick', 30, 'frame.jpg', reset_position={'joint':2})
        self.assertEqual(send.call_args.args[0]['reset_position'], {'joint':2})
