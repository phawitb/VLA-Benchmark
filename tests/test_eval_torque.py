import unittest
from unittest.mock import Mock, call, patch
from fastapi.testclient import TestClient
import main


class EvalTorqueTests(unittest.TestCase):
    def setUp(self):
        self.bus = Mock()
        self.bus.sync_read.side_effect = lambda name, **kw: (
            {'joint': 2048} if name == 'Present_Position' else {'joint': self.enabled})
        self.enabled = 1
        self.robot = Mock(bus=self.bus)
        for name, value in [('robot_state', {'connected': True, 'robot': self.robot}),
                            ('robot_operation_owner', None)]:
            p = patch.object(main, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app)

    def test_on_holds_current_position_before_enabling(self):
        response = self.client.post('/api/eval/torque', json={'enabled': True})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['ok'])
        self.assertEqual(self.bus.method_calls[:3], [
            call.sync_read('Present_Position', normalize=False),
            call.sync_write('Goal_Position', {'joint': 2048}, normalize=False),
            call.enable_torque()])
        self.bus.disconnect.assert_not_called()
        self.assertIsNone(main.robot_operation_owner)

    def test_off_does_not_write_positions(self):
        self.enabled = 0
        result = self.client.post('/api/eval/torque', json={'enabled': False}).json()
        self.assertTrue(result['ok'])
        self.bus.disable_torque.assert_called_once()
        self.bus.sync_write.assert_not_called()

    def test_busy_never_touches_hardware(self):
        main.robot_operation_owner = 'eval'
        response = self.client.post('/api/eval/torque', json={'enabled': False})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.bus.method_calls, [])

    def test_invalid_request_never_touches_hardware(self):
        response = self.client.post('/api/eval/torque', json={'enabled': 'false'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.bus.method_calls, [])

    def test_failure_releases_ownership(self):
        self.bus.enable_torque.side_effect = RuntimeError('Serial failure')
        result = self.client.post('/api/eval/torque', json={'enabled': True}).json()
        self.assertFalse(result['ok'])
        self.assertIsNone(main.robot_operation_owner)

    def test_temporary_connection_preserves_requested_torque_on_close(self):
        main.robot_state['connected'] = False
        with patch.object(main, '_create_robot', return_value=self.robot):
            result = self.client.post('/api/eval/torque', json={'enabled': True}).json()
        self.assertTrue(result['ok'])
        self.bus.connect.assert_called_once()
        self.bus.disconnect.assert_called_once_with(disable_torque=False)
        self.robot.connect.assert_not_called()

    def test_readback_mismatch_is_reported(self):
        self.enabled = 0
        result = self.client.post('/api/eval/torque', json={'enabled': True}).json()
        self.assertFalse(result['ok'])
        self.assertIn('confirm', result['error'])

    def test_capture_does_not_enable_torque_and_closes_temporary_bus(self):
        import tempfile
        from pathlib import Path
        main.robot_state['connected'] = False
        positions = {joint: 12.5 for joint in main.ROBOT_JOINTS}
        self.bus.sync_read.side_effect = None
        self.bus.sync_read.return_value = positions
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(main, '_create_robot', return_value=self.robot), \
             patch.object(main, 'EVAL_RESET_POS_FILE', Path(directory) / 'reset.json'):
            result = self.client.post('/api/eval/reset-position', json={}).json()
        self.assertEqual(result['position'], positions)
        self.bus.enable_torque.assert_not_called()
        self.bus.disable_torque.assert_not_called()
        self.bus.sync_write.assert_not_called()
        self.bus.disconnect.assert_called_once_with(disable_torque=False)
        self.assertIsNone(main.robot_operation_owner)
