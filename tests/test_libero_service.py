import json
import tempfile
import time
import unittest
from pathlib import Path
from libero_service import LiberoService, validate_config


class LiberoServiceTests(unittest.TestCase):
    def test_preview_caches_scene_but_reapplies_current_challenge(self):
        import base64
        import cv2
        import numpy as np
        from unittest.mock import patch
        from types import SimpleNamespace
        encoded = base64.b64encode(cv2.imencode('.png', np.full((16,16,3),200,np.uint8))[1]).decode()
        scene = {'images':[{'camera':'image','image':encoded},{'camera':'wrist_image','image':encoded}], 'task_name':'test scene'}
        with tempfile.TemporaryDirectory() as tmp:
            service=LiberoService(Path(tmp))
            service.catalog=lambda:{'suites':[{'id':'libero_spatial','tasks':[{'id':0}]}]}
            body={'suite':'libero_spatial','task_id':0,'seed':0,'challenge_env':{'enabled':False}}
            with patch('libero_service.subprocess.run', return_value=SimpleNamespace(returncode=0,stdout='LIBERO_PREVIEW '+json.dumps(scene))) as launch:
                original=service.preview(body)
                self.assertEqual(original['previews'][0]['original'],original['previews'][0]['augmented'])
                body['challenge_env']={'enabled':True,'mode':'fix','light':{'brightness':True},'fixed_params':{'light':{'brightness':.5}}}
                changed=service.preview(body)
                self.assertEqual(launch.call_count,1)
                self.assertEqual(len(changed['previews']),2)
                for camera in changed['previews']:
                    image=cv2.imdecode(np.frombuffer(base64.b64decode(camera['augmented']),np.uint8),cv2.IMREAD_COLOR)
                    self.assertTrue((image==100).all())
                body['task_id']=999
                with self.assertRaises(ValueError): service.preview(body)

    def test_delete_restore_preserves_media_and_rejects_active_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = LiberoService(Path(tmp))
            run_id = 'b' * 32
            directory = Path(tmp) / run_id
            directory.mkdir()
            run = {'id': run_id, 'status': 'completed'}
            (directory / 'run.json').write_text(json.dumps(run))
            (directory / 'preview.jpg').write_bytes(b'video-data')
            service.run = run
            service.delete_run(run_id)
            self.assertEqual(service.history(), [])
            self.assertIsNone(service.run)
            with self.assertRaises(FileNotFoundError): service.media_path(run_id, 'preview.jpg')
            service.restore_run(run_id)
            self.assertEqual(service.read_run(run_id), run)
            self.assertEqual((directory / 'preview.jpg').read_bytes(), b'video-data')
            run['status'] = 'running'
            (directory / 'run.json').write_text(json.dumps(run))
            with self.assertRaises(ValueError): service.delete_run(run_id)
            with self.assertRaises(ValueError): service.restore_run('../outside')

    def test_validation_rejects_invalid_tasks_and_limits(self):
        suites = [{'id':'libero_spatial','tasks':[{'id':0}]}]
        config = {'model_path':'user/model','suite':'libero_spatial','task_ids':[0],'episodes':1}
        self.assertEqual(validate_config(config,suites)['task_ids'],[0])
        for changes in [{'task_ids':[99]},{'episodes':0},{'device':'shell'},{'max_steps':100001}]:
            with self.assertRaises(ValueError):
                validate_config(dict(config,**changes),suites)

    def test_success_and_failure_are_persisted_from_real_subprocess(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            script=root/'worker.py'
            script.write_text('import json\nprint(\'LIBERO_EVENT \'+json.dumps({"event":"episode","episode":{"task_id":0,"episode":1,"success":True,"steps":3}}),flush=True)\nprint(\'LIBERO_EVENT \'+json.dumps({"event":"complete"}),flush=True)\n')
            service=LiberoService(root/'runs',worker=script,python=sys.executable)
            service.catalog=lambda: {'ok':True,'suites':[{'id':'libero_spatial','tasks':[{'id':0}]}],'runtime':{'available':True}}
            result=service.start({'model_path':'user/model','suite':'libero_spatial','task_ids':[0],'episodes':1})
            self.assertTrue(result['ok'])
            deadline=time.monotonic()+5
            while service.status()['run']['status'] in ('starting','running') and time.monotonic()<deadline:
                time.sleep(.02)
            run=service.status()['run']
            self.assertEqual(run['status'],'completed')
            self.assertEqual(run['completed'],1)
            self.assertEqual(run['successes'],1)
            self.assertEqual(service.history()[0]['id'],run['id'])
            self.assertEqual(service.read_run(run['id'])['episodes'][0]['steps'],3)
            script.write_text('raise RuntimeError("failed simulator")\n')
            self.assertTrue(service.start({'model_path':'user/model','suite':'libero_spatial','task_ids':[0],'episodes':1})['ok'])
            deadline=time.monotonic()+5
            while service.status()['run']['status'] in ('starting','running') and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertEqual(service.status()['run']['status'],'failed')

    def test_history_paths_are_confined(self):
        with tempfile.TemporaryDirectory() as tmp:
            service=LiberoService(Path(tmp))
            with self.assertRaises(ValueError): service.read_run('../config')
            with self.assertRaises(ValueError): service.media_path('a'*32,'../config.yaml')

    def test_router_start_json_and_history(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from libero_service import make_router
        from unittest.mock import Mock
        service=Mock()
        service.start.return_value={'ok':True,'run_id':'a'*32}
        service.history.return_value=[]
        app=FastAPI()
        app.include_router(make_router(service))
        client=TestClient(app)
        response=client.post('/api/libero/start',json={'model_path':'user/model'})
        self.assertEqual(response.status_code,200)
        service.start.assert_called_once_with({'model_path':'user/model'})
        self.assertEqual(client.get('/api/libero/history').json()['runs'],[])

    def test_cancel_keeps_partial_history_and_releases_process(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            script=root/'worker.py'
            script.write_text('import time\nprint("ready",flush=True)\ntime.sleep(60)\n')
            service=LiberoService(root/'runs',worker=script,python=sys.executable)
            service.catalog=lambda: {'ok':True,'suites':[{'id':'libero_spatial','tasks':[{'id':0}]}],'runtime':{'available':True}}
            body={'model_path':'user/model','suite':'libero_spatial','task_ids':[0],'episodes':1}
            self.assertTrue(service.start(body)['ok'])
            self.assertFalse(service.start(body)['ok'])
            self.assertTrue(service.stop()['ok'])
            deadline=time.monotonic()+3
            while service.status()['run']['status'] in ('starting','running') and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertEqual(service.status()['run']['status'],'cancelled')
            self.assertIsNotNone(service.process.poll())
            self.assertEqual(service.history()[0]['completed'],0)
