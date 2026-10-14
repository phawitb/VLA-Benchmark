"""Persistent, isolated LIBERO evaluation jobs for the web UI."""
from __future__ import annotations

import copy
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

ACTIVE = {'starting', 'running'}
ROOT = Path(__file__).resolve().parent


def runtime_python():
    explicit = os.environ.get('LIBERO_PYTHON')
    if explicit:
        return explicit
    for path in [Path(sys.prefix).parent / 'causalvla/bin/python',
                 Path('/opt/miniconda3/envs/causalvla/bin/python')]:
        if path.is_file():
            return str(path)
    return sys.executable


def validate_config(body, suites):
    if not isinstance(body, dict):
        raise ValueError('Invalid evaluation settings')
    model = str(body.get('model_path', '')).strip()
    if not model or len(model) > 1000:
        raise ValueError('Select a LIBERO-compatible checkpoint or Hugging Face model ID')
    suite = next((s for s in suites if s['id'] == body.get('suite')), None)
    if not suite:
        raise ValueError('Unknown LIBERO suite')
    tasks = body.get('task_ids')
    valid_tasks = {task['id'] for task in suite['tasks']}
    if not isinstance(tasks, list) or not tasks or any(type(t) is not int or t not in valid_tasks for t in tasks):
        raise ValueError('Select valid tasks from the suite')
    def integer(key, default, low, high):
        value = body.get(key, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{key} must be an integer between {low} and {high}')
        return value
    device = body.get('device', 'auto')
    if device not in {'auto', 'cpu', 'cuda', 'mps'}:
        raise ValueError('Unsupported device')
    challenge = body.get('challenge_env') or {'enabled':False}
    if not isinstance(challenge, dict):
        raise ValueError('Invalid Challenge Environment')
    if challenge.get('mode', 'random') not in {'random', 'fix'}:
        raise ValueError('Challenge mode must be Random or Fix')
    if challenge.get('enabled') and challenge.get('robot', {}).get('enabled'):
        raise ValueError('SO101 joint noise is not compatible with LIBERO actions')
    if len(json.dumps(challenge, allow_nan=False)) > 20000:
        raise ValueError('Challenge settings are too large')
    return dict(model_path=model, suite=suite['id'], task_ids=list(dict.fromkeys(tasks)),
                episodes=integer('episodes', 10, 1, 1000),
                max_steps=integer('max_steps', 300, 1, 5000),
                seed=integer('seed', 0, 0, 2147483647), device=device,
                save_video=bool(body.get('save_video', True)), challenge_env=challenge)


def local_models():
    roots = [ROOT/'models']
    extra = os.environ.get('LIBERO_MODELS_DIR')
    if extra:
        roots.append(Path(extra).expanduser())
    reference = ROOT.parent/'CausalVLA'/'outputs'/'train'
    if reference.exists():
        roots.append(reference)
    records = []
    for root in roots:
        if not root.exists():
            continue
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [s for s in subdirs if not s.startswith('.') and s not in {'wandb','videos','logs'}]
            if len(Path(directory).relative_to(root).parts) > 7:
                subdirs[:] = []
            if 'config.json' not in files:
                continue
            path=Path(directory)
            try:
                cfg=json.loads((path/'config.json').read_text())
                shape=cfg.get('output_features',{}).get('action',{}).get('shape',[])
                if not shape:
                    continue
                compatible=shape == [7]
                records.append(dict(id=str(path), path=str(path),
                    name=str(path.relative_to(root)), compatible=compatible,
                    reason='' if compatible else f'Action shape {shape}; LIBERO requires 7 actions'))
            except (OSError,ValueError,TypeError):
                continue
    return records


class LiberoService:
    def __init__(self, root, worker=None, python=None):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.worker=Path(worker or ROOT/'libero_worker.py')
        self.python=python or runtime_python()
        self.lock=threading.RLock()
        self.process=None
        self.run=None
        self.cancelled=False
        self._catalog=None
        self._catalog_time=0
        self._preview_lock=threading.Lock()
        self._preview_cache={}
        # A run cannot silently remain "running" after server shutdown.
        for path in self.root.glob('*/run.json'):
            try:
                run=json.loads(path.read_text())
                if run.get('status') in ACTIVE:
                    run.update(status='interrupted', error='Server restarted before evaluation finished')
                    self._write(run)
            except (OSError, ValueError):
                pass

    def catalog(self):
        if self._catalog and time.monotonic()-self._catalog_time < 60:
            return copy.deepcopy(self._catalog)
        try:
            result=subprocess.run([self.python,str(self.worker),'--catalog'],cwd=ROOT,
                                  capture_output=True,text=True,timeout=45,stdin=subprocess.DEVNULL)
            payload=None
            for line in reversed(result.stdout.splitlines()):
                try:
                    payload=json.loads(line.removeprefix('LIBERO_EVENT '))
                    if isinstance(payload,dict) and 'suites' in payload:
                        break
                except ValueError:
                    continue
            if not payload or result.returncode:
                raise RuntimeError((result.stderr or result.stdout or 'LIBERO runtime unavailable')[-2000:])
            payload.setdefault('runtime',{'available':True})
            payload['ok']=True
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            payload={'ok':True,'suites':[],'runtime':{'available':False,'error':str(exc)}}
        payload['models']=local_models()
        payload['runtime']['python']=self.python
        payload['defaults']={'episodes':10,'max_steps':300,'seed':0,'device':'auto'}
        self._catalog=payload
        self._catalog_time=time.monotonic()
        return copy.deepcopy(payload)

    def _write(self, run):
        directory=self.root/run['id']
        directory.mkdir(parents=True,exist_ok=True)
        path=directory/'run.json'
        temporary=directory/'run.json.tmp'
        temporary.write_text(json.dumps(run,ensure_ascii=False,indent=2))
        temporary.replace(path)

    def start(self, body):
        with self.lock:
            if self.run and self.run['status'] in ACTIVE:
                return {'ok':False,'error':'An evaluation is already running'}
        catalog=self.catalog()
        if not catalog['runtime'].get('available'):
            return {'ok':False,'error':catalog['runtime'].get('error','LIBERO is unavailable')}
        try:
            config=validate_config(body,catalog['suites'])
        except (ValueError, TypeError, AttributeError) as exc:
            return {'ok':False,'error':str(exc)}
        with self.lock:
            if self.run and self.run['status'] in ACTIVE:
                return {'ok':False,'error':'An evaluation is already running'}
            run_id=uuid.uuid4().hex
            directory=self.root/run_id
            directory.mkdir()
            (directory/'config.json').write_text(json.dumps(config))
            self.run=dict(id=run_id,status='starting',config=config,
                          created_at=datetime.now(timezone.utc).isoformat(),
                          started_at=time.time(),elapsed_seconds=0,completed=0,
                          total=len(config['task_ids'])*config['episodes'],
                          successes=0,episodes=[],logs=[],error=None)
            self.cancelled=False
            try:
                self.process=subprocess.Popen([self.python,str(self.worker),'--config',str(directory/'config.json'),
                    '--output',str(directory)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,text=True,bufsize=1,start_new_session=True)
                self.run['pid']=self.process.pid
                self._write(self.run)
                threading.Thread(target=self._read,args=(self.process,self.run),daemon=True).start()
                return {'ok':True,'run_id':run_id}
            except OSError as exc:
                self.run.update(status='failed',error=str(exc))
                self._write(self.run)
                return {'ok':False,'error':str(exc)}

    def _read(self, process, run):
        completed_event=False
        for raw in process.stdout:
            line=raw.rstrip()
            with self.lock:
                run['logs']=(run['logs']+[line])[-200:]
                if line.startswith('LIBERO_EVENT '):
                    try:
                        event=json.loads(line[len('LIBERO_EVENT '):])
                        kind=event.get('event',event.get('type'))
                        if kind in {'ready','progress'}:
                            run['status']='running'
                            for key in ['current_task','current_episode','current_step','max_steps']:
                                if key in event: run[key]=event[key]
                        elif kind=='episode':
                            episode=event.get('episode',event.get('record'))
                            if isinstance(episode,dict):
                                if episode.get('video_path'):
                                    episode['video_url']=self._media_url(run['id'],episode['video_path'])
                                run['episodes'].append(episode)
                                run['completed']=sum(not e.get('cancelled',False) for e in run['episodes'])
                                run['successes']=sum(bool(e.get('success')) for e in run['episodes'] if not e.get('cancelled'))
                        elif kind=='preview' and event.get('path'):
                            run['preview_url']=self._media_url(run['id'],event['path'])
                            run['preview_version']=time.time()
                        elif kind=='complete':
                            completed_event=True
                        elif kind=='error':
                            run['error']=str(event.get('error','Evaluation failed'))
                    except (ValueError,TypeError,KeyError):
                        run['logs'].append('Invalid worker event ignored')
                run['elapsed_seconds']=round(time.time()-run['started_at'],2)
                if line.startswith('LIBERO_EVENT '):
                    self._write(run)
        code=process.wait()
        process.stdout.close()
        with self.lock:
            run['elapsed_seconds']=round(time.time()-run['started_at'],2)
            if self.cancelled:
                run['status']='cancelled'
            elif code==0 and completed_event and run['completed']==run['total'] and not run['error']:
                run['status']='completed'
            else:
                run['status']='failed'
                run['error']=run['error'] or f'Worker exited ({code}) before all episodes completed. See logs.'
            self._write(run)

    def _media_url(self, run_id, path):
        from urllib.parse import quote
        relative=Path(path)
        if relative.is_absolute():
            relative=relative.relative_to(self.root/run_id)
        self.media_path(run_id,str(relative),must_exist=False)
        return f'/api/libero/media/{run_id}/{quote(relative.as_posix())}'

    def status(self):
        with self.lock:
            result=copy.deepcopy(self.run)
            if result and result['status'] in ACTIVE:
                result['elapsed_seconds']=round(time.time()-result['started_at'],2)
            return {'ok':True,'run':result}

    def stop(self):
        with self.lock:
            process=self.process
            if not process or process.poll() is not None:
                return {'ok':True}
            self.cancelled=True
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: return {'ok':True}
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait(timeout=5)
        return {'ok':True}

    def read_run(self, run_id):
        if not re.fullmatch('[a-f0-9]{32}',run_id):
            raise ValueError('Invalid run ID')
        return json.loads((self.root/run_id/'run.json').read_text())

    def history(self):
        runs=[]
        with self.lock:
            for path in self.root.glob('*/run.json'):
                try:
                    run=json.loads(path.read_text())
                    run.pop('logs',None)
                    runs.append(run)
                except (OSError,ValueError):
                    continue
        return sorted(runs,key=lambda r:r.get('created_at',''),reverse=True)

    def preview(self, body):
        import base64
        import cv2
        import numpy as np
        from augmentation_core import _eval_challenge_params, _aug_frame
        task_id = body.get('task_id', 0)
        validated = validate_config({**body, 'model_path': 'preview', 'task_ids': [task_id], 'episodes': 1}, self.catalog()['suites'])
        seed = (validated['seed'] + task_id * 100000) % (2**32)
        scene_config = {'suite': validated['suite'], 'task_id': task_id, 'seed': seed}
        key = (validated['suite'], task_id, seed)
        with self._preview_lock:
            scene = self._preview_cache.get(key)
            if scene is None:
                with tempfile.TemporaryDirectory(prefix='libero-preview-') as folder:
                    path = Path(folder)
                    (path/'config.json').write_text(json.dumps(scene_config))
                    proc = subprocess.run([self.python,str(self.worker),'--preview','--config',str(path/'config.json'),'--output',folder],
                                          cwd=ROOT,capture_output=True,text=True,timeout=60)
                    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith('LIBERO_PREVIEW ')), None)
                    if proc.returncode or not line:
                        error = next((json.loads(l[13:]).get('error') for l in reversed(proc.stdout.splitlines()) if l.startswith('LIBERO_EVENT ')), None)
                        raise ValueError(error or 'Unable to render LIBERO preview')
                    scene = json.loads(line[len('LIBERO_PREVIEW '):])
                if len(self._preview_cache) >= 8: self._preview_cache.pop(next(iter(self._preview_cache)))
                self._preview_cache[key] = scene
        params = _eval_challenge_params(validated['challenge_env'], seed, 20, validated['max_steps']/20) or {}
        rng = np.random.default_rng(seed)
        previews = []
        for camera in scene['images']:
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(camera['image']),dtype=np.uint8),cv2.IMREAD_COLOR)
            augmented = _aug_frame(frame,params.get('camera'),params.get('light'),cv2,np,rng=rng)
            ok, encoded = cv2.imencode('.png', augmented)
            if not ok: raise ValueError('Could not encode challenge preview')
            previews.append({'original':camera['image'],'augmented':base64.b64encode(encoded).decode(),
                             'camera':camera['camera'],'task_name':scene['task_name'],'params':params})
        return {'ok': True, 'previews': previews, 'seed': seed}

    def delete_run(self, run_id):
        with self.lock:
            run = self.read_run(run_id)
            if run.get('status') in ACTIVE or (self.run and self.run['id'] == run_id and
                    (self.run['status'] in ACTIVE or self.process and self.process.poll() is None)):
                raise ValueError('Cancel the running evaluation before deleting it.')
            trash = self.root / '.trash'
            trash.mkdir(exist_ok=True)
            (self.root / run_id).rename(trash / run_id)
            if self.run and self.run['id'] == run_id:
                self.run = None
            return {'ok': True}

    def restore_run(self, run_id):
        if not re.fullmatch('[a-f0-9]{32}', run_id):
            raise ValueError('Invalid run ID')
        with self.lock:
            source = self.root / '.trash' / run_id
            if not (source / 'run.json').is_file():
                raise FileNotFoundError('Deleted run not found')
            if (self.root / run_id).exists():
                raise ValueError('Run already exists')
            source.rename(self.root / run_id)
            return {'ok': True}

    def media_path(self,run_id,path,must_exist=True):
        if not re.fullmatch('[a-f0-9]{32}',run_id):
            raise ValueError('Invalid run ID')
        directory=(self.root/run_id).resolve()
        file=(directory/path).resolve()
        if directory not in file.parents or file.suffix.lower() not in {'.jpg','.jpeg','.png','.mp4','.webm'}:
            raise ValueError('Invalid media path')
        if must_exist and not file.is_file():
            raise FileNotFoundError('Media not found')
        return file


def make_router(service):
    import asyncio
    router=APIRouter(prefix='/api/libero')
    @router.get('/catalog')
    async def catalog():
        return await asyncio.to_thread(service.catalog)
    @router.post('/start')
    async def start(request: Request):
        return await asyncio.to_thread(service.start,await request.json())
    @router.get('/status')
    async def status():
        return service.status()
    @router.post('/preview')
    async def preview(request: Request):
        try: return await asyncio.to_thread(service.preview, await request.json())
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            return JSONResponse(status_code=400,content={'ok':False,'error':str(exc)})
    @router.post('/stop')
    async def stop():
        return await asyncio.to_thread(service.stop)
    @router.get('/history')
    async def history():
        return {'ok':True,'runs':await asyncio.to_thread(service.history)}
    @router.get('/history/{run_id}')
    async def detail(run_id: str):
        try: return {'ok':True,'run':service.read_run(run_id)}
        except (ValueError,OSError): return JSONResponse(status_code=404,content={'ok':False,'error':'Run not found'})
    @router.delete('/history/{run_id}')
    async def delete(run_id: str):
        try: return await asyncio.to_thread(service.delete_run, run_id)
        except ValueError as exc: return JSONResponse(status_code=409,content={'ok':False,'error':str(exc)})
        except OSError: return JSONResponse(status_code=404,content={'ok':False,'error':'Run not found'})
    @router.post('/history/{run_id}/restore')
    async def restore(run_id: str):
        try: return await asyncio.to_thread(service.restore_run, run_id)
        except ValueError as exc: return JSONResponse(status_code=409,content={'ok':False,'error':str(exc)})
        except OSError: return JSONResponse(status_code=404,content={'ok':False,'error':'Deleted run not found'})
    @router.get('/media/{run_id}/{path:path}')
    async def media(run_id: str,path: str):
        try: return FileResponse(service.media_path(run_id,path),headers={'Cache-Control':'no-store'})
        except (ValueError,OSError): return JSONResponse(status_code=404,content={'ok':False,'error':'Media not found'})
    return router
