#!/usr/bin/env python3
"""Isolated LIBERO runner. All simulator/policy imports happen inside this process.

stdout protocol: LIBERO_EVENT <JSON>. result.json is atomically checkpointed after
all completed episodes and on errors/cancellation. Success is simulator-reported,
never inferred from rewards or the mere absence of an exception.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

_STOP = False


def emit(event, **fields):
    print('LIBERO_EVENT ' + json.dumps({'event': event, **fields}, allow_nan=False), flush=True)


def stop_requested(*_):
    global _STOP
    _STOP = True


def setup_policy_dependencies():
    """Expose CausalVLA's companion package from the active LeRobot checkout."""
    if importlib.util.find_spec('causal_aug') is not None:
        return
    spec = importlib.util.find_spec('lerobot')
    if spec is None or not spec.submodule_search_locations:
        return
    for location in spec.submodule_search_locations:
        # CausalVLA/lerobot/src/lerobot and CausalVLA/causal_aug/causal_aug.
        package = Path(location).resolve().parents[2] / 'causal_aug'
        if (package / 'causal_aug' / '__init__.py').is_file():
            sys.path.insert(0, str(package))
            return


def bootstrap(directory):
    """Use a process-private LIBERO config; never prompt or rewrite ~/.libero."""
    source = os.environ.get('LIBERO_LEROBOT_SRC')
    if source:
        sys.path.insert(0, str(Path(source).expanduser().resolve()))
    setup_policy_dependencies()
    spec = importlib.util.find_spec('libero')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('LIBERO is not installed in this Python runtime. Set LIBERO_PYTHON to a LeRobot + LIBERO environment.')
    root = Path(next(iter(spec.submodule_search_locations))) / 'libero'
    config = Path(directory) / 'libero_config'
    config.mkdir(parents=True, exist_ok=True)
    supplied = Path(os.environ.get('LIBERO_CONFIG_PATH', str(Path.home() / '.libero'))) / 'config.yaml'
    if supplied.is_file():
        content = supplied.read_text()
    else:
        content = json.dumps({
            'benchmark_root': str(root), 'bddl_files': str(root / 'bddl_files'),
            'init_states': str(root / 'init_files'), 'datasets': str(root.parent / 'datasets'),
            'assets': str(root / 'assets'),
        })  # JSON is valid YAML.
    (config / 'config.yaml').write_text(content)
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '15')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    if sys.platform != 'darwin':
        os.environ.setdefault('MUJOCO_GL', 'egl')


def libero_asset_paths(get_path):
    paths = {kind: Path(get_path(kind)) for kind in ('assets', 'bddl_files', 'init_states')}
    cached = Path.home() / '.cache' / 'libero' / 'assets'
    if not paths['assets'].is_dir() and cached.is_dir():
        paths['assets'] = cached
    return paths


def catalog():
    from libero.libero import benchmark, get_libero_path
    missing = [name for name in ('lerobot', 'torch', 'cv2', 'av', 'gymnasium', 'mujoco', 'robosuite')
               if importlib.util.find_spec(name) is None]
    asset_errors = [kind for kind, path in libero_asset_paths(get_libero_path).items() if not path.is_dir()]
    suites = []
    unavailable = []
    for name, constructor in sorted(benchmark.get_benchmark_dict().items()):
        try:
            suite = constructor()
        except Exception as exc:
            unavailable.append({'id': name, 'error': str(exc)})
            continue
        suites.append({'id': name, 'name': name.replace('_', ' ').upper(), 'tasks': [
            {'id': i, 'name': suite.get_task(i).name, 'description': suite.get_task(i).language}
            for i in range(suite.n_tasks)
        ]})
    return {'ok': True, 'suites': suites, 'unavailable_suites': unavailable, 'runtime': {'available': not (missing or asset_errors), 'python': sys.executable,
        'error': ('Missing dependencies: ' + ', '.join(missing) if missing else '') +
                 (' Missing LIBERO asset directories: ' + ', '.join(asset_errors) if asset_errors else '')}}


def validate_config(config):
    if not config.get('model_path'):
        raise ValueError('A LIBERO-compatible model checkpoint is required.')
    for key in ('episodes', 'max_steps'):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config.get('device', 'auto') not in ('auto', 'cpu', 'cuda', 'mps'):
        raise ValueError('device must be auto, cpu, cuda, or mps')
    challenge = config.get('challenge_env') or {}
    robot = challenge.get('robot') or {}
    if challenge.get('enabled') and robot.get('enabled') and any(robot.get(k) for k in ('initial_noise', 'trajectory_jitter', 'random_start')):
        raise ValueError('Robot joint noise and recording trim use SO101 units and are not supported in LIBERO. Use camera, lighting, and occlusion image challenges.')


def validate_action_features(features):
    action = features.get('action')
    shape = getattr(action, 'shape', None) if not isinstance(action, dict) else action.get('shape')
    if not shape or tuple(shape) != (7,):
        raise ValueError(f'LIBERO requires a 7-dimensional Cartesian/gripper action checkpoint; got {shape}. SO101 6-joint checkpoints are incompatible.')


def save_result(output, config, episodes, status, error=None):
    successes = sum(bool(ep['success']) for ep in episodes if not ep.get('cancelled'))
    completed = sum(not ep.get('cancelled', False) for ep in episodes)
    result = {'config': config, 'episodes': episodes, 'status': status,
              'completed': completed, 'successes': successes,
              'total': len(config.get('task_ids') or []) * config.get('episodes', 0),
              'success_rate': successes / completed if completed else None,
              'challenge_scope': 'Policy input image transformations; simulator camera and lighting physics unchanged.'}
    if error:
        result['error'] = str(error)
    temp = output / 'result.json.tmp'
    temp.write_text(json.dumps(result, indent=2, allow_nan=False))
    temp.replace(output / 'result.json')
    return result


def successful(info):
    """Handle both direct Gym environments and vector terminal info envelopes."""
    if 'is_success' in info:
        value = info['is_success']
        return bool(value.item() if hasattr(value, 'item') else value)
    final = info.get('final_info')
    if isinstance(final, dict):
        return successful(final)
    if final is not None and len(final):
        return successful(final[0] or {})
    return False


def run_episode(env, infer, *, seed, max_steps, progress, capture=None, stopped=lambda: _STOP):
    """Small dependency-injected rollout core, also tested without a simulator."""
    observation, _ = env.reset(seed=seed)
    success = False
    steps = 0
    started = time.monotonic()
    for index in range(max_steps):
        if stopped():
            break
        action = infer(observation, index)
        if stopped():
            break
        observation, _, terminated, truncated, info = env.step(action)
        steps = index + 1
        success = success or successful(info)
        if capture:
            capture(observation, index)
        progress(steps)
        if success or bool(terminated) or bool(truncated):
            break
    return {'success': success, 'steps': steps, 'duration_seconds': time.monotonic() - started,
            'cancelled': bool(stopped())}


class ImageChallenge:
    def __init__(self, config, seed, max_steps):
        import numpy as np
        from augmentation_core import _eval_challenge_params
        self.params = _eval_challenge_params(config, seed, 20, max_steps / 20) or {}
        self.rng = np.random.default_rng(seed)

    def apply(self, observation):
        if not (self.params.get('camera') or self.params.get('light')):
            return observation
        import cv2
        import numpy as np
        import torch
        from augmentation_core import _aug_frame
        result = dict(observation)
        for key, tensor in observation.items():
            if key.startswith('observation.images.'):
                # Environment preprocessing has already oriented LIBERO cameras.
                frames = []
                for item in tensor:
                    rgb = (item.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    transformed = _aug_frame(bgr, self.params.get('camera'), self.params.get('light'), cv2, np, rng=self.rng)
                    rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
                    frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255)
                result[key] = torch.stack(frames).to(tensor.device)
        return result


def configure_evaluation_checkpoint(policy_cfg, model_path, device):
    """Select the evaluation repo, discarding the training base model's revision."""
    policy_cfg.pretrained_path = model_path
    # Saved training configs can pin a commit belonging to a different base repo.
    # The config above was loaded from the selected repo's default revision.
    policy_cfg.pretrained_revision = None
    policy_cfg.device = device


def run(config, output):
    import cv2  # before Torch, avoiding duplicate OpenMP initialization on macOS
    import numpy as np
    import torch
    from libero.libero import benchmark, get_libero_path
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as EnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.random_utils import set_seed
    from libero_video import EpisodeVideoWriter

    validate_config(config)
    seed = int(config.get('seed', 0))
    set_seed(seed)
    choices = benchmark.get_benchmark_dict()
    if config['suite'] not in choices:
        raise ValueError(f"Unknown LIBERO suite: {config['suite']}")
    suite = choices[config['suite']]()
    ids = config.get('task_ids')
    if ids is None:
        ids = list(range(suite.n_tasks))
    if not ids or any(type(i) is not int or not 0 <= i < suite.n_tasks for i in ids):
        raise ValueError('Select valid task IDs from the installed suite catalog.')
    config['task_ids'] = list(dict.fromkeys(ids))
    paths = libero_asset_paths(get_libero_path)
    if str(paths['assets']) != get_libero_path('assets'):
        import libero.libero as libero_module
        libero_module._assets_path_cache = str(paths['assets'])
    for kind, path in paths.items():
        if not path.is_dir():
            raise RuntimeError(f'LIBERO {kind} are missing: {path}. Install assets before evaluation.')
    device = config.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available() or device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError(f'Requested device {device} is unavailable in this runtime.')
    policy_cfg = PreTrainedConfig.from_pretrained(config['model_path'])
    validate_action_features(policy_cfg.output_features)  # factory otherwise replaces this with env features
    configure_evaluation_checkpoint(policy_cfg, config['model_path'], device)
    image_keys = [k for k in policy_cfg.input_features if k.startswith('observation.images.')]
    wrist_key = next((k for k in image_keys if k.endswith(('.wrist_image', '.image2', '.wrist'))), None)
    main_key = next((k for k in image_keys if k != wrist_key), None)
    if main_key is None or len(image_keys) > 2:
        raise ValueError(f'Checkpoint camera inputs are incompatible with the two LIBERO cameras: {image_keys}')
    camera_mapping = {'agentview_image': main_key.removeprefix('observation.images.'),
                      'robot0_eye_in_hand_image': wrist_key.removeprefix('observation.images.') if wrist_key else 'image2'}
    env_cfg = EnvConfig(task=config['suite'], task_ids=config['task_ids'], episode_length=config['max_steps'],
                        camera_name_mapping=camera_mapping, observation_height=256, observation_width=256)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=config['model_path'],
        preprocessor_overrides={'device_processor': {'device': device}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    episodes = []
    total = len(config['task_ids']) * config['episodes']
    emit('ready', total=total, completed=0, successes=0, device=device)
    save_result(output, config, episodes, 'running')
    try:
        for task_id in config['task_ids']:
            if _STOP:
                break
            task = suite.get_task(task_id)
            env = LiberoEnv(task_suite=suite, task_id=task_id, task_suite_name=config['suite'],
                episode_length=config['max_steps'], obs_type='pixels_agent_pos',
                observation_height=env_cfg.observation_height, observation_width=env_cfg.observation_width,
                camera_name_mapping=camera_mapping)
            try:
                for episode in range(config['episodes']):
                    if _STOP:
                        break
                    episode_seed = (seed + task_id * 100000 + episode) % (2**32)
                    challenge = ImageChallenge(config.get('challenge_env'), episode_seed, config['max_steps'])
                    for processor in (policy, pre, post, env_pre, env_post):
                        if hasattr(processor, 'reset'):
                            processor.reset()
                    writer = None
                    video_path = None
                    base = {'current_task': task_id, 'current_episode': episode + 1,
                            'max_steps': config['max_steps'], 'total': total, 'completed': len(episodes),
                            'successes': sum(ep['success'] for ep in episodes)}
                    emit('progress', current_step=0, **base)

                    def infer(raw, step):
                        nonlocal writer, video_path
                        def batch(value):
                            return {k: batch(v) for k, v in value.items()} if isinstance(value, dict) else np.expand_dims(value, 0)
                        observation = preprocess_observation(batch(raw))
                        observation['task'] = [task.language]
                        observation = challenge.apply(env_pre(observation))
                        image = next((v for k, v in observation.items() if k.startswith('observation.images.')), None)
                        if image is not None:
                            rgb = (image[0].detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            if config.get('save_video'):
                                if writer is None:
                                    video_path = f'videos/task_{task_id}_episode_{episode + 1}.mp4'
                                    (output / 'videos').mkdir(exist_ok=True)
                                    writer = EpisodeVideoWriter(output / video_path, bgr.shape[1], bgr.shape[0])
                                writer.write(bgr)
                            if step % 10 == 0:
                                temp = output / 'preview.tmp.jpg'
                                if not cv2.imwrite(str(temp), bgr):
                                    raise RuntimeError('Could not write preview JPEG.')
                                temp.replace(output / 'preview.jpg')
                                emit('preview', path='preview.jpg', current_step=step, **base)
                        amp = torch.autocast(device_type=device) if policy_cfg.use_amp else nullcontext()
                        with torch.inference_mode(), amp:
                            action = policy.select_action(pre(observation))
                            action = env_post({'action': post(action)})['action']
                        array = action.detach().cpu().numpy()
                        if array.shape != (1, 7) or not np.isfinite(array).all():
                            raise ValueError(f'Policy produced invalid LIBERO action: shape {array.shape}; expected finite (1, 7).')
                        return array[0]

                    try:
                        record = run_episode(env, infer, seed=episode_seed, max_steps=config['max_steps'],
                            progress=lambda step: emit('progress', current_step=step, **base))
                    finally:
                        if writer is not None:
                            writer.release()
                    record.update(task_id=task_id, task_name=task.name, episode=episode + 1,
                                  seed=episode_seed, video_path=video_path, challenge_params=challenge.params)
                    episodes.append(record)
                    result = save_result(output, config, episodes, 'cancelled' if _STOP else 'running')
                    emit('episode', episode=record, completed=result['completed'], total=total, successes=result['successes'])
            finally:
                env.close()
    except Exception as exc:
        save_result(output, config, episodes, 'error', exc)
        raise
    result = save_result(output, config, episodes, 'cancelled' if _STOP else 'complete')
    emit('complete', status=result['status'], completed=result['completed'], total=total, successes=result['successes'], result_path='result.json')
    return result


def preview_scene(config):
    """Render both policy cameras at reset without loading a model or stepping it."""
    import base64
    import cv2
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.configs import LiberoEnv as EnvConfig
    from lerobot.envs.utils import preprocess_observation
    import libero.libero as libero_module
    paths = libero_asset_paths(get_libero_path)
    libero_module._assets_path_cache = str(paths['assets'])
    suite = benchmark.get_benchmark_dict()[config['suite']]()
    mapping = {'agentview_image': 'image', 'robot0_eye_in_hand_image': 'wrist_image'}
    cfg = EnvConfig(task=config['suite'], camera_name_mapping=mapping,
                    observation_height=256, observation_width=256)
    pre, _ = cfg.get_env_processors()
    env = LiberoEnv(task_suite=suite, task_id=config['task_id'], task_suite_name=config['suite'],
                   episode_length=1, obs_type='pixels_agent_pos', observation_height=256,
                   observation_width=256, camera_name_mapping=mapping)
    try:
        raw, _ = env.reset(seed=config['seed'])
        def batch(value):
            return {k: batch(v) for k, v in value.items()} if isinstance(value, dict) else np.expand_dims(value, 0)
        observation = pre(preprocess_observation(batch(raw)))
        images = []
        for key, value in observation.items():
            if key.startswith('observation.images.'):
                rgb = (value[0].cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                if not ok: raise RuntimeError('Could not encode LIBERO preview')
                images.append({'camera': key, 'image': base64.b64encode(encoded).decode()})
        return {'ok': True, 'images': images, 'task_name': suite.get_task(config['task_id']).name}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--output')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    config = {}
    output = None
    try:
        if args.catalog:
            with tempfile.TemporaryDirectory(prefix='libero-catalog-') as folder:
                bootstrap(folder)
                print(json.dumps(catalog()), flush=True)
            return 0
        if not args.config or not args.output:
            parser.error('--config and --output are required for evaluation')
        config = json.loads(Path(args.config).read_text())
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        bootstrap(output)
        os.chdir(output)  # Simulator scratch/log files stay inside this run.
        if args.preview:
            print('LIBERO_PREVIEW ' + json.dumps(preview_scene(config)), flush=True)
            return 0
        run(config, output)
        return 0
    except Exception as exc:
        if args.catalog:
            print(json.dumps({'ok': False, 'suites': [], 'error': str(exc), 'runtime': {'available': False, 'python': sys.executable}}), flush=True)
        else:
            if output is not None and not (output / 'result.json').exists():
                save_result(output, config, [], 'error', exc)
            emit('error', error=str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
