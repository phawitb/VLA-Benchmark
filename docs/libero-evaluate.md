# Libero-Evaluate

The Run Model sidebar includes a simulation-only LIBERO workspace. It does not
connect to the physical robot or its cameras.

Choose a compatible checkpoint or Hugging Face model ID, suite, tasks, episodes,
maximum steps, seed and device. The policy loads once for the selected tasks.
The page shows episode progress, success rate, current step, previews, logs and
videos. Videos use H.264/yuv420p with MP4 fast-start for browser playback and are
published after encoding finishes. Cancel keeps partial results; cancelled
episodes do not count as failures.

Challenge setup opens in a centered dialog with settings on the left and real LIBERO
before/after camera previews on the right. The Enable challenge checkbox appears
both on the main page beside Setup and in the dialog, with synchronized state. Random/Fix controls and slider ranges follow
Evaluate's shared field/group schema. Random shows each selected technique's
minimum/maximum range at the current intensity; Fix shows sliders. Preview resets
the selected suite/task at the run seed without loading a policy, displays both
cameras, and caches clean frames for responsive slider updates. Closing the dialog
retains the settings.

Data Augmentation, Evaluate, and LIBERO all use augmentation_core.py for camera,
lighting, occlusion transforms and parameter generation. Dataset row processing
and physical robot action perturbation retain their separate domain-specific
adapters. libero_challenge.py only re-exports the shared implementation for
compatibility and contains no copied algorithms.

History stores configuration, per-task and per-episode outcomes, seeds, sampled
challenge parameters and media in data/libero_runs. Saved configurations can be
reloaded and results exported as JSON.
Delete history moves a finished run and its media to data/libero_runs/.trash;
The left History column opens in browse mode. Select reveals checkboxes, Select all,
and Delete selected; Cancel hides them and clears the selection.
Undo delete restores the deleted selection in the current page. Active runs cannot
be deleted. Per-episode applied challenge parameters are inspectable in History.

## Suites and checkpoints

The catalog comes from the installed LIBERO registry. The tested installation
provides Spatial (10), Object (10), Goal (10), Long/libero_10 (10), and LIBERO-90
(90). Broken registry entries appear separately in the API unavailable_suites list.

LIBERO needs seven Cartesian/gripper actions. SO101 six-joint checkpoints are
rejected rather than remapped. LeRobot environment and policy processors handle
observations/actions, including common image and wrist_image camera features.

Random/Fix challenges alter policy input images: perspective, affine/rotation,
brightness, contrast, saturation, HSV jitter, shadows, noise, blur and occlusion.
Simulator physics and success criteria stay unchanged. SO101 joint noise, command
trimming and language changes are not applied. Each episode saves its parameters.

## Runtime

Set LIBERO_PYTHON before starting the app to a compatible Python environment with
LeRobot, LIBERO, MuJoCo, robosuite, Torch, OpenCV and PyAV with libx264. Otherwise the service checks
for a sibling causalvla conda environment, the existing /opt/miniconda3/envs/causalvla
environment, then the server interpreter.

Optional environment settings:

- LIBERO_LEROBOT_SRC: explicit LeRobot source checkout.
- LIBERO_MODELS_DIR: additional checkpoint directory.
- LIBERO_CONFIG_PATH: existing LIBERO asset configuration directory.
- MUJOCO_GL: renderer selection on Linux; defaults to EGL.
- HF_HUB_OFFLINE=1: restrict loading to cached models.

The worker copies asset configuration into its run directory to avoid interactive
setup without rewriting the original configuration. When using a CausalVLA
LeRobot checkout, the worker also exposes its companion causal_aug package if it
is not already installed, as required by CausalVLA policy variants.
Available models uses the
same Hugging Face account and /api/hf-models endpoint as the regular Evaluate page.
Downloaded checkpoints use their local path when known; remote entries retain
their full repository ID. If HF is unavailable, local checkpoints are shown.
Explicit checkpoint paths remain supported, including the optional sibling
CausalVLA outputs/train directory.

## Validation

Run python3 -m unittest discover -s tests -p 'test_*.py' and node --test tests/*.cjs.
Actual validation used a cached CausalVLA SmolVLA checkpoint on MPS, Spatial task
0, one episode with two simulation steps, image challenges and video enabled.
Catalog, inference, progress, preview, history and media serving were exercised.
This smoke test does not measure benchmark accuracy.
The challenge effect tests exercise every image technique in Random and Fix modes
on both policy camera tensors, checking reproducibility, disabled groups, output
ranges, and preservation of state observations. New setups enable all camera and
light technique checkboxes; the master challenge switch and occlusion remain opt-in.
