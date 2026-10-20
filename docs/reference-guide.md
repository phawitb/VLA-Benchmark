# VLA-Benchmark

A local web application for collecting robot demonstrations, preparing LeRobot datasets, training robot policies on an SSH GPU server, and evaluating policies on a real robot.

This repository's `main` branch is the **no-LIBERO version**. It includes real-robot evaluation and Camera Dropout, but no LIBERO simulator, simulator installation, or LIBERO page.

The interface is organized around this workflow:

**Setup Environments → Data Collecting → Datasets → Training → Evaluate**

You can also start with an existing dataset and skip robot setup and collection until you are ready to run hardware.

## Contents

- [1. What you need](#1-what-you-need)
- [2. Install and open the website](#2-install-and-open-the-website)
- [3. Install the robot runtime](#3-install-the-robot-runtime)
- [4. First-time robot setup](#4-first-time-robot-setup)
- [5. Record a dataset](#5-record-a-dataset)
- [6. Import, inspect, and edit datasets](#6-import-inspect-and-edit-datasets)
- [7. Data augmentation and Camera Dropout](#7-data-augmentation-and-camera-dropout)
- [8. Configure a GPU server and train](#8-configure-a-gpu-server-and-train)
- [9. Download and evaluate a policy](#9-download-and-evaluate-a-policy)
- [10. Challenge Environment and Camera Dropout](#10-challenge-environment-and-camera-dropout)
- [11. Files, configuration, and backups](#11-files-configuration-and-backups)
- [12. Troubleshooting](#12-troubleshooting)
- [13. Development and verification](#13-development-and-verification)

## 1. What you need

| Activity | Required |
| --- | --- |
| Open the UI, import datasets, inspect videos, generate augmented datasets | Python 3.12, the application requirements, FFmpeg for video processing, and disk space |
| Download or publish datasets/models | Hugging Face account and access to the relevant repository |
| Collect demonstrations | Calibrated SO-101 follower and leader arms, USB serial connections, cameras, and LeRobot hardware dependencies |
| Train through this UI | An SSH-accessible Linux GPU server with compatible PyTorch/CUDA, LeRobot, Conda, and `tmux` |
| Evaluate on a robot | Follower arm, cameras matching the policy's input features, a local policy checkpoint, and policy-specific dependencies |

The local reference environment uses macOS, Conda environment `lerobot`, and Python 3.12. The GPU training workflow uses Linux shell commands. Native Windows hardware operation has not been verified by this project; opening the UI alone does not establish hardware compatibility.

**The web dependencies and the robot dependencies are separate.** Installing `requirements.txt` starts the website; it does not install LeRobot, PyTorch, motor drivers, or every policy's dependencies.

Keep the robot's workspace clear before using controls that move it. `Start Teleop`, joint sliders, `Home (all 0°)`, `Move To Reset`, `Test Inference`, and `Start Episode` can command real hardware.

## 2. Install and open the website

Install [Git](https://git-scm.com/downloads) and [Miniforge](https://github.com/conda-forge/miniforge) or Miniconda first. Open a terminal where you want to keep the project.

### 2.1 Clone the application

```bash
git clone https://github.com/zhaw-physical-ai/VLA-Benchmark.git
cd VLA-Benchmark
git switch main
```

These instructions refer to `main` in **zhaw-physical-ai/VLA-Benchmark**. Other forks may use different branch names.

### 2.2 Create the environment and install dependencies

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
python -m pip install --upgrade pip
python -m pip install -r requirements.txt huggingface_hub
conda install -c conda-forge ffmpeg -y
```

If you already have an environment named `lerobot`, activate it instead of recreating it. Install FFmpeg even if your particular LeRobot video backend does not require it: this application invokes the `ffmpeg` executable when editing and augmenting videos.

Check that the environment is active:

```bash
python -c "import sys; print(sys.executable)"
ffmpeg -version
```

The Python path should point into the `lerobot` environment. FFmpeg should print its version rather than `command not found`.

### 2.3 Start the server

Run from the repository root, where `main.py`, `config.yaml`, and `static/` are located:

```bash
conda activate lerobot
python -m uvicorn main:app --host 127.0.0.1 --port 8080
```

Open **[http://localhost:8080/](http://localhost:8080/)** in your browser. The first page is **Setup Environments → 1. Config**.

Keep the terminal open while using the application. Press **Ctrl+C** in that terminal to stop the server. Repeat the activation and startup commands whenever you open a new terminal.

If port 8080 is already occupied:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8081
```

Then open **[http://localhost:8081/](http://localhost:8081/)**. The port follows the startup command; it is not a separate application setting.

`python main.py` is also supported and uses port 8080, but binds to all network interfaces. The localhost-only command above is the preferred local startup method. The application has no built-in user authentication; do not publish it as an unauthenticated public service.

### 2.4 First use without a robot

To check the installation without moving hardware:

1. Open **Datasets** in the left sidebar.
2. Click **Import Local** and choose an existing dataset, or use **Download from HF** after logging into Hugging Face.
3. Select the dataset in the dropdown and click an episode.
4. Check that videos and **Joint Trajectories** appear.
5. Open **Augment** to inspect the available techniques and previews.

An empty dataset dropdown after a fresh clone is normal. Datasets and model weights are not included in Git.

## 3. Install the robot runtime

Skip this section if you only need the website and offline dataset tools.

### 3.1 LeRobot version and installation

This application imports LeRobot APIs directly. Its development environment contains an editable LeRobot checkout at upstream commit [`8194897994be649cc85405fe54eb9ede751886b1`](https://github.com/huggingface/lerobot/commit/8194897994be649cc85405fe54eb9ede751886b1), reporting version `0.5.2`. That environment also has local LeRobot changes; those changes are **not bundled in this repository**. There is no complete environment lockfile or clean-install hardware certification. The following pinned upstream checkout is a reproducible starting point, rather than a claim that all robots and checkpoints have been tested on an unmodified installation.

Do not try `pip install lerobot==0.5.2`: the development version string is not a published PyPI release. Do not silently upgrade an existing working robot environment to the latest LeRobot version.

With the `lerobot` environment active, install the upstream checkout next to the application:

```bash
cd ..
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 8194897994be649cc85405fe54eb9ede751886b1
python -m pip install -e '.[core-scripts,feetech,training]'
cd ../VLA-Benchmark
python -m pip install -r requirements.txt
```

The extras install the collection/teleoperation scripts, Feetech motor dependencies, dataset tools, and training dependencies. For additional policy families, run the corresponding command **inside the LeRobot source directory**:

```bash
# SmolVLA dependencies
python -m pip install -e '.[smolvla]'

# pi-family dependencies, if you intend to use those checkpoints
python -m pip install -e '.[pi]'
```

ACT is the simplest initial policy path to verify. SmolVLA and pi-family policies can download additional pretrained models and may require model access approval and substantially more memory. The pi0/pi05 paths are implemented in the application; this README does not certify arbitrary checkpoints or policy versions.

On a CUDA GPU server, first install the PyTorch build appropriate for that server using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install LeRobot and verify CUDA on the server:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For NVIDIA GPU training, the last line should be `True`.

### 3.2 Adjust the local Conda executable paths

**Do this before recording or teleoperation on a new machine.** Several commands in `main.py` currently refer to the original development environment:

```text
/opt/miniconda3/envs/lerobot/bin/
```

Activating a different Conda installation does not replace those absolute paths. Check your executable location:

```bash
python -c "import sys; from pathlib import Path; print(Path(sys.executable).parent)"
```

If it differs, stop the application, activate your intended robot environment, and run this one-time local adaptation from the application root:

```bash
python - <<'PY'
from pathlib import Path
import sys

source = Path('main.py')
old = '/opt/miniconda3/envs/lerobot/bin/'
new = str(Path(sys.executable).parent) + '/'
text = source.read_text()
print(f'Replacing {text.count(old)} executable-path references with {new}')
source.write_text(text.replace(old, new))
PY
```

Restart the application afterward. This changes your local `main.py`; review that local modification before a Git update. It does not change the remote GPU server's Conda environment, which is configured separately in the UI.

Check the robot commands before opening hardware:

```bash
command -v lerobot-record
command -v lerobot-teleoperate
command -v lerobot-calibrate
python -c "from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig; from lerobot.datasets import LeRobotDataset; print('Robot and dataset imports OK')"
```

### 3.3 Configure and calibrate the arms

Complete the official [SO-101 assembly, motor setup, and calibration guide](https://huggingface.co/docs/lerobot/so101) before using the application. The **Control** page is not a substitute for motor setup or calibration.

Find the serial ports:

```bash
lerobot-find-port
```

Follow its instructions to identify each arm. macOS typically uses `/dev/tty.usbmodem...`; Linux commonly uses `/dev/ttyACM...` or `/dev/ttyUSB...`. Use the ports found on your machine, not the values committed in `config.yaml`.

Example calibration commands, after replacing the port placeholders:

```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=YOUR_FOLLOWER_PORT \
  --robot.id=my_follower

lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=YOUR_LEADER_PORT \
  --teleop.id=my_leader
```

Use these same IDs in the UI so LeRobot can find the corresponding calibration files. On Linux, ensure your account has permission to access the serial devices; log out and back in after changing group membership.

## 4. First-time robot setup

### 4.1 Setup Environments → 1. Config

Fill in these fields:

| Field | What to enter |
| --- | --- |
| **Robot Port** | Serial port of the follower arm |
| **Robot ID** | The ID used when calibrating the follower |
| **Teleop Port** | Serial port of the leader arm |
| **Teleop ID** | The ID used when calibrating the leader |
| **Teleop Type** | Normally `so101_leader` for this workflow |
| **Top Camera Index** | OpenCV index of the overhead camera |
| **Wrist Camera Index** | OpenCV index of the wrist camera |
| **External3 / External4 Camera Index** | Enable only when those cameras are connected |

1. Click **Check Cameras**.
2. Match the returned camera images to their intended names. Camera indices are machine-specific and can change after reconnecting USB devices.
3. Correct any swapped top/wrist indices.
4. Click **Save Config** and wait for **Saved!**.

The setup UI exposes top, wrist, external3, and external4. The Camera Dropout logic uses the dataset or evaluation camera list and is not hard-coded to two views.

### 4.2 Hugging Face login

In **Setup Environments → 1. Config → Hugging Face**:

1. Create a token at [Hugging Face token settings](https://huggingface.co/settings/tokens).
2. Give it read access for downloads, and appropriate write access if you will publish datasets or models.
3. Paste it into **HF Token** and click **Login**.
4. Check that the account is recognized.

Set `hf_repo_name` in your local `config.yaml` to your own Hugging Face username or organization. The Data Collecting page derives its `repo_id` from this namespace and the dataset name. Reopen the page after changing the configuration.

Use **Login** rather than relying only on an environment variable: different application features read credentials from `.env`, saved configuration, or the Hugging Face cache. Never commit your saved tokens, SSH password, or W&B key. In particular, `config.yaml` is tracked by Git and can contain credentials after using the UI.

### 4.3 Verify, test teleoperation, and use manual control

1. Open **2. Verify** and click **Run All Checks**.
2. Resolve missing serial ports, camera failures, missing `lerobot-record`, and disk-space errors before recording.
3. Open **3. Test Mimic**, click **Start Teleop**, and gently move the leader.
4. Confirm that the follower mirrors the intended movement and that both camera views are correct.
5. Click **Stop Teleop** before starting another operation.

**4. Control → Connect** provides manual joint controls. **Read Positions** refreshes joint values; sliders command movements. **Home (all 0°)** commands all joints toward zero and should only be used when that motion is appropriate for your robot and workspace.

Do not run collection, teleoperation, and evaluation against the same hardware simultaneously, including from another application instance.

## 5. Record a dataset

1. Open **Data Collecting**.
2. Enter a new **Dataset Name**, for example `pick_cube_v1`.
3. Check the generated `repo_id` and local `root` shown below it.
4. Enter a clear **Task Description**, such as `pick up the cube and place it in the bowl`.
5. Set **Number of Episodes**, **Encoder Threads**, and each camera's recording settings.
6. Choose whether to enable **Push to Hugging Face Hub after recording**.
7. Review **Resume previous recording session** carefully.
8. Click **Start Recording** and monitor the **Log**, episode progress, and cameras.
9. Demonstrate the task with the leader arm. Follow the active LeRobot recorder's prompts for episode boundaries and resets.
10. Click **Stop & Finish** to end the recording process, then wait for video encoding/finalization.
11. Open **Datasets → Refresh** and select the resulting dataset.

**Important: when Resume is disabled, starting a recording can delete an existing dataset at the selected local root before recording a replacement.** Use a new dataset name for a fresh experiment, or enable Resume deliberately for a compatible existing dataset.

The website currently starts `lerobot-record`; it does not expose the interactive stdin controls of `collect_python.py`. Do not assume that arrow keys pressed in the browser control recording. If the recorder is waiting for keyboard input, stop the browser-launched recorder, use **Command → Copy**, and run that generated command in an interactive terminal with `lerobot` activated. Do not run both recorders together. The installed LeRobot version determines its keyboard controls and episode timing.

## 6. Import, inspect, and edit datasets

### 6.1 Import or download

**Import Local** copies a dataset directory into the application's `data/` directory. Select the dataset root containing `meta/info.json`, not an individual MP4 file. The import workflow also supports a parent folder containing datasets. A path entered in the browser refers to the computer running the Python server.

Alternatively, use **Download from HF**, enter a dataset repository ID such as `your-account/pick_cube_v1`, choose a local name, and wait for download completion. Click **Refresh** afterward if the dropdown has not updated.

The processing code is built around the LeRobot v3-style structure:

```text
data/pick_cube_v1/
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   └── episodes/chunk-000/file-000.parquet
├── data/chunk-000/file-000.parquet
└── videos/
    ├── observation.images.top/chunk-000/file-000.mp4
    └── observation.images.wrist/chunk-000/file-000.mp4
```

Chunk/file indices can differ. Merely having `meta/info.json` is not sufficient for all tools: the frame tables, episode metadata, camera feature keys, and referenced videos must agree. Older LeRobot formats may need conversion before using these workflows.

### 6.2 Explore

1. Select a dataset from the dropdown.
2. Read the episode/frame counts, FPS, robot type, and task.
3. Click an episode in the **Episodes** column.
4. Use **Play** and the timeline to inspect synchronized camera views.
5. Inspect **Joint Trajectories** for unexpected jumps or stationary sections.

### 6.3 Editing and export buttons

| Button | Purpose / effect |
| --- | --- |
| **Edit Name** | Rename the dataset directory and associated augmentation metadata |
| **Edit Task** | Change task text; selected-episode editing is available through **Select → Task** |
| **Delete Beginning** | Create a new dataset with a specified initial duration removed |
| **Remove Camera** | Create a new dataset without one camera feature; this removes that view rather than replacing it with black images |
| **Augment** | Create a new dataset with selected augmentation techniques |
| **Remove Idle** | Create a new dataset with detected idle frames removed |
| **Random Sample** | Create a new dataset containing a subset of episodes |
| **Combine** | Combine selected compatible datasets into a new dataset |
| **Push to HF** | Upload the selected dataset to the chosen Hugging Face repository |
| **Select → Del** | Delete selected episodes from the selected dataset |
| **Delete Dataset** | Delete the selected local dataset |

Back up the source before using in-place edits or deletion. A new output name is required for tools that create a derived dataset. Verify matching camera features, dimensions, action/state structure, and FPS when combining datasets.

## 7. Data augmentation and Camera Dropout

### 7.1 Create an augmented dataset

1. Open **Datasets**, select a source dataset, and click **Augment**.
2. Set **Target Episodes**. This is the total output count, not the number of extra episodes added to the source.
3. Enter a unique **New Dataset Name**, such as `pick_cube_v1_aug40`.
4. Choose **Random** or **Fix** and an **Intensity (1–5)**.
5. Enable only the **Techniques** you want.
6. Inspect the original/augmented image pairs in **Preview**.
7. Click **Start Augmentation** and wait for completion.
8. Select the new dataset and inspect several episodes before using it for training.

**Random** randomly chooses source episodes and samples enabled effects. **Fix** distributes copies across source episodes and exposes fixed effect values. Except for Camera Dropout, the first output occurrence of a source episode is retained without those effects. Camera Dropout follows its own schedule across output episodes, including those source-copy episodes.

| Technique | What changes |
| --- | --- |
| **Camera Shift** | Perspective, affine transforms, and rotation |
| **Light & Quality** | Brightness, contrast, saturation, color jitter, shadow, noise, and blur |
| **Random Occlusion** | A black rectangle covering part of an image |
| **Robot Noise** | Enabled action/state perturbations and start trimming |
| **Language Variation** | Enabled variations of task text |
| **Camera Dropout** | Entire camera views become black for a whole episode |

Augmentation creates new files; the source dataset is retained. It needs additional disk space and can take time because videos are decoded and re-encoded.

### 7.2 Configure Camera Dropout

The **Camera Dropout** card is the last card under Techniques.

1. Check **Camera Dropout**.
2. Set **Episodes with dropped cameras (%)**: the proportion of output episodes that use only some views.
3. In **Camera pool for dropout episodes**, select the cameras eligible to remain active.
4. Set **Active cameras per episode** to the number retained from that pool.
5. Choose **Rotate evenly** or **Random subset per episode**.

On a dropout episode, cameras outside the active set are black for every frame. Unchecked pool cameras are always black **on dropout episodes**. The other episodes retain all cameras. Camera feature names, video dimensions, and frame structure are retained, so a trained policy still receives its expected camera inputs.

The default percentage follows Intensity:

| Intensity | Default episodes with dropped cameras |
| --- | --- |
| 1 | 10% |
| 2 | 30% |
| 3 | 50% |
| 4 | 70% |
| 5 | 90% |

You can enter another percentage manually. **Changing Intensity resets the percentage to its default.** Changing unrelated controls preserves the manual percentage.

Example: create 40 output episodes from a dataset containing `top` and `wrist`:

- **Episodes with dropped cameras (%)**: `50`
- Selected camera pool: `top`, `wrist`
- **Active cameras per episode**: `1`
- Selection: **Rotate evenly**

The output has **20 episodes with both cameras** and **20 dropout episodes**. Among dropout episodes, the active view alternates between top-only and wrist-only. Dropped views are black, not absent.

With four cameras, selecting all four and keeping two rotates a group of two views; Random chooses subsets from the pool. Rotate evenly cycles adjacent groups in pool order, rather than enumerating every possible combination. At least one view must remain active. Keeping all available views means no view is actually blacked out.

Percentages are rounded to the nearest whole episode. For example, 50% of 3 output episodes gives 2 dropout episodes and 1 all-camera episode. Use the displayed episode counts when planning a small experiment.

## 8. Configure a GPU server and train

Training in the UI is submitted to an SSH server. It does not start a local training job merely because your laptop has a GPU.

### 8.1 Prepare the server

On your Linux GPU server:

1. Install the correct NVIDIA driver and PyTorch/CUDA build.
2. Install Conda and create a Python 3.12 environment named `lerobot`.
3. Install the compatible LeRobot checkout and the `training` extra, plus policy-specific extras as needed, as described above.
4. Install `tmux` and create a working directory.

```bash
sudo apt update
sudo apt install -y tmux
mkdir -p ~/vla_benchmark_ws/logs
conda activate lerobot
command -v lerobot-train
python -c "import torch; print(torch.cuda.is_available())"
```

Make sure the local computer can SSH to the server. SSH key authentication is recommended. Password-based authentication uses the local `sshpass` executable and will fail if it is unavailable.

### 8.2 Configure the connection in the application

In **Setup Environments → 1. Config → GPU Server (SSH)**:

1. Enter **SSH Host**, **SSH User**, and **SSH Port**.
2. Leave **Password** blank for SSH-key authentication, or supply the server password if using that mode.
3. Set **Conda Env** to the environment installed on the server.
4. Set **Working Directory** to an absolute server path, for example `/home/your-user/vla_benchmark_ws`.
5. Enter **W&B API Key** if you use W&B logging.
6. Click **Save Config**.
7. Open **2. Verify → GPU Server Connection → Test Connection** and inspect the log.

**Fix Environment** attempts to install or configure software on the server. Review its log; it is not a substitute for validating CUDA or policy-specific dependencies. The UI also provides a **Manual Setup Guide**.

### 8.3 Submit training

1. Ensure the training dataset is published on Hugging Face and accessible from the server. Use **Datasets → Push to HF** for a locally generated dataset.
2. Open **Training → New Training**.
3. Select the dataset.
4. Choose **Policy Type** and set **Steps**, **Batch Size**, and **Learning Rate**.
5. Set a unique **Job Name** so experiments do not reuse an output directory.
6. Review policy-specific options such as **Chunk Size**, **N Action Steps**, or the vision backbone.
7. Use **Show SSH Command** to inspect the actual command.
8. Click **Start Training**.
9. Watch the **GPU Server**, **Queue**, and log panels. Use **History → Refresh** to review jobs.

The server worker runs jobs through `tmux`, waits for GPU memory availability, and writes output below `<Working Directory>/outputs/train/<Job Name>`. Reusing a job name can replace an existing output directory when the worker prepares the job. **Kill All Jobs** affects the server's training queue/processes; it is not a pause button for one experiment.

After training, publish or copy a usable checkpoint back to the local application before evaluation. A successful training job does not by itself guarantee that its checkpoint is available in the local `models/` directory.

## 9. Download and evaluate a policy

Evaluation runs the policy locally and commands the connected follower arm. Use a checkpoint trained for the robot's action/state layout and the same camera names and ordering conventions.

### 9.1 Make a checkpoint available

1. Open **Evaluate → Run**.
2. Click **Refresh** to discover models.
3. If needed, click **Load All Models from HF**, select the relevant models, and click **Download Selected**.
4. Wait for completion, then select an executable local model.

Remote-only entries may be shown but cannot be executed until downloaded. The local model registry expects a directory directly under `models/` with both `config.json` and `train_config.json`, together with the actual weights and processor files required by that policy:

```text
models/my_policy/
├── config.json
├── train_config.json
├── model.safetensors
└── ... policy-specific files
```

The exact weight and processor filenames depend on the policy. A folder with only the two JSON files is not a usable checkpoint. Models requiring unconfigured camera names are marked unavailable.

### 9.2 Configure and run episodes

After selecting a model, the evaluation cards become visible:

1. Enter a **Session Name** for the experiment.
2. In **Evaluation Config**, enter the **Task Description** and **Target Runs**.
3. Set **Run Duration (sec)**, **Cameras**, **FPS**, **Width**, and **Height** to match your experiment and policy.
4. Leave **Chunk Size** and **N Action Steps** at their defaults unless you intend to override the checkpoint's settings.
5. Configure **Challenge Environment** if needed; otherwise leave it disabled for a baseline.
6. Check **Record Video** if you want episode recordings.
7. Review the **Reset Position** section before starting.
8. Place the robot and objects appropriately, clear the movement area, and click **Start Episode**.
9. Monitor the countdown, robot, and **Log**. Use the stop control if the episode needs to end early.
10. When the episode ends, record the outcome using **1**, **2**, **3**, or **Fail**.
11. Reset the scene and repeat until Target Runs is reached.

**1 / 2 / 3** mean success after the corresponding number of attempts within the episode; **Fail** records an unsuccessful episode. Success is manually recorded by the operator, not inferred automatically from the camera images.

Completed results update the progress and success rate. Sessions are saved locally and can be reopened from **Sessions**. **Export JSON** exports results; the **Results** tab aggregates saved sessions for comparison.

**Test Inference** is a separate continuous test path that runs until stopped. It is not equivalent to completing and scoring one bounded evaluation episode.

### 9.3 Reset position and worker lifecycle

**Capture Current** stores joint positions. **Torque Off** allows repositioning when supported; support the arm as needed. **Torque On** enables motor holding/control. **Move To Reset** commands the stored reset position, and **Clear** removes it.

A saved reset position can also be applied automatically when an evaluation episode starts. Choose a pose and path that are clear of obstacles. Stop teleoperation before evaluation. The inference worker can retain a loaded model between episodes to reduce startup time; a warm model is not proof that a serial port or camera is free for another process.

## 10. Challenge Environment and Camera Dropout

### 10.1 Open the challenge settings

1. Select a local model in **Evaluate → Run**.
2. Enable **Challenge Environment**.
3. Click **Setup & Preview**.
4. Choose **Random** or **Fix**, then enable the desired **Techniques**.
5. Select a **Source dataset** for preview.
6. Set **Challenge episode** to inspect the conditions assigned to that evaluation episode.
7. Inspect the original/challenge pairs and click **Done** when finished.

Settings save automatically. The dialog contains **Camera Shift**, **Light & Quality**, **Occlusion**, **Robot Noise**, and **Camera Dropout** as the last technique. In Fix mode, the applicable technique values are directly adjustable. Preview does not move the robot.

### 10.2 Use Camera Dropout during evaluation

The controls have the same meaning as in Data Augmentation:

- **Episodes with dropped cameras (%)** is the proportion of evaluation episodes that lose some views.
- **Camera pool for dropout episodes** selects views eligible to remain active.
- **Active cameras per episode** selects how many views stay visible.
- **Rotate evenly / Random subset per episode** selects the view scheduling method.

The available views come from the configured evaluation cameras. At least **two active evaluation cameras** are required. Do not choose a single-camera evaluation configuration while leaving Camera Dropout enabled. Dropout retains the model's camera inputs and sends black images for dropped views, rather than disconnecting or deleting a camera.

Intensity 1–5 sets default dropout percentages to **10%, 30%, 50%, 70%, 90%**. You can override the percentage; changing Intensity resets it. Other enabled challenge techniques continue to affect retained views.

Example for a controlled comparison:

1. Set **Target Runs** to `10`.
2. Select both top and wrist in the evaluation camera configuration.
3. Enable **Camera Dropout**, set the percentage to `50`, retain `1` active view, and select **Rotate evenly**.
4. Preview challenge episodes 1 onward.
5. Run and score the 10 episodes normally.

This produces **5 all-camera episodes and 5 dropout episodes**, with the active camera rotating across the dropout episodes. The ratio is calculated from Target Runs, rounded to whole episodes, and distributed across the run. Image pairs identify each view as **Active** or **Black**.

The same challenge configuration, episode number, camera pool, and Target Runs produce repeatable view selection across models. Random subsets are seeded by episode. Keep those settings identical when comparing policies. Changing Target Runs or the pool partway through a session can change the schedule for subsequent episodes.

Use a preview dataset with camera names matching the evaluation setup. A two-camera preview dataset cannot demonstrate a missing third-camera input from a four-camera configuration. The selected view remains black for the whole episode, and per-episode challenge parameters are stored with the result for inspection.

## 11. Files, configuration, and backups

| Location | Contents |
| --- | --- |
| `main.py` | FastAPI application and server entry point |
| `static/index.html` | Main browser UI |
| `static/eval_challenge.js` | Evaluation challenge controls and preview |
| `augmentation_core.py` | Shared transformations and camera scheduling |
| `infer_python.py` | Local policy inference worker |
| `vla_execution.py` | Worker process lifecycle and command construction |
| `model_registry.py` | Local/remote policy discovery and availability checks |
| `config.yaml` | Local hardware, data collection, SSH, and storage settings |
| `.env` | Optional local service credentials; use `.env.example` as a template |
| `data/<dataset>/` | Local datasets |
| `augment_configs/<dataset>.json` | Augmentation configuration and per-episode source/effect metadata |
| `models/<model>/` | Downloaded local policy checkpoints |
| `data/eval_sessions/` | Saved evaluation settings, reset position, and sessions |
| `data/eval_videos/` | Recorded evaluation videos |

The committed `config.yaml` is a starting configuration containing machine-specific example values. Replace ports, IDs, camera indices, Hugging Face namespace, and SSH details for your own setup.

For optional environment-based configuration:

```bash
cp .env.example .env
```

Replace placeholders only for services you actually use. A Gemini API key is not required to open the UI or use Camera Dropout. Prefer the Hugging Face and GPU Server login/settings controls described above for their respective workflows.

Back up **datasets, `augment_configs/`, models, evaluation sessions, videos, and calibration files** separately from the Git repository. `data/` and `models/` are ignored by Git, so cloning or pulling the repository does not restore them. Augmentation metadata is stored outside the generated LeRobot dataset; keep the matching JSON when transferring a dataset if you want the UI to explain its augmentation history.

Before any commit, inspect `git status` and ensure that local credentials in `config.yaml`, `.env`, or generated commands/logs are not included.

### Updating the application

Stop the server before updating. Preserve local configuration changes first:

```bash
git status --short
git pull --ff-only
conda activate lerobot
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8080
```

If Git reports local changes, reconcile them before retrying; do not discard local settings blindly. Reload the browser after an update. Use a hard refresh if it still shows an older UI.

## 12. Troubleshooting

| Symptom | What to check |
| --- | --- |
| `conda: command not found` | Install/initialize Conda for your shell, reopen the terminal, then activate `lerobot` |
| `No module named fastapi` or another web dependency | Activate the correct environment and run `python -m pip install -r requirements.txt` |
| `No module named lerobot`, missing motor SDK, or a policy import error | Install the robot runtime/extras; inspect the LeRobot version and API compatibility, not just the website requirements |
| `Address already in use` | Stop the existing server or start on another port such as 8081; open the matching URL |
| Website returns `Not Found` | Open `/` on the server/port running this version; verify that an older or unrelated server is not occupying the port |
| `Directory 'static' does not exist` | Start the server from the application root |
| `ffmpeg` missing or augmentation/video encoding fails | Install FFmpeg in the active environment and check `ffmpeg -version` and available encoders |
| Verify reports `/opt/miniconda3/.../lerobot-record` missing | Apply the Conda executable-path adaptation in section 3.2 and restart |
| Camera image is black or top/wrist are swapped | Check indices with **Check Cameras**, OS camera permissions, USB connections, and whether another process owns the device |
| Serial port not found / permission denied | Re-identify the port, check user permissions and calibration ID, and stop other hardware clients |
| Robot or camera is busy | Stop teleoperation, recording, or another inference/application instance before starting the next operation |
| Dataset is listed but videos or augmentation fail | Check the v3-style frame/episode metadata, `dtype: video` camera features, timestamps, and referenced video files |
| No models / Hugging Face access denied | Verify login, `hf_repo_name`, repository permissions, and required gated-model access |
| Model is unavailable or checkpoint cannot load | Download the complete checkpoint; check policy type, `config.json`, `train_config.json`, weights/processors, and configured camera names |
| Challenge Preview cannot match a camera | Select a dataset with the same views and update the camera pool; a selected view may be absent from the source dataset |
| Dropout percentage does not match a small run exactly | Percentages round to whole episodes; check the displayed full/dropout counts |
| Dropout is enabled but no image becomes black | Ensure at least one camera is outside the retained active set and the previewed episode is a dropout episode |
| SSH connection fails | Verify host/user/port, key authentication or `sshpass`, and that the remote environment and working directory exist |
| Training remains pending | Inspect the remote queue/log and GPU-memory availability; verify `tmux`, `lerobot-train`, and CUDA |
| GPU out of memory | Reduce batch size or use a smaller policy; review policy-specific memory settings |
| Recording appears stuck at an input prompt | Use the interactive-terminal recording fallback in section 5 and the installed recorder's controls |
| Changes appear lost after reopening | Verify the same server port/project directory is in use and check local saved config/session files |

When reporting a problem, include the page/button used, the relevant log/error, OS, Python/LeRobot versions, policy type, camera names, and whether the failure occurs before or after hardware connection. Remove credentials from logs and copied commands.

## 13. Development and verification

Install test dependencies in the active Python environment, and install Node.js if you want to run the UI tests:

```bash
python -m pip install pytest httpx
python -m pytest -q
node --test tests/*.cjs
```

The no-LIBERO Camera Dropout revision was checked with **52 Python tests and 20 JavaScript tests**. Tests cover dataset generation and video blackout, camera scheduling and quotas, challenge preview, saved settings, inference observation transforms, and worker lifecycle. These are automated checks; they do not replace an end-to-end test with your physical robot and checkpoint.

Read-only server checks while the application is running:

```bash
curl -f http://localhost:8080/ -o /dev/null
curl -f http://localhost:8080/api/datasets
```

The FastAPI schema is available at `/openapi.json` and API documentation at `/docs`. The API includes endpoints that write files and command hardware; use the checks above for a simple non-actuating smoke test.

Upstream references:

- [LeRobot installation](https://huggingface.co/docs/lerobot/installation)
- [SO-101 assembly and calibration](https://huggingface.co/docs/lerobot/so101)
- [Pinned upstream LeRobot reference](https://github.com/huggingface/lerobot/commit/8194897994be649cc85405fe54eb9ede751886b1)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Hugging Face access tokens](https://huggingface.co/settings/tokens)

Upstream documentation may describe newer releases. Match installation instructions and dependencies to the checkout and checkpoint you actually use.
