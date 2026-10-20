# VLA-Benchmark

A local web app for collecting robot demonstrations, preparing datasets, training policies, and evaluating them on a real robot.

**Workflow:** Setup Environments → Data Collecting → Datasets → Training → Evaluate

This repository's `main` branch contains the **no-LIBERO** version.

## Install and start

Install Git and [Conda / Miniforge](https://github.com/conda-forge/miniforge), then run:

```bash
git clone https://github.com/zhaw-physical-ai/VLA-Benchmark.git
cd VLA-Benchmark
conda create -n lerobot python=3.12 -y
conda activate lerobot
python -m pip install -r requirements.txt huggingface_hub
conda install -c conda-forge ffmpeg -y
python -m uvicorn main:app --host 127.0.0.1 --port 8080
```

Open **[http://localhost:8080/](http://localhost:8080/)**. Keep the terminal open; press **Ctrl+C** to stop. If port 8080 is busy, change it to 8081 and open that port instead.

Already have a `lerobot` environment? Activate it instead of creating another one.

**For real hardware:** install LeRobot and the required policy/motor dependencies, then calibrate the arms. Some executable paths in `main.py` still refer to `/opt/miniconda3/envs/lerobot/bin/`; adapt them if your Conda installation is elsewhere. See the [runtime installation guide](docs/reference-guide.md#3-install-the-robot-runtime) before recording or inference. Website requirements alone do not install the robot runtime.

## 1. Set up the robot

In **Setup Environments → 1. Config**:

1. Enter the follower **Robot Port / ID** and leader **Teleop Port / ID** used during calibration.
2. Set **Top Camera Index** and **Wrist Camera Index**. Enable extra cameras if needed.
3. Click **Check Cameras**, correct swapped views, then **Save Config**.
4. For Hugging Face downloads/uploads, enter your token and click **Login**. Set `hf_repo_name` in your local `config.yaml` to your username or organization.

Then open **2. Verify → Run All Checks**. Use **3. Test Mimic → Start Teleop** to check leader/follower movement, and **Stop Teleop** when finished.

Keep the robot's movement area clear. Do not run multiple processes against the same robot or cameras.

## 2. Collect demonstrations

In **Data Collecting**:

1. Enter a unique **Dataset Name** and **Task Description**.
2. Set **Number of Episodes** and camera recording settings.
3. Review **Resume previous recording session** and the upload option.
4. Click **Start Recording**, follow the recorder's prompts, and demonstrate the task.
5. Click **Stop & Finish**, wait for finalization, then open **Datasets → Refresh**.

**Resume disabled can replace an existing dataset at the selected root.** Use a new name for a new experiment.

If recording waits for keyboard input, stop it, use **Command → Copy**, and run the generated command in an interactive terminal. Browser arrow keys do not control the recorder.

## 3. Explore and prepare datasets

Open **Datasets**:

- Use **Import Local** to copy an existing LeRobot dataset, or **Download from HF**.
- Select a dataset and an episode, then use **Play** and the timeline to inspect its videos and joint trajectories.
- Use **Random Sample**, **Combine**, **Remove Idle**, **Remove Camera**, or **Delete Beginning** to create a derived dataset.
- Use **Push to HF** when the dataset is ready for remote training.

Processing expects LeRobot v3-style metadata, Parquet frame/episode tables, and matching camera videos. Back up datasets before in-place edits or deletion.

### Data augmentation

1. Select a dataset and click **Augment**.
2. Set **Target Episodes** (the total output count), a new dataset name, **Mode**, and **Intensity**.
3. Enable the desired **Techniques** and inspect **Preview**.
4. Click **Start Augmentation** and wait for completion.

**Random** samples source episodes/effects; **Fix** distributes copies across source episodes and exposes fixed effect values. The source dataset is retained.

### Camera Dropout

The last Techniques card simulates missing camera views by making them black for the entire episode.

Example: **50% all-camera episodes, 50% alternating top-only / wrist-only**:

- Enable **Camera Dropout**.
- Set **Episodes with dropped cameras (%)** to **50**.
- Select `top` and `wrist`; keep **1** active camera per episode.
- Choose **Rotate evenly**, or **Random subset per episode** for random selection.

For 40 output episodes, this gives 20 with all views and 20 with some views blacked out. Multiple-camera datasets can retain more than one active view. At least one view must stay active.

Intensity 1–5 defaults to **10%, 30%, 50%, 70%, 90%** dropout. You can edit the percentage; changing Intensity resets it. Counts round to whole episodes.

## 4. Train a policy

Training runs on an **SSH GPU server**, not automatically on the local computer.

1. Install compatible PyTorch/CUDA, LeRobot, Conda, and `tmux` on the server.
2. Fill in **Setup Environments → 1. Config → GPU Server (SSH)** and **Save Config**.
3. Open **2. Verify → Test Connection**.
4. Publish the dataset to Hugging Face so the server can access it.
5. In **Training → New Training**, select the dataset and policy, set steps/batch size, and use a unique **Job Name**.
6. Review **Show SSH Command**, click **Start Training**, and monitor the queue/logs or **History**.

Download or copy the resulting complete checkpoint locally before evaluation.

## 5. Evaluate a policy

In **Evaluate → Run**:

1. Click **Refresh**. Use **Load All Models from HF → Download Selected** if needed.
2. Select a local model, then enter **Session Name**, task, **Target Runs**, duration, and camera settings.
3. Review the **Reset Position** and optionally enable **Record Video**.
4. Reset the scene and click **Start Episode**.
5. After the episode, record **1 / 2 / 3** for success after that many attempts, or **Fail**.
6. Repeat until Target Runs is reached. Use **Export JSON** or the **Results** tab to review outcomes.

Success is recorded manually. A saved reset position may move the robot automatically at episode start. Model checkpoints must include weights, processor files, `config.json`, and `train_config.json`, and match the configured camera names.

### Challenge Environment

Enable **Challenge Environment → Setup & Preview** to configure camera shifts, lighting, occlusion, robot noise, and **Camera Dropout**.

Select a preview dataset with matching cameras. **Camera Dropout** uses the same controls as dataset augmentation, but its percentage applies to **Target Runs**. Preview labels show **Active / Black**; preview does not move the robot. Settings save automatically.

For comparisons across models, keep the challenge settings, episode numbers, camera pool, and Target Runs identical. Evaluation Camera Dropout requires at least two active cameras.

## Common issues

| Issue | Check |
| --- | --- |
| Missing Python module | Activate `lerobot` and install the relevant web or robot dependencies |
| Camera/serial device unavailable | Check indices/ports, permissions, calibration IDs, and other running processes |
| `lerobot-record` not found | Check the absolute Conda paths in `main.py` |
| Model unavailable | Download the full checkpoint and match its required camera names |
| Training does not start | Check SSH, remote Conda, `tmux`, dataset access, and GPU memory |
| Video processing fails | Verify `ffmpeg -version`, free disk space, and dataset metadata |

Local datasets live in `data/`, models in `models/`, augmentation history in `augment_configs/`, and evaluation results in `data/eval_sessions/`. Back these up separately. **Do not commit tokens or passwords saved in local configuration files.**

For detailed installation, calibration, server setup, backups, and troubleshooting, see the [reference guide](docs/reference-guide.md).
