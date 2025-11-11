LEO Experiment
================

Quick start:

1. Create and activate the virtual environment (you already have one named `venv`):

   python3 -m venv venv
   source venv/bin/activate

2. Install requirements (recommended inside venv):

   pip install stable-baselines3 gymnasium torch tensorboard

3. Train a PPO agent on the minimal LEO env:

   python train_leo.py

Notes:
- `envs/leo_env.py` is a minimal, toy environment designed as a starting point. Extend it
  to capture your full model layer sizes, transmission timing, and energy models.
- The training script uses PPO; swap to other algorithms in Stable-Baselines3 if desired.
# Dissertation_Experiment